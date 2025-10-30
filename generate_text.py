"""
Text Generation Script for Trained Models

This script loads a trained transformer model and generates text based on a prompt.
Compatible with checkpoints saved by train_model.py and train_optimized.py.

Usage:
    # Basic generation
    python generate_text.py --checkpoint output/best_checkpoint --prompt "Once upon a time"
    
    # With parameters
    python generate_text.py \
        --checkpoint output/best_checkpoint \
        --prompt "The quick brown fox" \
        --max-length 100 \
        --temperature 0.8 \
        --top-k 50
"""

import argparse
import json
import os
import jax
import jax.numpy as jnp
from tokenizers import Tokenizer
from orbax import checkpoint as ocp

from model import ProductionTransformer


def load_model_and_tokenizer(checkpoint_dir: str, tokenizer_path: str = None):
    """
    Load model configuration, parameters, and tokenizer.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        tokenizer_path: Path to tokenizer.json (auto-detected if None)
    
    Returns:
        Tuple of (model, params, tokenizer, config)
    """
    # Auto-detect tokenizer path
    if tokenizer_path is None:
        # Try common locations
        parent_dir = os.path.dirname(checkpoint_dir)
        candidates = [
            os.path.join(parent_dir, "tokenizer.json"),
            os.path.join(checkpoint_dir, "tokenizer.json"),
            "tokenizer.json"
        ]
        for path in candidates:
            if os.path.exists(path):
                tokenizer_path = path
                break
        
        if tokenizer_path is None:
            raise ValueError(
                f"Could not find tokenizer.json. Please specify with --tokenizer-path"
            )
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Load model config
    config_path = os.path.join(os.path.dirname(checkpoint_dir), "model_config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Model config not found at {config_path}")
    
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    from config import ModelConfig
    model_config = ModelConfig(**config_dict)
    
    # Create model
    print(f"Creating model...")
    model = ProductionTransformer(config=model_config)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_dir}")
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(checkpoint_dir)
    
    print(f"✓ Model loaded successfully!")
    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Max length: {model_config.max_len}")
    print(f"  Model size: {model_config.d_model}")
    print(f"  Layers: {model_config.num_layers}")
    
    return model, params, tokenizer, model_config


def create_generate_step(model, top_k):
    """Create JIT-compiled generation step function for a specific model and top_k."""
    
    @jax.jit
    def generate_step(params, input_ids, rng, temperature):
        """
        Single generation step (JIT-compiled for speed).
        Returns next token ID.
        """
        # Get logits for next token
        logits = model.apply(params, input_ids, deterministic=True)
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Apply top-k filtering if specified (top_k captured in closure)
        if top_k > 0:
            top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
            next_token_logits = jnp.full_like(next_token_logits, float('-inf'))
            next_token_logits = next_token_logits.at[top_k_indices].set(top_k_logits)
        
        # Sample next token
        next_token = jax.random.categorical(rng, next_token_logits)
        return next_token
    
    return generate_step


def generate_text(
    model,
    params,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng_seed: int = 42
):
    """
    Generate text from a prompt using the trained model.
    
    Args:
        model: The transformer model
        params: Model parameters
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input text prompt
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens with highest probability (0 = disabled)
        top_p: Nucleus sampling - keep top tokens with cumulative probability >= top_p
        rng_seed: Random seed for generation
    
    Returns:
        Generated text string
    """
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids = jnp.array([encoded.ids], dtype=jnp.int32)
    
    print(f"\n{'='*80}")
    print("GENERATING TEXT")
    print("="*80)
    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {len(encoded.ids)}")
    print(f"Max new tokens: {max_length}")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k if top_k > 0 else 'disabled'}")
    print(f"Top-p: {top_p if top_p < 1.0 else 'disabled'}")
    print(f"\n{'-'*80}")
    print("Generated text:")
    print(f"{'-'*80}")
    print(prompt, end="", flush=True)
    
    # Generate tokens
    rng = jax.random.PRNGKey(rng_seed)
    
    # Create JIT-compiled generation step (top_k captured in closure)
    generate_step = create_generate_step(model, top_k)
    
    # Store generated token IDs
    generated_tokens = []
    
    # Warmup JIT compilation (first call is slow, rest are fast)
    print(" ", end="", flush=True)
    rng, warmup_rng = jax.random.split(rng)
    _ = generate_step(params, input_ids, warmup_rng, temperature)
    print("\b", end="", flush=True)
    
    for i in range(max_length):
        # Generate next token (JIT-compiled, FAST!)
        rng, step_rng = jax.random.split(rng)
        next_token = generate_step(params, input_ids, step_rng, temperature)
        
        # Store token ID
        next_token_int = int(next_token)
        generated_tokens.append(next_token_int)
        
        # Decode and print incrementally (decode all generated tokens for proper spacing)
        token_text = tokenizer.decode(generated_tokens)
        # Print only the new character(s) since last iteration
        if i == 0:
            new_text = token_text
        else:
            prev_text = tokenizer.decode(generated_tokens[:-1])
            new_text = token_text[len(prev_text):]
        
        print(new_text, end="", flush=True)
        
        # Add token to sequence
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
        
        # Check for end of sequence tokens
        eos_tokens = ["<eos>", "</s>", "<|endoftext|>", "[EOS]"]
        if token_text.strip() in eos_tokens:
            break
        
        # Check if we've exceeded max length
        if input_ids.shape[1] >= model.config.max_len:
            print("\n[Max sequence length reached]", flush=True)
            break
    
    print(f"\n{'-'*80}")
    
    # Decode full sequence
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    print(f"\nGenerated {len(generated_tokens)} new tokens")
    print(f"Total tokens: {len(generated_ids)}")
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation
  python generate_text.py --checkpoint output/best_checkpoint --prompt "Once upon a time"
  
  # Creative generation (higher temperature)
  python generate_text.py --checkpoint output/best_checkpoint --prompt "The story begins" --temperature 1.2
  
  # Focused generation (lower temperature + top-k)
  python generate_text.py --checkpoint output/best_checkpoint --prompt "In conclusion" --temperature 0.7 --top-k 40
  
  # Nucleus sampling
  python generate_text.py --checkpoint output/best_checkpoint --prompt "Hello world" --top-p 0.9
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., output/best_checkpoint)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input text prompt for generation"
    )
    
    # Optional arguments
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer.json (auto-detected if not specified)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random, lower = more focused) (default: 0.8)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Keep only top k tokens with highest probability (0 = disabled) (default: 50)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling - keep top tokens with cumulative prob >= top_p (1.0 = disabled) (default: 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.temperature <= 0:
        parser.error("Temperature must be positive")
    if args.top_k < 0:
        parser.error("Top-k must be non-negative")
    if args.top_p <= 0 or args.top_p > 1.0:
        parser.error("Top-p must be in (0, 1]")
    
    print(f"\n{'='*80}")
    print(" TEXT GENERATION".center(80))
    print("="*80)
    
    # Check backend
    backend = jax.default_backend()
    print(f"\nBackend: {backend.upper()}")
    
    # Load model
    try:
        # Convert checkpoint path to absolute path (required by Orbax)
        checkpoint_path = os.path.abspath(args.checkpoint)
        model, params, tokenizer, config = load_model_and_tokenizer(
            checkpoint_path,
            args.tokenizer_path
        )
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return 1
    
    # Generate text
    try:
        generated_text = generate_text(
            model=model,
            params=params,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            rng_seed=args.seed
        )
        
        print(f"\n{'='*80}")
        print(" GENERATION COMPLETE".center(80))
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
