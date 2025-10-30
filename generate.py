import jax
import jax.numpy as jnp
import json
from config import ModelConfig
from model import ProductionTransformer
from utils import load_tokenizer, tokenize_text, detokenize_text
from trainer import Trainer
import argparse

def generate_text(model, params, tokenizer, prompt, max_len=100, temperature=1.0, use_multi_device=False):
    """Generate text from a prompt.
    
    Args:
        model: The transformer model
        params: Model parameters (may be replicated across devices)
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input text prompt
        max_len: Maximum number of tokens to generate
        temperature: Sampling temperature
        use_multi_device: Whether params are replicated across devices
    
    Returns:
        Generated text string
    """
    # Detect multi-device setup
    devices = jax.devices()
    num_devices = len(devices)
    is_multi_device = use_multi_device and num_devices > 1
    
    if is_multi_device:
        print(f"Using multi-device generation with {num_devices} devices")
    
    tokens = tokenize_text(prompt, tokenizer, model.config.max_len)
    input_ids = jnp.array([tokens])
    
    # Create generation function
    if is_multi_device:
        @jax.pmap
        def generate_step(params, input_ids, rng):
            logits = model.apply(params, input_ids, deterministic=True)
            next_token_logits = logits[0, -1, :] / temperature
            next_token = jax.random.categorical(rng, next_token_logits)
            return next_token
        
        # Note: For simplicity, use first device for generation
        # Full multi-device generation would require batching multiple prompts
        params_single = jax.tree_map(lambda x: x[0], params)
        use_pmap = False  # Fall back to single device
    else:
        @jax.jit
        def generate_step(params, input_ids, rng):
            logits = model.apply(params, input_ids, deterministic=True)
            next_token_logits = logits[0, -1, :] / temperature
            next_token = jax.random.categorical(rng, next_token_logits)
            return next_token
        params_single = params

    rng = jax.random.PRNGKey(0)
    for _ in range(max_len):
        rng, step_rng = jax.random.split(rng)
        next_token = generate_step(params_single, input_ids, step_rng)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
        if next_token == tokenizer.token_to_id("<eos>") or next_token == tokenizer.token_to_id("</s>"):
            break

    generated_tokens = input_ids[0].tolist()
    return detokenize_text(generated_tokens, tokenizer)

def main():
    parser = argparse.ArgumentParser(description="Generate text using trained model")
    parser.add_argument("--model_path", type=str, default="trained_model", help="Path to saved model")
    parser.add_argument("--prompt", type=str, default="To be or not to be", help="Prompt for generation")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--multi-device", action="store_true", help="Enable multi-device inference")
    args = parser.parse_args()
    
    # Detect devices
    devices = jax.devices()
    print(f"Detected {len(devices)} device(s): {jax.default_backend().upper()}")
    if args.multi_device and len(devices) > 1:
        print(f"Multi-device inference enabled")

    # Load config
    with open(f"{args.model_path}/config.json", "r") as f:
        config_dict = json.load(f)
    model_config = ModelConfig(**config_dict)

    # Load tokenizer
    tokenizer = load_tokenizer("char_tokenizer.json")

    # Load model
    model = ProductionTransformer(model_config)

    # Create trainer to load checkpoint
    from config import TrainingConfig
    train_config = TrainingConfig(
        batch_size=1, learning_rate=1e-3, num_epochs=1,
        max_steps=1, warmup_steps=0, weight_decay=0.01,
        gradient_clip_norm=1.0, save_steps=1000, eval_steps=1, log_steps=10
    )  # Dummy config
    trainer = Trainer(model_config, train_config)
    trainer.load_checkpoint(args.model_path)

    # Generate text
    if trainer.state is not None:
        # Check if params are replicated (multi-device training was used)
        use_multi_device = args.multi_device and trainer.use_multi_device
        generated_text = generate_text(
            model, trainer.state.params, tokenizer, 
            args.prompt, args.max_len, args.temperature,
            use_multi_device=use_multi_device
        )
    else:
        raise ValueError("Failed to load model state")
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()