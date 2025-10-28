import jax
import jax.numpy as jnp
import json
from config import ModelConfig
from model import ProductionTransformer
from utils import load_tokenizer, tokenize_text, detokenize_text
from trainer import Trainer
import argparse

def generate_text(model, params, tokenizer, prompt, max_len=100, temperature=1.0):
    """Generate text from a prompt."""
    tokens = tokenize_text(prompt, tokenizer, model.config.max_len)
    input_ids = jnp.array([tokens])

    for _ in range(max_len):
        logits = model.apply(params, input_ids)
        next_token_logits = logits[0, -1, :] / temperature
        next_token = jax.random.categorical(jax.random.PRNGKey(0), next_token_logits)
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
    args = parser.parse_args()

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
        generated_text = generate_text(model, trainer.state.params, tokenizer, args.prompt, args.max_len, args.temperature)
    else:
        raise ValueError("Failed to load model state")
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()