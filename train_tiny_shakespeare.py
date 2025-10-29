import os
import requests
import json
import argparse
from tokenizers import CharBPETokenizer
from config import ModelConfig, TrainingConfig, DataConfig, CloudConfig
from trainer import Trainer
from model import ProductionTransformer
from utils import load_tokenizer
import jax
import jax.numpy as jnp

# Download Tiny Shakespeare dataset
def download_tiny_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    with open("tiny_shakespeare.txt", "w") as f:
        f.write(response.text)
    return "tiny_shakespeare.txt"

# Train character-level tokenizer
def train_char_tokenizer(file_path):
    tokenizer = CharBPETokenizer()
    tokenizer.train([file_path], vocab_size=256, min_frequency=2, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
    tokenizer.save("char_tokenizer.json")
    return tokenizer

# Prepare data for training
def prepare_data(tokenizer, file_path, seq_len=100):
    with open(file_path, "r") as f:
        text = f.read()
    tokens = tokenizer.encode(text).ids
    # Create sequences
    inputs = []
    targets = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        inputs.append(tokens[i:i+seq_len])
        targets.append(tokens[i+1:i+seq_len+1])
    return inputs, targets

# Main training function
def main(args):
    # Download dataset
    data_file = download_tiny_shakespeare()

    # Train tokenizer
    tokenizer = train_char_tokenizer(data_file)

    # Prepare data
    inputs, targets = prepare_data(tokenizer, data_file)

    # Model config
    vocab_size = tokenizer.get_vocab_size()
    model_config = ModelConfig(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=256,
        max_len=100,
        dropout_rate=0.1,
        activation="gelu",
        use_lora=False,
        lora_rank=8
    )

    # Cloud config (if provided)
    cloud_config = None
    if args.cloud_bucket:
        cloud_config = CloudConfig(
            provider=args.cloud_provider,
            bucket_name=args.cloud_bucket,
            region=args.cloud_region,
            prefix=args.cloud_prefix,
            sas_token=args.azure_sas_token,
            account_name=args.azure_account_name
        )

    # Training config
    train_config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-3,
        num_epochs=5,
        max_steps=len(inputs) // 16,
        warmup_steps=0,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        save_steps=500,
        eval_steps=1,
        log_steps=10,
        timeout_seconds=args.timeout,
        cloud_config=cloud_config
    )

    # Trainer
    trainer = Trainer(model_config, train_config)

    # Load checkpoint if resuming
    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Override data loader for custom data
    class CustomDataLoader:
        def __init__(self, inputs, targets, batch_size):
            self.inputs = inputs
            self.targets = targets
            self.batch_size = batch_size
            self.index = 0

        def get_batch(self, batch_size):
            if self.index + batch_size > len(self.inputs):
                self.index = 0
            batch_inputs = jnp.array(self.inputs[self.index:self.index+batch_size])
            batch_targets = jnp.array(self.targets[self.index:self.index+batch_size])
            self.index += batch_size
            return {"input_ids": batch_inputs, "labels": batch_targets}

    trainer.data_loader = CustomDataLoader(inputs, targets, train_config.batch_size)

    # Train
    rng = jax.random.PRNGKey(42)
    result = trainer.fit(rng)

    # Check if training was stopped due to timeout
    if result and train_config.timeout_seconds:
        print(f"Training stopped due to timeout. Checkpoint saved to: {result}")
        print(f"To resume training, use: --resume-from '{result}'")
    else:
        # Save the trained model
        trainer.save_checkpoint("trained_model")
        # Save model config
        with open("trained_model/config.json", "w") as f:
            json.dump(model_config.dict(), f)
        print("Training completed! Model saved to 'trained_model'")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tiny Shakespeare Transformer")

    # Timeout and cloud options
    parser.add_argument("--timeout", type=int, default=None,
                       help="Training timeout in seconds")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume training from checkpoint (local path or cloud URL)")

    # Cloud storage options
    parser.add_argument("--cloud-provider", type=str, choices=["s3", "gcs", "azure"],
                       default="s3", help="Cloud storage provider")
    parser.add_argument("--cloud-bucket", type=str, default=None,
                       help="Cloud bucket/container name")
    parser.add_argument("--cloud-region", type=str, default=None,
                       help="Cloud region (AWS region or Azure region)")
    parser.add_argument("--cloud-prefix", type=str, default="checkpoints",
                       help="Prefix for checkpoint files in cloud storage")

    # Azure specific options
    parser.add_argument("--azure-sas-token", type=str, default=None,
                       help="Azure SAS token for authentication")
    parser.add_argument("--azure-account-name", type=str, default=None,
                       help="Azure storage account name")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)