import os
import requests
from tokenizers import CharBPETokenizer
from config import ModelConfig, TrainingConfig, DataConfig
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
def main():
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
        log_steps=10
    )

    # Trainer
    trainer = Trainer(model_config, train_config)

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
    trainer.fit(rng)

    print("Training completed!")

if __name__ == "__main__":
    main()