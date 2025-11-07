"""
Quick test script for train_model.py

Tests the training pipeline with a small model and limited data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_model import train_model

if __name__ == "__main__":
    print("Testing train_model.py with minimal configuration...")
    print("This will train a small 500K parameter model on limited data.\n")

    results = train_model(
        # Dataset
        dataset_id="skeskinen/TinyStories-Instruct-hf",
        text_column="text",
        # Model - small for quick testing
        target_params=500_000,
        vocab_size=5000,
        seq_len=64,
        # Training - minimal for quick testing
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        # Data - limited for quick testing
        max_examples=100,
        tokenizer_train_examples=100,
        # Output
        output_dir="test_output",
        log_every=5,
    )

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    for key, value in results.items():
        print(f"  {key}: {value}")

    print("\n✓ Test completed successfully!")
