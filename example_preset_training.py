"""
Example: Using Model Presets in Training

This shows how to use the new preset system in your training scripts.
"""

from config import ModelConfig, TrainingConfig

def example_quick_training():
    """Quick training example with nano preset (fastest)."""
    print("="*80)
    print("EXAMPLE 1: Quick Training with NANO Preset")
    print("="*80)
    
    # Use nano for ultra-fast training/testing
    model_config = ModelConfig.from_preset(
        "nano",
        vocab_size=8986,
        max_len=128
    )
    
    train_config = TrainingConfig(
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=5,
        save_steps=500
    )
    
    print(f"\nModel Config:")
    print(f"  Preset: nano")
    print(f"  d_model: {model_config.d_model}")
    print(f"  num_layers: {model_config.num_layers}")
    print(f"  Parameters: ~4M")
    print(f"  Training time: VERY FAST (perfect for iteration)")
    print(f"\nTraining Config:")
    print(f"  batch_size: {train_config.batch_size}")
    print(f"  learning_rate: {train_config.learning_rate}")
    print(f"  num_epochs: {train_config.num_epochs}")
    print()


def example_production_training():
    """Production training with tiny preset (recommended for 5-15M params)."""
    print("="*80)
    print("EXAMPLE 2: Production Training with TINY Preset")
    print("="*80)
    
    # Use tiny for production (good balance of speed and quality)
    model_config = ModelConfig.from_preset(
        "tiny",
        vocab_size=8986,
        max_len=128,
        dropout_rate=0.15  # Custom override
    )
    
    train_config = TrainingConfig(
        batch_size=64,
        learning_rate=5e-4,
        num_epochs=20,
        warmup_steps=1000,
        save_steps=2000,
        eval_steps=500
    )
    
    print(f"\nModel Config:")
    print(f"  Preset: tiny (with custom dropout)")
    print(f"  d_model: {model_config.d_model}")
    print(f"  num_layers: {model_config.num_layers}")
    print(f"  dropout_rate: {model_config.dropout_rate} (customized)")
    print(f"  Parameters: ~13M")
    print(f"  Generation speed: 11x faster than 92-layer config")
    print(f"\nTraining Config:")
    print(f"  batch_size: {train_config.batch_size}")
    print(f"  learning_rate: {train_config.learning_rate}")
    print(f"  warmup_steps: {train_config.warmup_steps}")
    print()


def example_custom_preset():
    """Custom configuration starting from a preset."""
    print("="*80)
    print("EXAMPLE 3: Custom Config Based on TINY Preset")
    print("="*80)
    
    # Start with tiny, but customize everything
    model_config = ModelConfig.from_preset(
        "tiny",
        vocab_size=8986,
        max_len=256,              # Longer sequences
        dropout_rate=0.2,         # Higher dropout
        activation="silu",        # Different activation
        use_lora=True,            # Enable LoRA
        lora_rank=16              # Custom LoRA rank
    )
    
    print(f"\nModel Config:")
    print(f"  Base preset: tiny")
    print(f"  d_model: {model_config.d_model}")
    print(f"  num_layers: {model_config.num_layers}")
    print(f"  max_len: {model_config.max_len} (customized)")
    print(f"  dropout_rate: {model_config.dropout_rate} (customized)")
    print(f"  activation: {model_config.activation} (customized)")
    print(f"  use_lora: {model_config.use_lora} (customized)")
    print(f"  lora_rank: {model_config.lora_rank} (customized)")
    print()


def example_comparison():
    """Compare old bad config with new preset."""
    print("="*80)
    print("EXAMPLE 4: Before & After Comparison")
    print("="*80)
    
    print("\n❌ OLD CONFIG (Your original - SLOW):")
    print("-" * 60)
    old_config_desc = """
    config = ModelConfig(
        vocab_size=8986,
        d_model=64,        # Too narrow
        num_layers=92,     # WAY too deep
        num_heads=4,
        d_ff=128,
        max_len=128
    )
    
    Problems:
    - 92 sequential layers = VERY SLOW generation
    - Narrow width (64) = poor GPU utilization
    - Bad ratios (d_ff should be ~4x d_model)
    - Generation: 9,200 operations for 100 tokens
    """
    print(old_config_desc)
    
    print("\n✅ NEW CONFIG (Recommended - FAST):")
    print("-" * 60)
    new_config = ModelConfig.from_preset("tiny", vocab_size=8986, max_len=128)
    new_config_desc = f"""
    config = ModelConfig.from_preset("tiny", vocab_size=8986, max_len=128)
    
    Result:
    - d_model: {new_config.d_model} (4x wider)
    - num_layers: {new_config.num_layers} (11x fewer)
    - num_heads: {new_config.num_heads} (2x more)
    - d_ff: {new_config.d_ff} (8x larger)
    - Generation: 800 operations for 100 tokens (11x faster!)
    - Similar parameter count (~13M vs ~5M)
    - Much better performance (proven architecture)
    """
    print(new_config_desc)


def main():
    print("\n" + "="*80)
    print("MODEL PRESET USAGE EXAMPLES")
    print("="*80 + "\n")
    
    # Show all examples
    example_quick_training()
    input("Press Enter to continue...")
    
    example_production_training()
    input("Press Enter to continue...")
    
    example_custom_preset()
    input("Press Enter to continue...")
    
    example_comparison()
    
    print("\n" + "="*80)
    print("HOW TO USE IN YOUR SCRIPTS")
    print("="*80)
    print("""
Just replace your manual ModelConfig creation with:

    from config import ModelConfig
    
    # OLD:
    # config = ModelConfig(vocab_size=8986, d_model=64, num_layers=92, ...)
    
    # NEW:
    config = ModelConfig.from_preset("tiny", vocab_size=8986, max_len=128)

Then use it normally in your training:

    from trainer import Trainer
    from config import TrainingConfig
    
    model_config = ModelConfig.from_preset("tiny", vocab_size=8986, max_len=128)
    train_config = TrainingConfig(batch_size=32, learning_rate=1e-3, num_epochs=10)
    
    trainer = Trainer(model_config, train_config)
    trainer.train(train_dataset, eval_dataset)

That's it! Your model will be 11x faster with better performance.
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
