"""
Example demonstrating Flash Attention usage with Kvax in the transformer model.

This script shows:
1. How to check flash attention availability
2. How to configure flash attention in ModelConfig
3. How to train a model with flash attention enabled
4. Performance comparison between flash attention and standard attention
"""

import sys
import time
from pathlib import Path

import jax

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import ModelConfig
from src.models.flash_attention import get_flash_attention_config
from src.models.model import ProductionTransformer


def print_flash_attention_info():
    """Print information about flash attention availability and configuration."""
    print("\n" + "=" * 80)
    print("Flash Attention Configuration".center(80))
    print("=" * 80)

    config = get_flash_attention_config()

    print(f"\n📦 Kvax Available:           {config['kvax_available']}")
    print(f"🖥️  Device Type:              {config['device_type']}")
    print(f"⚡ Flash Attention Supported: {config['flash_attention_supported']}")
    print(f"🔧 Device Info:              {config['device_info']}")

    if config.get("forward_params"):
        print("\n🎯 Forward Pass Parameters:")
        for key, value in config["forward_params"].items():
            print(f"   {key:20s}: {value}")

    if config.get("backward_params"):
        print("\n🔄 Backward Pass Parameters:")
        for key, value in config["backward_params"].items():
            print(f"   {key:20s}: {value}")

    print("=" * 80 + "\n")


def create_model_with_flash_attention(vocab_size: int = 1000, use_flash: bool = True):
    """Create a model with flash attention enabled or disabled."""
    config = ModelConfig.from_preset(
        preset="tiny",
        vocab_size=vocab_size,
        max_len=128,
        use_flash_attention=use_flash,
        use_rope=True,
        use_rmsnorm=True,
        use_swiglu=True,
    )

    return ProductionTransformer(config=config), config


def benchmark_attention(
    model, config, batch_size: int = 4, seq_len: int = 128, num_iterations: int = 10
):
    """Benchmark attention performance."""
    # Create dummy data
    key = jax.random.PRNGKey(0)
    x = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    # Initialize model
    params = model.init(key, x, deterministic=True)

    # Warmup
    for _ in range(3):
        _ = model.apply(params, x, deterministic=True)

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        output = model.apply(params, x, deterministic=True)
        output.block_until_ready()  # Ensure computation completes
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    return avg_time, output


def main():
    """Main example function."""
    print("\n🚀 Flash Attention Example for Kvax + JAX\n")

    # 1. Check flash attention availability
    print_flash_attention_info()

    # 2. Create models with and without flash attention
    print("📝 Creating models...\n")

    model_with_flash, config_with_flash = create_model_with_flash_attention(
        vocab_size=1000, use_flash=True
    )
    print("✅ Model with Flash Attention created")
    print(f"   - Model dimension: {config_with_flash.d_model}")
    print(f"   - Num layers: {config_with_flash.num_layers}")
    print(f"   - Num heads: {config_with_flash.num_heads}")
    print(f"   - Use Flash Attention: {config_with_flash.use_flash_attention}")

    model_without_flash, config_without_flash = create_model_with_flash_attention(
        vocab_size=1000, use_flash=False
    )
    print("\n✅ Model without Flash Attention created")
    print(f"   - Use Flash Attention: {config_without_flash.use_flash_attention}")

    # 3. Benchmark performance
    print("\n" + "=" * 80)
    print("Performance Benchmark".center(80))
    print("=" * 80)

    print("\n⏱️  Benchmarking with Flash Attention...")
    time_with_flash, output_with_flash = benchmark_attention(
        model_with_flash,
        config_with_flash,
        batch_size=4,
        seq_len=128,
        num_iterations=10,
    )
    print(f"   Average time: {time_with_flash:.4f} seconds")
    print(f"   Output shape: {output_with_flash.shape}")

    print("\n⏱️  Benchmarking without Flash Attention...")
    time_without_flash, output_without_flash = benchmark_attention(
        model_without_flash,
        config_without_flash,
        batch_size=4,
        seq_len=128,
        num_iterations=10,
    )
    print(f"   Average time: {time_without_flash:.4f} seconds")
    print(f"   Output shape: {output_without_flash.shape}")

    # 4. Calculate speedup
    if time_without_flash > 0:
        speedup = time_without_flash / time_with_flash
        print(f"\n🚀 Speedup with Flash Attention: {speedup:.2f}x")

        if speedup > 1:
            print(f"   ✨ Flash Attention is {speedup:.2f}x faster!")
        elif speedup < 1:
            print(
                f"   ⚠️  Flash Attention is {1 / speedup:.2f}x slower (may be due to overhead on small models/sequences)"
            )
        else:
            print("   ⚖️  Performance is similar")

    # 5. Show how to use in training
    print("\n" + "=" * 80)
    print("Usage in Training".center(80))
    print("=" * 80)

    print("""
To use Flash Attention in your training:

1. Install kvax:
   pip install kvax

2. Create config with flash attention enabled:
   config = ModelConfig.from_preset(
       preset="tiny",
       vocab_size=8000,
       use_flash_attention=True,  # Enable flash attention
   )

3. The model will automatically use flash attention when available
   and fall back to standard attention otherwise.

4. Flash Attention provides:
   ✓ Reduced memory usage (O(N) instead of O(N²))
   ✓ Faster attention computation
   ✓ Support for longer sequences
   ✓ Automatic device optimization (CPU/GPU/TPU)

5. The implementation is fully compatible with:
   ✓ Rotary Position Embeddings (RoPE)
   ✓ Causal masking
   ✓ Padding masks
   ✓ Multi-device training
   ✓ Gradient checkpointing
""")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
