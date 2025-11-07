# Flash Attention Implementation Guide (Kvax)

This guide explains how to use the Kvax Flash Attention implementation in this JAX-based transformer project.

## Overview

Flash Attention is a fast and memory-efficient attention mechanism that reduces the computational complexity from O(N²) to O(N) by using block-wise computation. This implementation uses **Kvax**, a Triton-based Flash Attention 2 implementation for JAX.

### Key Features

- ⚡ **Faster Attention**: Significantly faster than standard attention for long sequences
- 💾 **Memory Efficient**: Reduced memory footprint (O(N) vs O(N²))
- 🔄 **Automatic Fallback**: Falls back to standard attention if Kvax is unavailable
- 🖥️ **Multi-Device Support**: Compatible with CPU, GPU, and TPU
- 🎯 **Seamless Integration**: Works with existing RoPE, RMSNorm, and SwiGLU
- 📊 **Training Ready**: Supports gradient computation and distributed training

## Installation

Install Kvax along with other dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` now includes:
```
kvax  # Flash Attention 2 for JAX
```

## Quick Start

### 1. Basic Usage

```python
from src.config.config import ModelConfig
from src.models.model import ProductionTransformer

# Create config with flash attention enabled (default)
config = ModelConfig.from_preset(
    preset="tiny",
    vocab_size=8000,
    max_len=128,
    use_flash_attention=True,  # Enable flash attention
)

# Create model - flash attention is automatically used
model = ProductionTransformer(config=config)
```

### 2. Check Flash Attention Availability

```python
from src.models.flash_attention import get_flash_attention_config

# Get flash attention configuration
config = get_flash_attention_config()

print(f"Kvax Available: {config['kvax_available']}")
print(f"Device Type: {config['device_type']}")
print(f"Flash Attention Supported: {config['flash_attention_supported']}")
```

### 3. Run Example

```bash
python examples/example_flash_attention.py
```

This will:
- Show flash attention availability and configuration
- Create models with and without flash attention
- Benchmark performance comparison
- Display usage instructions

## Configuration Options

### ModelConfig Parameters

```python
config = ModelConfig(
    vocab_size=8000,
    d_model=512,
    num_heads=8,
    num_layers=12,
    d_ff=2048,
    max_len=512,
    
    # Flash Attention Options
    use_flash_attention=True,              # Enable/disable flash attention
    flash_attention_query_block_size=None, # Query block size (None = auto)
    flash_attention_kv_block_size=None,    # KV block size (None = auto)
    
    # Other optimizations (compatible with flash attention)
    use_rope=True,       # Rotary Position Embeddings
    use_rmsnorm=True,    # RMS Normalization
    use_swiglu=True,     # SwiGLU activation
)
```

### Block Size Configuration

For advanced users, you can customize block sizes:

```python
config = ModelConfig(
    vocab_size=8000,
    d_model=512,
    num_heads=8,
    use_flash_attention=True,
    flash_attention_query_block_size=64,  # Custom query block size
    flash_attention_kv_block_size=64,     # Custom KV block size
)
```

**Note**: Leave these as `None` for automatic configuration based on your hardware.

## Architecture Integration

### How It Works

The flash attention implementation is integrated at multiple levels:

1. **Flash Attention Wrapper** (`src/models/flash_attention.py`):
   - Provides `adaptive_flash_attention()` function
   - Handles device detection and fallback
   - Manages Kvax-specific setup (masks, positions, segment IDs)

2. **MultiHeadAttention Module** (`src/models/model.py`):
   - Uses `adaptive_flash_attention()` when enabled
   - Automatically converts tensor shapes for Kvax compatibility
   - Falls back to standard attention if needed

3. **TransformerBlock**:
   - Passes flash attention configuration to attention layers
   - Works seamlessly with RoPE, RMSNorm, and SwiGLU

### Compatibility with Existing Features

Flash attention works seamlessly with:

| Feature | Compatible | Notes |
|---------|-----------|-------|
| Rotary Position Embeddings (RoPE) | ✅ Yes | RoPE applied before flash attention |
| Causal Masking | ✅ Yes | Handled by Kvax mask creation |
| Padding Masks | ✅ Yes | Automatically integrated |
| RMSNorm | ✅ Yes | Applied in transformer blocks |
| SwiGLU | ✅ Yes | Used in feed-forward layers |
| LoRA | ✅ Yes | Applied to linear projections |
| Multi-device Training | ✅ Yes | Supports sharding specs |
| Gradient Checkpointing | ✅ Yes | Compatible with JAX gradient APIs |

## Device Compatibility

### CPU

Flash attention works on CPU with automatic fallback:

```python
# Automatically detects CPU and uses appropriate implementation
config = ModelConfig.from_preset("tiny", vocab_size=8000, use_flash_attention=True)
model = ProductionTransformer(config=config)
```

### GPU (CUDA)

Optimal performance on NVIDIA GPUs:

```python
# Kvax uses Triton kernels optimized for GPU
import jax
print(f"Device: {jax.devices()[0]}")  # Should show GPU

config = ModelConfig.from_preset("medium", vocab_size=50000, use_flash_attention=True)
model = ProductionTransformer(config=config)
```

**Hardware-specific optimizations**:
- H100: Automatically uses optimized block sizes
- A100: Standard optimizations
- Other GPUs: Automatic tuning

### TPU

Compatible with TPU through JAX:

```python
# TPU support through JAX's XLA compilation
config = ModelConfig.from_preset("large", vocab_size=50000, use_flash_attention=True)
model = ProductionTransformer(config=config)
```

## Performance Considerations

### When to Use Flash Attention

Flash attention provides the most benefit for:

1. **Long Sequences**: Sequences > 512 tokens
2. **Large Models**: Models with many attention heads
3. **Batch Training**: Larger batch sizes
4. **Memory-Constrained Scenarios**: When standard attention OOMs

### Performance Tips

1. **Use with RoPE**: RoPE + Flash Attention = Best performance
   ```python
   config = ModelConfig(use_flash_attention=True, use_rope=True, ...)
   ```

2. **Enable Other Optimizations**: Combine with RMSNorm and SwiGLU
   ```python
   config = ModelConfig(
       use_flash_attention=True,
       use_rope=True,
       use_rmsnorm=True,
       use_swiglu=True,
   )
   ```

3. **Longer Sequences**: Flash attention shines with longer contexts
   ```python
   config = ModelConfig(max_len=2048, use_flash_attention=True, ...)
   ```

4. **Gradient Accumulation**: Use with gradient accumulation for large batches

### Benchmarking

Run the benchmark script:

```bash
python examples/example_flash_attention.py
```

Expected improvements:
- **Speed**: 1.5-3x faster for sequences > 512
- **Memory**: 2-4x less memory usage
- **Throughput**: Higher batch sizes possible

## Distributed Training

### Data Parallelism

```python
import jax

devices = jax.devices()
mesh = jax.sharding.Mesh(devices, ("data",))

# Flash attention automatically uses the mesh
config = ModelConfig.from_preset("medium", vocab_size=50000, use_flash_attention=True)
model = ProductionTransformer(config=config)
```

### Tensor Parallelism

```python
# Shard attention heads across devices
devices = jax.devices()[:4].reshape(2, 2)
mesh = jax.sharding.Mesh(devices, ("data", "model"))

# Flash attention supports tensor parallelism
# Sharding is handled internally
```

### Context Parallelism

For very long sequences, Kvax supports context parallelism:

```python
from kvax.utils import (
    permute_tokens_context_parallelism,
    unpermute_tokens_context_parallelism,
)

# Permute tokens before attention
embeddings, positions, segment_ids = permute_tokens_context_parallelism(
    (embeddings, positions, segment_ids)
)

# Run model with flash attention
output = model.apply(params, embeddings, ...)

# Unpermute back
output = unpermute_tokens_context_parallelism(output)
```

## Troubleshooting

### Flash Attention Not Available

If you see: `Warning: kvax not available. Using standard attention implementation.`

**Solutions**:
1. Install Kvax: `pip install kvax`
2. Check JAX installation: `pip install --upgrade jax jaxlib`
3. Verify CUDA version (for GPU): Kvax requires CUDA-compatible JAX

### Performance Issues

If flash attention is slower than expected:

1. **Check sequence length**: Flash attention benefits longer sequences (> 512)
2. **Verify GPU usage**: `jax.devices()` should show GPU
3. **Increase batch size**: Flash attention is more efficient with larger batches
4. **Profile**: Use JAX profiler to identify bottlenecks

### Memory Issues

If you encounter OOM errors:

1. **Reduce batch size**: Flash attention uses less memory, but still has limits
2. **Use gradient checkpointing**: Trade compute for memory
3. **Adjust block sizes**: Smaller blocks use less memory
   ```python
   config = ModelConfig(
       flash_attention_query_block_size=32,
       flash_attention_kv_block_size=32,
   )
   ```

### Numerical Differences

Flash attention may produce slightly different results than standard attention due to:
- Different numerical precision
- Different accumulation order
- Hardware-specific optimizations

These differences are typically negligible (< 1e-5) and don't affect training.

## Testing

Run the test suite:

```bash
pytest tests/test_flash_attention.py -v
```

Tests include:
- Device detection
- Flash attention availability
- Attention computation correctness
- CPU/GPU/TPU compatibility
- Fallback mechanism
- Model integration

## Advanced Usage

### Custom Attention Masks

```python
import jax.numpy as jnp
from src.models.flash_attention import adaptive_flash_attention

# Create custom mask
batch_size, seq_len = 2, 128
padding_mask = jnp.ones((batch_size, seq_len))
padding_mask = padding_mask.at[:, 100:].set(0)  # Mask last 28 tokens

# Use with flash attention
output = adaptive_flash_attention(
    query=q,
    key=k,
    value=v,
    scale=1.0 / jnp.sqrt(head_dim),
    causal=True,
    padding_mask=padding_mask,
    use_flash_attention=True,
)
```

### Direct Kvax Usage

For advanced users who want direct Kvax control:

```python
from kvax.ops import flash_attention, create_attention_mask
from kvax.utils import attention_specs, PADDING_SEGMENT_ID

# Create positions and segment IDs
positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

# Setup mesh
devices = jax.devices()
mesh = jax.sharding.Mesh(devices, ("data",))

with mesh, attention_specs(
    query_specs=("data", None, None, None),
    kv_specs=("data", None, None, None)
):
    mask = create_attention_mask(
        positions, segment_ids,
        positions, segment_ids,
        calc_bwd_mask=True,
        skip_pad_tokens=True,
    )
    
    output = flash_attention(
        query=q, key=k, value=v,
        query_positions=positions,
        query_segment_ids=segment_ids,
        kv_positions=positions,
        kv_segment_ids=segment_ids,
        mask=mask,
        scale=scale,
    )
```

## References

- [Kvax GitHub Repository](https://github.com/nebius/kvax)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)
- [JAX Documentation](https://jax.readthedocs.io/)

## FAQ

**Q: Does flash attention work on CPU?**
A: Yes, with automatic fallback to standard attention if needed.

**Q: Can I use flash attention with RoPE?**
A: Yes, RoPE is applied before flash attention and works seamlessly.

**Q: Does flash attention change the model output?**
A: Outputs are numerically similar but may have small differences due to implementation details.

**Q: Can I disable flash attention?**
A: Yes, set `use_flash_attention=False` in ModelConfig.

**Q: Does flash attention support KV caching?**
A: KV caching is not currently implemented but can be added in the future.

**Q: What's the minimum sequence length to benefit from flash attention?**
A: Generally, sequences > 512 tokens see significant benefits.

**Q: Does it work with gradient accumulation?**
A: Yes, flash attention is fully compatible with gradient accumulation.

## Contributing

To improve the flash attention implementation:

1. Test on different hardware (GPU models, TPU versions)
2. Benchmark with various sequence lengths
3. Optimize block sizes for specific use cases
4. Add KV caching support
5. Implement additional masking strategies

## License

This implementation follows the same license as the main project.
