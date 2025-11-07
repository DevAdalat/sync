# Flash Attention Quick Reference

## Installation

```bash
pip install -r requirements.txt
# Installs: jax, flax, kvax, and other dependencies
```

## Quick Start

### Enable Flash Attention

```python
from src.config.config import ModelConfig
from src.models.model import ProductionTransformer

# Method 1: Using preset
config = ModelConfig.from_preset(
    preset="tiny",
    vocab_size=8000,
    use_flash_attention=True  # ⚡ Enable flash attention
)

# Method 2: Custom config
config = ModelConfig(
    vocab_size=8000,
    d_model=512,
    num_heads=8,
    num_layers=12,
    d_ff=2048,
    max_len=512,
    use_flash_attention=True  # ⚡ Enable flash attention
)

model = ProductionTransformer(config=config)
```

### Check Availability

```python
from src.models.flash_attention import get_flash_attention_config

info = get_flash_attention_config()
print(f"Device: {info['device_type']}")
print(f"Supported: {info['flash_attention_supported']}")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_flash_attention` | bool | `True` | Enable/disable flash attention |
| `flash_attention_query_block_size` | int | `None` | Query block size (None = auto) |
| `flash_attention_kv_block_size` | int | `None` | KV block size (None = auto) |

## Device Compatibility

| Device | Status | Notes |
|--------|--------|-------|
| CPU | ✅ Supported | With automatic fallback |
| GPU (CUDA) | ✅ Optimal | Best performance with Triton |
| TPU | ✅ Supported | Through JAX XLA |

## API Reference

### `adaptive_flash_attention()`

Main attention function with automatic fallback.

```python
from src.models.flash_attention import adaptive_flash_attention

output = adaptive_flash_attention(
    query=q,                      # [batch, heads, seq, dim]
    key=k,                        # [batch, heads, seq, dim]
    value=v,                      # [batch, heads, seq, dim]
    scale=1.0/sqrt(head_dim),     # Attention scale
    causal=True,                  # Causal masking
    padding_mask=None,            # Optional [batch, seq]
    use_flash_attention=True,     # Try flash attention
)
```

### `get_flash_attention_config()`

Get configuration information.

```python
from src.models.flash_attention import get_flash_attention_config

config = get_flash_attention_config()
# Returns: dict with keys:
#   - kvax_available: bool
#   - device_type: str
#   - flash_attention_supported: bool
#   - device_info: str
#   - forward_params: dict (if available)
#   - backward_params: dict (if available)
```

### `detect_device_type()`

Detect current device.

```python
from src.models.flash_attention import detect_device_type

device = detect_device_type()
# Returns: 'cpu', 'gpu', or 'tpu'
```

## Examples

### Basic Training

```python
import jax
from src.config.config import ModelConfig
from src.models.model import ProductionTransformer

# Setup
config = ModelConfig.from_preset("tiny", vocab_size=8000, use_flash_attention=True)
model = ProductionTransformer(config=config)

# Initialize
key = jax.random.PRNGKey(0)
dummy_input = jax.random.randint(key, (4, 128), 0, 8000)
params = model.init(key, dummy_input)

# Forward pass - flash attention is automatic
output = model.apply(params, dummy_input, deterministic=True)
```

### Compare Performance

```python
import time
from src.models.flash_attention import get_flash_attention_config

# Check availability
info = get_flash_attention_config()
print(f"Flash Attention: {info['flash_attention_supported']}")

# Create two models
config_with = ModelConfig.from_preset("tiny", vocab_size=8000, use_flash_attention=True)
config_without = ModelConfig.from_preset("tiny", vocab_size=8000, use_flash_attention=False)

model_with = ProductionTransformer(config=config_with)
model_without = ProductionTransformer(config=config_without)

# Benchmark (see examples/example_flash_attention.py for full code)
```

### Custom Block Sizes

```python
config = ModelConfig(
    vocab_size=8000,
    d_model=512,
    num_heads=8,
    use_flash_attention=True,
    flash_attention_query_block_size=64,  # Custom
    flash_attention_kv_block_size=64,     # Custom
)
```

## Testing

```bash
# Run tests
pytest tests/test_flash_attention.py -v

# Run specific test
pytest tests/test_flash_attention.py::TestDeviceDetection -v

# Run with output
pytest tests/test_flash_attention.py -v -s
```

## Troubleshooting

### Kvax Not Available

**Problem**: `Warning: kvax not available`

**Solutions**:
1. Install: `pip install kvax`
2. Check JAX: `python -c "import jax; print(jax.devices())"`
3. Verify CUDA (GPU): Check CUDA version matches JAX

### Slower Than Expected

**Problem**: Flash attention slower than standard

**Solutions**:
1. Use longer sequences (> 512 tokens)
2. Increase batch size
3. Check device: Should be GPU/TPU for best performance
4. Warmup: Run a few iterations first

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
1. Reduce batch size
2. Use smaller block sizes:
   ```python
   config = ModelConfig(
       flash_attention_query_block_size=32,
       flash_attention_kv_block_size=32,
   )
   ```
3. Enable gradient checkpointing

## Performance Tips

### ✅ Best Practices

1. **Use with Long Sequences**: Flash attention shines at seq_len > 512
2. **Combine Optimizations**: Enable RoPE, RMSNorm, SwiGLU together
3. **Larger Batches**: Flash attention scales well with batch size
4. **GPU/TPU**: Use GPU or TPU for optimal performance

### ❌ Avoid

1. **Very Short Sequences**: < 256 tokens may have overhead
2. **CPU-Only Production**: Use GPU/TPU when possible
3. **Disabling on Long Sequences**: Keep enabled for seq_len > 512

## Code Snippets

### Check If Flash Attention Is Active

```python
from src.models.flash_attention import KVAX_AVAILABLE, is_flash_attention_supported

if KVAX_AVAILABLE and is_flash_attention_supported():
    print("✅ Using Flash Attention")
else:
    print("⚠️  Using Standard Attention (fallback)")
```

### Disable Flash Attention Temporarily

```python
# In config
config = ModelConfig.from_preset("tiny", vocab_size=8000, use_flash_attention=False)

# Or modify existing config
config.use_flash_attention = False
```

### Get Block Sizes

```python
from src.models.flash_attention import get_flash_attention_config

config = get_flash_attention_config()
if 'forward_params' in config:
    print(f"Query blocks: {config['forward_params']['query_block_size']}")
    print(f"KV blocks: {config['forward_params']['kv_block_size']}")
```

## File Locations

| File | Purpose |
|------|---------|
| `src/models/flash_attention.py` | Core implementation |
| `src/models/model.py` | Model integration |
| `src/config/config.py` | Configuration |
| `examples/example_flash_attention.py` | Usage example |
| `tests/test_flash_attention.py` | Test suite |
| `docs/FLASH_ATTENTION_GUIDE.md` | Full documentation |

## Command Reference

```bash
# Installation
pip install -r requirements.txt

# Verify installation
python -c "from src.models.flash_attention import get_flash_attention_config; print(get_flash_attention_config())"

# Run example
python examples/example_flash_attention.py

# Run tests
pytest tests/test_flash_attention.py -v

# Check device
python -c "import jax; print(jax.devices())"
```

## Integration Checklist

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Verify installation (run `python examples/example_flash_attention.py`)
- [ ] Enable in config (`use_flash_attention=True`)
- [ ] Test with your data
- [ ] Benchmark performance
- [ ] Review documentation (`docs/FLASH_ATTENTION_GUIDE.md`)

## Common Patterns

### Pattern 1: Standard Training

```python
config = ModelConfig.from_preset("medium", vocab_size=50000, use_flash_attention=True)
model = ProductionTransformer(config=config)
# Train as usual - flash attention is automatic
```

### Pattern 2: Distributed Training

```python
import jax

devices = jax.devices()
mesh = jax.sharding.Mesh(devices, ("data",))

config = ModelConfig.from_preset("large", vocab_size=50000, use_flash_attention=True)
model = ProductionTransformer(config=config)
# Flash attention uses mesh automatically
```

### Pattern 3: Inference with Long Context

```python
config = ModelConfig.from_preset(
    "medium",
    vocab_size=50000,
    max_len=2048,  # Long context
    use_flash_attention=True,  # Essential for long sequences
)
model = ProductionTransformer(config=config)
```

## Support

- **Documentation**: `docs/FLASH_ATTENTION_GUIDE.md`
- **Examples**: `examples/example_flash_attention.py`
- **Tests**: `tests/test_flash_attention.py`
- **Issues**: Check test output for diagnostics

## Version Info

- **Implementation**: Kvax (Nebius Flash Attention 2)
- **Backend**: JAX with Triton kernels
- **Status**: Production ready
- **Updated**: 2025-11-07
