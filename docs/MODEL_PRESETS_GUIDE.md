# Model Presets Guide

This guide shows how to use the predefined model size presets instead of manually configuring architecture parameters.

## Available Presets

| Preset  | Parameters | d_model | Layers | Heads | Use Case |
|---------|-----------|---------|--------|-------|----------|
| **nano**   | ~4M  | 128  | 6  | 4  | Ultra-fast testing & prototyping |
| **tiny**   | ~13M | 256  | 8  | 8  | Fast inference, resource-constrained |
| **small**  | ~60M | 512  | 12 | 8  | Balanced general tasks |
| **medium** | ~127M| 768  | 12 | 12 | GPT-2 Small / BERT Base scale |
| **large**  | ~421M| 1024 | 24 | 16 | High-quality generation |
| **xlarge** | ~862M| 1280 | 32 | 20 | Production-grade large model |

## Why Use Presets?

**Before (manual config - BAD):**
```python
config = ModelConfig(
    vocab_size=8986,
    d_model=64,        # Too narrow!
    num_layers=92,     # Way too deep! → SUPER SLOW
    num_heads=4,
    d_ff=128,
    max_len=128
)
# Result: 92 sequential operations = VERY SLOW generation
```

**After (with preset - GOOD):**
```python
config = ModelConfig.from_preset("tiny", vocab_size=8986, max_len=128)
# Result: 8 sequential operations = 11x FASTER generation
```

## Usage Examples

### 1. Basic Usage
```python
from config import ModelConfig

# Create a tiny model (recommended for 5M-15M params)
config = ModelConfig.from_preset("tiny", vocab_size=8986, max_len=128)

# Create a nano model (for ultra-fast testing)
config = ModelConfig.from_preset("nano", vocab_size=8986, max_len=128)

# Create a medium model (GPT-2 Small scale)
config = ModelConfig.from_preset("medium", vocab_size=50000, max_len=512)
```

### 2. With Custom Overrides
```python
# Start with tiny preset, but customize dropout and activation
config = ModelConfig.from_preset(
    "tiny",
    vocab_size=8986,
    max_len=128,
    dropout_rate=0.2,      # Higher dropout
    activation="silu",     # Different activation
    use_lora=True          # Enable LoRA fine-tuning
)
```

### 3. List All Presets
```python
from config import ModelConfig

# Print all available presets with their specs
ModelConfig.list_presets()
```

### 4. In Training Scripts
```python
from config import ModelConfig, TrainingConfig
from trainer import Trainer

# Create model config from preset
model_config = ModelConfig.from_preset(
    "tiny",
    vocab_size=8986,
    max_len=128
)

# Create training config
train_config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=10
)

# Train
trainer = Trainer(model_config, train_config)
trainer.train(train_dataset, eval_dataset)
```

## Parameter Count by Preset

For vocab_size=8986, max_len=128:

- **NANO**:   ~3.9M parameters
- **TINY**:   ~13M parameters  ← **Recommended for your use case**
- **SMALL**:  ~60M parameters
- **MEDIUM**: ~127M parameters
- **LARGE**:  ~421M parameters
- **XLARGE**: ~862M parameters

## Speed Comparison

Generating 100 tokens requires:

| Config | Operations | Speed |
|--------|-----------|-------|
| Your old (d_model=64, layers=92) | 9,200 | 🐌 Baseline (SLOW) |
| NANO preset (d_model=128, layers=6) | 600 | ⚡ 15x faster |
| TINY preset (d_model=256, layers=8) | 800 | ⚡ 11x faster |
| SMALL preset (d_model=512, layers=12) | 1,200 | ⚡ 8x faster |

## Recommendations

### For Your 5M Parameter Target:
- **Use NANO preset** (~4M params, 15x faster than your 92-layer config)
- The TINY preset is ~13M params (slightly larger but still very fast)

### General Guidelines:
- **Prototyping/Testing**: Use `nano` (ultra-fast, trains in minutes)
- **Production (small)**: Use `tiny` or `small` (good quality, fast)
- **Production (large)**: Use `medium` or `large` (high quality)
- **Research**: Use `xlarge` (if you have GPUs)

### Architecture Philosophy:
- **Width > Depth** for models under 1B parameters
- Wider models are faster (parallel computation)
- Deeper models are slower (sequential computation)
- Presets use proven ratios from GPT-2, BERT, LLaMA

## Migration from Old Config

If you have an existing model with manual config:

```python
# OLD WAY (don't do this)
config = ModelConfig(
    vocab_size=8986,
    d_model=64,
    num_layers=92,  # Too deep!
    num_heads=4,
    d_ff=128,
    max_len=128
)

# NEW WAY (much better)
config = ModelConfig.from_preset("nano", vocab_size=8986, max_len=128)
# Or if you want closer to original param count:
config = ModelConfig.from_preset(
    "nano",
    vocab_size=8986, 
    max_len=128,
    d_model=96,      # Custom: slightly wider
    num_layers=8     # Custom: fewer layers
)
```

## Why Your Original Config Was Slow

Your config had:
- `d_model=64` (very narrow)
- `num_layers=92` (extremely deep)

Problems:
1. **Sequential bottleneck**: 92 layers must execute one-by-one
2. **Poor parallelization**: Width (64) is too small to utilize GPU
3. **Memory thrashing**: Data bounces through 92 stages
4. **Inefficient ratios**: Should be `d_ff ≈ 4 × d_model`, not `d_ff < 2 × d_model`

The TINY preset fixes all of this:
- `d_model=256` (4x wider → better GPU utilization)
- `num_layers=8` (11x fewer → 11x faster generation)
- `d_ff=1024` (4x d_model → proper ratio)
- Proven architecture from successful models

## Additional Resources

- Run `python example_presets.py` to see all presets
- Run `python show_preset_params.py` to calculate exact parameter counts
- See `config.py` for preset definitions and `from_preset()` implementation
