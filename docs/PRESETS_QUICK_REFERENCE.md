# Model Presets - Quick Reference

## The Problem
Your model with `d_model=64, num_layers=92` is **super slow** because:
- 92 layers = 92 sequential operations per token
- Too narrow (64 dims) = poor GPU utilization
- **Result:** Generating 100 tokens requires 9,200 operations!

## The Solution
Use presets with **fewer layers, wider dimensions**:

```python
from config import ModelConfig

# ❌ OLD (SLOW):
config = ModelConfig(vocab_size=8986, d_model=64, num_layers=92, ...)

# ✅ NEW (FAST):
config = ModelConfig.from_preset("tiny", vocab_size=8986, max_len=128)
```

---

## Available Presets

| Preset | Params | d_model | Layers | Speed | Use Case |
|--------|--------|---------|--------|-------|----------|
| **nano** | 4M | 128 | 6 | ⚡⚡⚡⚡⚡ | Testing, prototyping |
| **tiny** | 13M | 256 | 8 | ⚡⚡⚡⚡ | Fast inference, recommended |
| **small** | 60M | 512 | 12 | ⚡⚡⚡ | Balanced quality/speed |
| **medium** | 127M | 768 | 12 | ⚡⚡ | GPT-2 Small scale |
| **large** | 421M | 1024 | 24 | ⚡ | High quality |
| **xlarge** | 862M | 1280 | 32 | 🐢 | Production large |

---

## Speed Comparison

For generating 100 tokens:

| Config | Operations | Speed |
|--------|-----------|-------|
| Your old (92 layers) | 9,200 | 🐌 Baseline |
| **nano** (6 layers) | 600 | **15x faster** ⚡ |
| **tiny** (8 layers) | 800 | **11x faster** ⚡ |
| small (12 layers) | 1,200 | 8x faster |

---

## Usage Examples

### Basic
```python
config = ModelConfig.from_preset("tiny", vocab_size=8986, max_len=128)
```

### With Overrides
```python
config = ModelConfig.from_preset(
    "tiny",
    vocab_size=8986,
    max_len=128,
    dropout_rate=0.2,    # Custom
    activation="silu"    # Custom
)
```

### List All Presets
```python
ModelConfig.list_presets()
```

---

## Recommendation

**For your 5M parameter target:**
- Use **`nano`** preset (~4M params, 15x faster)
- Similar parameter count to your original
- Much better architecture (6 layers × 128 dims)
- Proven ratios (d_ff = 4 × d_model)

**Why it's faster:**
- 6 sequential operations vs 92
- Wider dimensions = better GPU utilization
- Proper architecture ratios

---

## Key Insight

**Width > Depth** for small models:
- **Your old:** 64 dims × 92 layers = tall & narrow = SLOW
- **nano:** 128 dims × 6 layers = wide & shallow = FAST
- **tiny:** 256 dims × 8 layers = wider & shallow = FASTER quality

Both have similar parameters, but the second is 15x faster!

---

## Try It Now

```bash
# See all presets
uv run python example_presets.py

# Calculate exact parameter counts
uv run python show_preset_params.py

# See training examples
uv run python example_preset_training.py
```

---

## Why Presets Work

Based on successful architectures:
- **GPT-2 Small:** 768 dims × 12 layers (not 64 × 144!)
- **BERT Base:** 768 dims × 12 layers
- **LLaMA 7B:** 4096 dims × 32 layers

Industry standard: **Prioritize width over depth** ✅
