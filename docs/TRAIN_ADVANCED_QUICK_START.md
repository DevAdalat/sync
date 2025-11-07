# Train Advanced - Quick Start

**The simplest way to train a transformer model with just-in-time data processing.**

---

## TL;DR - One Command Training

```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset tiny \
  --auto-vocab-size \
  --epochs 3
```

That's it! No pre-processing, no manual configuration. Just works.

---

## What You Get

✅ **Just-in-Time Processing** - Data tokenized on-the-fly during training  
✅ **Auto Vocabulary** - Optimal vocab size determined automatically  
✅ **Streaming Data** - Memory-efficient loading (train on any size dataset)  
✅ **Smart Defaults** - Everything pre-configured for best results  
✅ **Model Presets** - Choose from nano to xlarge  

---

## 5 Essential Commands

### 1. Basic Training
```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset tiny \
  --auto-vocab-size \
  --epochs 3
```

### 2. Training with Validation
```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset small \
  --auto-vocab-size \
  --val-split "validation" \
  --epochs 5
```

### 3. Memory-Optimized Training
```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset tiny \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --use-mixed-precision \
  --auto-vocab-size
```

### 4. Custom Architecture
```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --d-model 768 \
  --num-layers 12 \
  --num-heads 12 \
  --vocab-size 20000 \
  --epochs 5
```

### 5. Resume from Checkpoint
```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset tiny \
  --resume-from "output_advanced/checkpoint_step_1000" \
  --epochs 10
```

---

## Model Presets

| Preset | Size | Use Case |
|--------|------|----------|
| `nano` | ~5M params | Testing, experiments |
| `tiny` | ~15M params | Small tasks, limited resources |
| `small` | ~50M params | **General purpose** ⭐ |
| `medium` | ~150M params | High quality |
| `large` | ~400M params | Advanced |
| `xlarge` | ~1B params | Research |

**Recommendation:** Start with `small` for production use.

---

## Key Parameters

### Must Specify
- `--dataset-id` - HuggingFace dataset (e.g., "roneneldan/TinyStories")

### Common Options
- `--model-preset` - Model size (nano/tiny/small/medium/large/xlarge)
- `--auto-vocab-size` - Auto-determine vocabulary size
- `--epochs` - Number of training epochs (default: 3)
- `--val-split` - Validation split name (e.g., "validation")
- `--batch-size` - Batch size per device (default: 32)
- `--learning-rate` - Peak learning rate (default: 5e-4)

### Memory Optimization
- `--batch-size 8` - Reduce batch size
- `--gradient-accumulation-steps 8` - Maintain effective batch size
- `--use-mixed-precision` - Use bfloat16 (saves 50% memory)
- `--shuffle-buffer-size 1000` - Smaller shuffle buffer

### Checkpoints
- `--save-every 1000` - Save checkpoint every N steps
- `--eval-every 500` - Validate every N steps
- `--resume-from PATH` - Resume from checkpoint

---

## Output Files

After training, find in `output_advanced/`:

```
output_advanced/
├── tokenizer.json           # ← Use for inference
├── model_config.json        # ← Model configuration
├── final_checkpoint/        # ← Final trained model
└── best_checkpoint/         # ← Best validation model (if val_split)
```

---

## Memory Requirements

Rough estimates for full precision (float32):

| Preset | Model Params | Memory (Training) | Recommended RAM |
|--------|--------------|-------------------|-----------------|
| nano | ~5M | ~500 MB | 2 GB |
| tiny | ~15M | ~1.5 GB | 4 GB |
| small | ~50M | ~4 GB | 8 GB |
| medium | ~150M | ~10 GB | 16 GB |
| large | ~400M | ~25 GB | 32 GB |
| xlarge | ~1B | ~60 GB | 64 GB+ |

**Note:** Use `--use-mixed-precision` to cut memory usage by ~50%

---

## Just-in-Time Processing vs Pre-Processing

### Traditional Approach ❌
```python
# Load all data
dataset = load_dataset("...", split="train")  # 10 GB in memory

# Tokenize ALL data upfront
for text in dataset:
    tokens = tokenizer.encode(text)  # Store all tokens

# Then train
train(tokenized_data)  # Additional memory for tokens
```

**Problem:** High memory usage, slow startup

### Train Advanced ✅
```python
# Just connect to dataset (no loading)
dataset = load_dataset("...", split="train", streaming=True)

# Train immediately (tokenize batches on-demand)
for batch_text in dataset:
    tokens = tokenizer.encode(batch_text)  # Only current batch
    train_step(tokens)
    # Discard tokens, fetch next batch
```

**Benefits:**
- ✅ Low memory (only current batch)
- ✅ Fast startup (no pre-processing)
- ✅ Train on any size dataset

---

## Troubleshooting

### Out of Memory?
```bash
# Reduce memory usage
--batch-size 8 \
--gradient-accumulation-steps 8 \
--use-mixed-precision \
--shuffle-buffer-size 1000
```

### Training Too Slow?
- Use GPU/TPU instead of CPU
- Increase `--batch-size` (if memory allows)
- Reduce `--save-every` and `--eval-every`

### Loss Not Decreasing?
- Increase `--learning-rate` (try 1e-3)
- Check your dataset quality
- Try larger model preset

---

## Examples

### Example 1: Quick Test
```bash
# Train nano model on 1000 examples (30 seconds)
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset nano \
  --max-train-examples 1000 \
  --epochs 1 \
  --auto-vocab-size
```

### Example 2: Production Training
```bash
# Train small model with validation
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset small \
  --epochs 10 \
  --val-split "validation" \
  --eval-every 500 \
  --save-every 1000 \
  --auto-vocab-size
```

### Example 3: Large Model, Limited Memory
```bash
# Train medium model on 8GB RAM
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset medium \
  --batch-size 8 \
  --gradient-accumulation-steps 16 \
  --use-mixed-precision \
  --shuffle-buffer-size 1000 \
  --auto-vocab-size
```

---

## What Makes This Different?

### vs `train_model.py`
- ❌ `train_model.py` - Pre-processes all data before training
- ✅ `train_advanced.py` - Just-in-time processing (lower memory)

### vs `train_optimized.py`
- ❌ `train_optimized.py` - Manual configuration required
- ✅ `train_advanced.py` - Auto vocab size, presets, all-in-one

### vs `train_hf_dataset.py`
- ❌ `train_hf_dataset.py` - Basic functionality only
- ✅ `train_advanced.py` - Full features (validation, checkpoints, cloud)

**Recommendation:** Use `train_advanced.py` for all new projects.

---

## Next Steps

1. **Train your first model** (2 minutes)
   ```bash
   python -m src.training.train_advanced \
     --dataset-id "roneneldan/TinyStories" \
     --model-preset tiny \
     --auto-vocab-size \
     --max-train-examples 1000 \
     --epochs 1
   ```

2. **Scale up** to full training
   ```bash
   python -m src.training.train_advanced \
     --dataset-id "roneneldan/TinyStories" \
     --model-preset small \
     --auto-vocab-size \
     --epochs 5 \
     --val-split "validation"
   ```

3. **Generate text** with your trained model
   - See [GENERATION_GUIDE.md](GENERATION_GUIDE.md)

---

## Full Documentation

For complete details, see [TRAIN_ADVANCED_GUIDE.md](TRAIN_ADVANCED_GUIDE.md)

**Topics covered:**
- All parameter reference
- Memory optimization strategies
- Best practices
- Advanced usage patterns
- Cloud storage integration
- Troubleshooting guide

---

## Summary

**One command to train a transformer model:**

```bash
python -m src.training.train_advanced \
  --dataset-id "YOUR_DATASET" \
  --model-preset small \
  --auto-vocab-size \
  --epochs 5
```

**Key benefits:**
- ✅ No data pre-processing needed
- ✅ Memory-efficient streaming
- ✅ Auto vocabulary size
- ✅ Smart defaults
- ✅ Production-ready

**Get started now!**
