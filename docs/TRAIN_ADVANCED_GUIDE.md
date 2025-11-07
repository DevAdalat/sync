# Advanced Training Guide

**Complete guide to using `train_advanced.py` - the all-in-one training solution with just-in-time data processing.**

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [Common Usage Patterns](#common-usage-patterns)
5. [Parameter Reference](#parameter-reference)
6. [Just-in-Time Processing Explained](#just-in-time-processing-explained)
7. [Memory Optimization](#memory-optimization)
8. [Best Practices](#best-practices)

---

## Overview

`train_advanced.py` is a comprehensive, production-ready training script that combines all best practices:

- **Just-in-time data processing** - No pre-processing required! Data is tokenized on-the-fly during training
- **Auto vocabulary detection** - Automatically determines optimal vocab size based on your dataset
- **Streaming data loading** - Memory-efficient loading from HuggingFace datasets
- **Model presets** - Use predefined architectures (nano to xlarge) or customize
- **Advanced training features** - Gradient accumulation, mixed precision, learning rate scheduling
- **Validation support** - Evaluate during training with automatic best model saving
- **Cloud storage integration** - Upload checkpoints to S3/GCS/Azure

---

## Key Features

### ✅ Just-in-Time Processing
Unlike traditional approaches that pre-process all data before training:
- ❌ **Old way**: Load dataset → Tokenize ALL data → Store in memory → Train
- ✅ **New way**: Load dataset → Train (tokenize batches on-the-fly)

**Benefits:**
- Lower memory usage (only current batch in memory)
- Faster startup time (no pre-processing wait)
- Can train on massive datasets that don't fit in RAM

### ✅ Auto Vocabulary Size
The script automatically analyzes your dataset and determines optimal vocabulary size:
```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset tiny \
  --auto-vocab-size \
  --epochs 3
```

No need to manually specify `--vocab-size`!

### ✅ Model Presets
Choose from predefined architectures optimized for different use cases:

| Preset | Parameters | Best For |
|--------|-----------|----------|
| **nano** | ~5M | Rapid experimentation, testing |
| **tiny** | ~15M | Small tasks, resource-constrained environments |
| **small** | ~50M | General-purpose, balanced performance |
| **medium** | ~150M | High-quality text generation |
| **large** | ~400M | Advanced applications |
| **xlarge** | ~1B | Research, maximum performance |

---

## Quick Start

### 1. Basic Training with Preset
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
  --epochs 5 \
  --val-split "validation" \
  --eval-every 500
```

### 3. Custom Architecture
```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --d-model 768 \
  --num-layers 12 \
  --num-heads 12 \
  --d-ff 3072 \
  --vocab-size 20000 \
  --epochs 5
```

### 4. Memory-Optimized Training
```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset small \
  --batch-size 16 \
  --gradient-accumulation-steps 4 \
  --use-mixed-precision \
  --shuffle-buffer-size 5000
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

## Common Usage Patterns

### Pattern 1: Large Dataset, Limited Memory
When training on a large dataset with limited memory:

```bash
python -m src.training.train_advanced \
  --dataset-id "your/large-dataset" \
  --model-preset small \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --shuffle-buffer-size 1000 \
  --use-mixed-precision \
  --auto-vocab-size
```

**Why this works:**
- Small batch size (8) keeps per-batch memory low
- Gradient accumulation (8x) gives effective batch size of 64
- Small shuffle buffer reduces memory overhead
- Mixed precision (bfloat16) cuts memory by ~50%

### Pattern 2: Maximum Quality Training
For best model quality with abundant resources:

```bash
python -m src.training.train_advanced \
  --dataset-id "your/dataset" \
  --model-preset large \
  --batch-size 64 \
  --gradient-accumulation-steps 2 \
  --learning-rate 3e-4 \
  --warmup-steps 1000 \
  --epochs 10 \
  --val-split "validation" \
  --eval-every 250 \
  --save-every 500
```

### Pattern 3: Rapid Experimentation
For quick iteration during development:

```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset nano \
  --batch-size 64 \
  --epochs 1 \
  --max-train-examples 10000 \
  --auto-vocab-size \
  --save-every 5000
```

### Pattern 4: Cloud Training with Checkpoints
Training with automatic cloud backup:

```bash
python -m src.training.train_advanced \
  --dataset-id "your/dataset" \
  --model-preset medium \
  --epochs 20 \
  --save-every 1000 \
  --cloud-provider "s3" \
  --cloud-bucket "my-training-bucket" \
  --cloud-region "us-west-2" \
  --cloud-prefix "transformer-v1"
```

---

## Parameter Reference

### Dataset Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset-id` | *required* | HuggingFace dataset ID (e.g., "roneneldan/TinyStories") |
| `--text-column` | "text" | Column name containing text data |
| `--dataset-config` | None | Dataset configuration name (if needed) |
| `--split` | "train" | Training data split |
| `--val-split` | None | Validation split name |
| `--max-train-examples` | None | Limit training examples (useful for testing) |
| `--max-val-examples` | None | Limit validation examples |

### Tokenizer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--auto-vocab-size` | False | Auto-determine optimal vocab size |
| `--vocab-size` | 16000 | Vocabulary size (ignored if auto-vocab-size) |
| `--min-vocab-size` | 5000 | Minimum vocab size for auto mode |
| `--max-vocab-size` | 50000 | Maximum vocab size for auto mode |
| `--tokenizer-train-examples` | 10000 | Examples to sample for tokenizer training |
| `--retrain-tokenizer` | False | Force retrain even if tokenizer exists |
| `--tokenizer-path` | None | Custom path to save/load tokenizer |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-preset` | None | Use preset architecture (nano/tiny/small/medium/large/xlarge) |
| `--d-model` | 512 | Hidden dimension size |
| `--num-layers` | 6 | Number of transformer layers |
| `--num-heads` | 8 | Number of attention heads |
| `--d-ff` | 2048 | Feed-forward network dimension |
| `--seq-len` | 256 | Maximum sequence length |
| `--dropout-rate` | 0.1 | Dropout rate |
| `--activation` | "gelu" | Activation function (relu/gelu/silu) |
| `--no-rmsnorm` | False | Disable RMSNorm (use LayerNorm) |
| `--no-swiglu` | False | Disable SwiGLU activation |
| `--no-rope` | False | Disable rotary position embeddings |
| `--use-lora` | False | Enable LoRA fine-tuning |
| `--lora-rank` | 8 | LoRA rank |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 32 | Batch size per device |
| `--gradient-accumulation-steps` | 1 | Gradient accumulation steps |
| `--learning-rate` | 5e-4 | Peak learning rate |
| `--weight-decay` | 0.01 | AdamW weight decay |
| `--warmup-steps` | 500 | Learning rate warmup steps |
| `--grad-clip` | 1.0 | Gradient clipping norm |

### Data Loading Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--no-streaming` | False | Disable streaming (load all into memory) |
| `--shuffle-buffer-size` | 10000 | Size of shuffle buffer |
| `--stride` | None | Stride for overlapping sequences |

### Optimization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-mixed-precision` | False | Enable bfloat16 mixed precision |

### Checkpoint & Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output-dir` | "output_advanced" | Output directory for checkpoints and logs |
| `--log-every` | 10 | Log training metrics every N steps |
| `--save-every` | 1000 | Save checkpoint every N steps |
| `--eval-every` | 500 | Run validation every N steps |
| `--resume-from` | None | Path to checkpoint to resume from |

### Cloud Storage

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cloud-provider` | None | Cloud provider (s3/gcs/azure) |
| `--cloud-bucket` | None | Cloud bucket name |
| `--cloud-region` | None | Cloud region |
| `--cloud-prefix` | "checkpoints" | Prefix for cloud uploads |
| `--azure-sas-token` | None | Azure SAS token |
| `--azure-account-name` | None | Azure storage account name |

---

## Just-in-Time Processing Explained

### What is Just-in-Time Processing?

Traditional training workflow:
```
1. Load entire dataset into memory
2. Tokenize ALL text → create sequences → store arrays
3. Start training
4. Feed batches from pre-processed arrays
```

Just-in-time workflow (train_advanced.py):
```
1. Connect to dataset (no loading)
2. Start training immediately
3. For each batch:
   - Fetch raw text
   - Tokenize on-the-fly
   - Create sequences
   - Feed to model
4. Discard batch, move to next
```

### How It's Implemented

Under the hood, `train_advanced.py` uses `StreamingDataLoader`:

```python
# This happens automatically when you use train_advanced.py
train_loader = StreamingDataLoader(
    dataset_id="roneneldan/TinyStories",
    tokenizer=tokenizer,
    seq_len=256,
    batch_size=32,
    streaming=True,  # ← Key parameter!
)

# During training - only processes current batch
for batch in train_loader.get_epoch_iterator():
    loss, grads = train_step(state, batch, rng)
    # Batch is processed and discarded
    # Next batch is tokenized on-demand
```

### Memory Comparison

**Pre-processing approach** (e.g., `prepare_sequences`):
```
Memory = vocab_size * num_sequences * seq_len * 4 bytes
Example: 10,000 sequences × 256 tokens × 4 bytes = 10.2 MB
```

**Just-in-time approach** (StreamingDataLoader):
```
Memory = batch_size * seq_len * 2 arrays * 4 bytes
Example: 32 × 256 × 2 × 4 bytes = 65.5 KB
```

**Memory savings: ~99%** for large datasets!

---

## Memory Optimization

### Understanding Memory Usage

Total memory during training:
```
Total = Model Parameters + Optimizer State + Batch Data + Activations
```

**Model Parameters:**
- Float32: params × 4 bytes
- Example: 50M params = 200 MB

**Optimizer State (AdamW):**
- Stores momentum and variance for each parameter
- Memory = params × 2 × 4 bytes
- Example: 50M params = 400 MB

**Batch Data:**
- Input + Labels = batch_size × seq_len × 2 × 4 bytes
- Example: 32 × 256 × 2 × 4 = 65 KB

**Activations:**
- Grows with model size and batch size
- Rough estimate: ~10-20× model params memory

### Memory Reduction Strategies

#### 1. Reduce Batch Size + Gradient Accumulation
```bash
# Instead of batch_size=128
--batch-size 32 --gradient-accumulation-steps 4
# Same effective batch size (128) but 4× less memory per step
```

#### 2. Use Mixed Precision
```bash
--use-mixed-precision
# Cuts activation memory by ~50%
```

#### 3. Reduce Shuffle Buffer
```bash
--shuffle-buffer-size 1000
# Instead of default 10000
# Reduces data loading memory
```

#### 4. Choose Smaller Model Preset
```bash
--model-preset tiny  # Instead of large
```

#### 5. Reduce Sequence Length
```bash
--seq-len 128  # Instead of 256
# Memory scales quadratically with seq_len due to attention
```

### Example: Training on Limited Memory

For 8GB RAM:
```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset tiny \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --seq-len 128 \
  --use-mixed-precision \
  --shuffle-buffer-size 1000 \
  --auto-vocab-size
```

---

## Best Practices

### 1. Start Small, Scale Up
Begin with `nano` or `tiny` preset to verify everything works:
```bash
# First run - quick validation
python -m src.training.train_advanced \
  --dataset-id "your/dataset" \
  --model-preset nano \
  --epochs 1 \
  --max-train-examples 1000

# If successful, scale up
python -m src.training.train_advanced \
  --dataset-id "your/dataset" \
  --model-preset medium \
  --epochs 10
```

### 2. Use Validation Split
Always use validation to prevent overfitting:
```bash
--val-split "validation" \
--eval-every 500
```

The script automatically saves the best model based on validation loss.

### 3. Tune Learning Rate
Default (5e-4) works well for most cases, but adjust based on:
- **Smaller models (nano/tiny)**: Try 1e-3
- **Larger models (large/xlarge)**: Try 3e-4
- **Fine-tuning**: Try 1e-4 to 5e-5

### 4. Adjust Warmup Steps
Rule of thumb: warmup_steps = 10% of total steps
```bash
# For ~10,000 total steps
--warmup-steps 1000
```

### 5. Monitor Training Metrics
Watch for:
- **Loss decreasing steadily** = good
- **Loss plateauing** = may need more epochs or higher LR
- **Loss oscillating** = reduce learning rate
- **Validation loss increasing while training loss decreasing** = overfitting

### 6. Use Cloud Checkpointing for Long Runs
For training that takes hours/days:
```bash
--cloud-provider "s3" \
--cloud-bucket "your-bucket" \
--save-every 500
```

### 7. Tokenizer Reuse
The tokenizer is saved to `output_dir/tokenizer.json` and reused automatically on subsequent runs:
```bash
# First run - trains tokenizer
python -m src.training.train_advanced --dataset-id "..." --auto-vocab-size

# Second run - reuses tokenizer
python -m src.training.train_advanced --dataset-id "..." --resume-from "..."
```

To force retraining:
```bash
--retrain-tokenizer
```

---

## Output Files

After training, `output_dir/` contains:

```
output_advanced/
├── tokenizer.json              # Trained tokenizer
├── model_config.json           # Model configuration
├── final_checkpoint/           # Final model parameters
├── best_checkpoint/            # Best validation checkpoint (if val_split used)
├── checkpoint_step_1000/       # Periodic checkpoints
├── checkpoint_step_2000/
└── ...
```

### Using Trained Model

Load checkpoint for inference:
```python
from orbax import checkpoint as ocp
from tokenizers import Tokenizer
from src.models.model import ProductionTransformer
from src.config.config import ModelConfig

# Load config
with open("output_advanced/model_config.json") as f:
    config_dict = json.load(f)
    config = ModelConfig(**config_dict)

# Load model
model = ProductionTransformer(config=config)
checkpointer = ocp.PyTreeCheckpointer()
params = checkpointer.restore("output_advanced/final_checkpoint")

# Load tokenizer
tokenizer = Tokenizer.from_file("output_advanced/tokenizer.json")

# Generate text (see GENERATION_GUIDE.md for details)
```

---

## Troubleshooting

### Out of Memory Errors
1. Reduce `--batch-size`
2. Increase `--gradient-accumulation-steps` to maintain effective batch size
3. Use `--use-mixed-precision`
4. Reduce `--seq-len`
5. Reduce `--shuffle-buffer-size`

### Training Too Slow
1. Increase `--batch-size` (if memory allows)
2. Use TPU/GPU instead of CPU
3. Reduce `--eval-every` and `--save-every`
4. Ensure `--no-streaming` is NOT set (streaming is default)

### Loss Not Decreasing
1. Increase `--learning-rate` (try 1e-3)
2. Reduce `--weight-decay`
3. Check data quality
4. Increase model size (try next preset up)

### Dataset Loading Errors
1. Verify dataset exists: `https://huggingface.co/datasets/YOUR-DATASET`
2. Check `--text-column` matches dataset
3. Try `--streaming True` explicitly
4. Check internet connection

---

## Summary

`train_advanced.py` provides a complete, production-ready training solution:

✅ **Just-in-time processing** - No pre-processing, memory-efficient  
✅ **Auto vocabulary** - No manual tuning needed  
✅ **Streaming datasets** - Train on any size dataset  
✅ **Model presets** - Quick experimentation  
✅ **Advanced features** - Gradient accumulation, validation, cloud storage  
✅ **One command** - Everything via command-line args  

**Start training with a single command:**
```bash
python -m src.training.train_advanced \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset tiny \
  --auto-vocab-size \
  --epochs 3
```

For more details, see:
- [Model Presets Guide](PRESETS_QUICK_REFERENCE.md)
- [Memory Optimization Guide](MEMORY_OPTIMIZATION_GUIDE.md)
- [Generation Guide](GENERATION_GUIDE.md)
