# Memory Optimization Guide

## 🚀 Quick Start - Use the Optimized Training Script

**For maximum memory efficiency, use the new optimized training script:**

```bash
# Memory-optimized training with streaming data
python train_optimized.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 1000000 \
    --batch-size 16 \
    --gradient-accumulation-steps 4 \
    --use-streaming \
    --epochs 3
```

## 📊 Memory Optimization Techniques Implemented

### 1. **Streaming Data Loading** (90-99% Memory Reduction)
**Problem**: Loading entire dataset into RAM
**Solution**: Stream data on-demand, load only current batch

**Before**:
```python
# OLD METHOD - Loads ALL data into memory
inputs, targets = dataset_loader.prepare_sequences(...)
inputs = jnp.array(inputs)  # ❌ Entire dataset in RAM!
```

**After**:
```python
# NEW METHOD - Stream data batch by batch
from streaming_data_loader import StreamingDataLoader

data_loader = StreamingDataLoader(
    dataset_id="...",
    tokenizer=tokenizer,
    seq_len=128,
    batch_size=32,
    streaming=True  # ✅ Only current batch in RAM!
)

for batch in data_loader:
    # Process one batch at a time
    train_step(state, batch)
```

**Memory Savings**:
- Old: 100K sequences × 128 tokens × 2 (input+target) × 4 bytes = **~100 MB**
- New: 32 sequences × 128 tokens × 2 × 4 bytes = **~0.03 MB**
- **Savings: 99.97%**

### 2. **Gradient Accumulation** (Simulate Large Batches)
**Problem**: Large batch sizes require more memory
**Solution**: Accumulate gradients over multiple small batches

```python
# Train with effective batch size of 128 using only 32 memory
python train_optimized.py \
    --batch-size 32 \
    --gradient-accumulation-steps 4  # Effective batch = 32 × 4 = 128
```

**How it works**:
1. Forward pass on batch 1 (batch_size=32)
2. Backward pass, accumulate gradients
3. Forward pass on batch 2
4. Backward pass, accumulate gradients
5. ...repeat for N steps
6. Apply accumulated gradients once

**Memory**: Same as batch_size=32, but training quality of batch_size=128!

### 3. **Mixed Precision Training** (50% Memory Reduction)
**Problem**: float32 uses 4 bytes per parameter
**Solution**: Use bfloat16 (2 bytes) for most operations

```python
python train_optimized.py \
    --use-mixed-precision  # Enable bfloat16 training
```

**Benefits**:
- **50% less memory** for activations and gradients
- **Faster computation** on TPU/modern GPUs
- Same training stability as float32

**Recommended for**: TPU, A100/H100 GPUs, Apple Silicon

### 4. **8-bit Optimizer States** (60% Memory Reduction)
**Problem**: AdamW stores 2 optimizer states (momentum, variance)
**Solution**: Store optimizer states in 8-bit precision

```python
python train_optimized.py \
    --use-8bit-optimizer  # Enable 8-bit optimizer states
```

**Memory Breakdown**:
- Model parameters: 1M params × 4 bytes = 4 MB
- Standard AdamW states: 1M × 2 states × 4 bytes = 8 MB
- **8-bit states**: 1M × 2 states × 1 byte = 2 MB
- **Savings: 75% on optimizer memory**

### 5. **On-the-Fly Tokenization**
**Problem**: Storing pre-tokenized data uses memory
**Solution**: Tokenize during data iteration

**Before**:
```python
# Pre-tokenize everything
all_tokens = []
for text in texts:
    tokens = tokenizer.encode(text).ids
    all_tokens.extend(tokens)  # ❌ Store all tokens
```

**After**:
```python
# Tokenize on-demand
for text in dataset:
    tokens = tokenizer.encode(text).ids  # ✅ Tokenize once, discard
    yield create_batch(tokens)
```

### 6. **Smaller Shuffle Buffers**
**Problem**: Large shuffle buffers use memory
**Solution**: Use smaller buffers (still maintains randomness)

```python
python train_optimized.py \
    --shuffle-buffer-size 10000  # Balance between memory and randomness
```

## 📈 Memory Usage Comparison

### Example: Training 1M Parameter Model on 1GB Dataset

| Method | Data Memory | Model Memory | Optimizer Memory | Total |
|--------|-------------|--------------|------------------|-------|
| **Original** | 1024 MB | 4 MB | 8 MB | **1036 MB** |
| **+ Streaming** | 0.03 MB | 4 MB | 8 MB | **12 MB** |
| **+ Mixed Precision** | 0.03 MB | 2 MB | 4 MB | **6 MB** |
| **+ 8-bit Optimizer** | 0.03 MB | 2 MB | 0.5 MB | **2.5 MB** |
| **Savings** | 99.97% | 50% | 93.75% | **99.76%** |

## 🎯 Recommended Settings by Hardware

### Limited RAM (< 8GB)
```bash
python train_optimized.py \
    --dataset-id your-dataset \
    --batch-size 8 \
    --gradient-accumulation-steps 8 \
    --use-streaming \
    --use-mixed-precision \
    --use-8bit-optimizer \
    --shuffle-buffer-size 5000
```

### Medium RAM (8-16GB)
```bash
python train_optimized.py \
    --dataset-id your-dataset \
    --batch-size 32 \
    --gradient-accumulation-steps 4 \
    --use-streaming \
    --use-mixed-precision \
    --shuffle-buffer-size 10000
```

### High RAM (>16GB)
```bash
python train_optimized.py \
    --dataset-id your-dataset \
    --batch-size 64 \
    --gradient-accumulation-steps 2 \
    --use-streaming \
    --shuffle-buffer-size 50000
```

### TPU (Optimize for Speed)
```bash
python train_optimized.py \
    --dataset-id your-dataset \
    --batch-size 128 \
    --gradient-accumulation-steps 1 \
    --use-streaming \
    --use-mixed-precision \
    --shuffle-buffer-size 100000
```

## 🔧 Advanced Optimizations

### 1. Reduce Sequence Length
```bash
# Shorter sequences = less memory
--seq-len 64  # Instead of 128 or 256
```

### 2. Smaller Model
```bash
# Fewer parameters = less memory
--target-params 500000  # Instead of 1M
```

### 3. Reduce Attention Heads
```bash
# Modify model_sizing.py to prefer fewer heads
# Attention memory scales with num_heads
```

### 4. Use Strided Sequences
```bash
# Create non-overlapping sequences
--stride 128  # Same as seq_len = no overlap
```

### 5. Limit Dataset Size
```bash
# Use subset for faster iteration
--max-examples 100000
```

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

**Symptom**: Training crashes with OOM error

**Solutions** (in order):
1. **Enable streaming**: `--use-streaming`
2. **Reduce batch size**: `--batch-size 8` (use gradient accumulation to compensate)
3. **Enable mixed precision**: `--use-mixed-precision`
4. **Enable 8-bit optimizer**: `--use-8bit-optimizer`
5. **Reduce sequence length**: `--seq-len 64`
6. **Reduce model size**: `--target-params 500000`
7. **Limit dataset**: `--max-examples 50000`

### Slow Training

**Symptom**: Training is very slow

**Solutions**:
1. **Increase batch size** (if memory allows): `--batch-size 64`
2. **Use gradient accumulation** for larger effective batch: `--gradient-accumulation-steps 4`
3. **Enable mixed precision** (faster on TPU/GPU): `--use-mixed-precision`
4. **Increase shuffle buffer** (faster data loading): `--shuffle-buffer-size 50000`
5. **Check backend**: TPU/GPU much faster than CPU

### Data Not Shuffling Well

**Symptom**: Model not learning or overfitting

**Solutions**:
1. **Increase shuffle buffer**: `--shuffle-buffer-size 50000`
2. **Multiple epochs**: More epochs = more shuffling
3. **Change random seed**: `--seed 123`

## 📝 Code Examples

### Basic Usage
```python
from streaming_data_loader import StreamingDataLoader
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

# Create streaming data loader
loader = StreamingDataLoader(
    dataset_id="skeskinen/TinyStories-Instruct-hf",
    tokenizer=tokenizer,
    seq_len=128,
    batch_size=32,
    streaming=True  # KEY: Enable streaming
)

# Iterate through data
for batch in loader:
    # batch contains only 32 sequences at a time!
    print(batch["input_ids"].shape)  # (32, 128)
    # Process batch...
```

### Estimate Memory Usage
```python
from streaming_data_loader import estimate_memory_usage

# See memory usage for different strategies
usage = estimate_memory_usage(
    num_sequences=100000,
    seq_len=128,
    batch_size=32
)

print(f"Full load: {usage['full_load_mb']} MB")
print(f"Streaming: {usage['streaming_mb']} MB")
print(f"Savings: {usage['memory_savings']}")
```

## 🎓 Best Practices

### ✅ DO:
- Always use `--use-streaming` for datasets > 100MB
- Use gradient accumulation for better training with small batches
- Enable mixed precision on TPU/modern GPUs
- Monitor memory usage during training
- Start with small batch size, increase gradually
- Use smaller shuffle buffers for very large datasets

### ❌ DON'T:
- Load entire dataset into memory if dataset > available RAM
- Use batch_size > available memory allows
- Disable streaming for large datasets
- Use overlapping sequences (stride < seq_len) unless necessary
- Pre-tokenize and store all data

## 📊 Monitoring Memory

### Check Memory During Training

**JAX/TPU**:
```python
import jax
print(jax.local_devices())  # See available devices
```

**System RAM**:
```bash
# Linux/Mac
htop
# or
watch -n 1 free -h

# During training, watch for memory usage
```

**GPU Memory**:
```bash
# NVIDIA GPUs
watch -n 1 nvidia-smi

# Look for GPU memory usage %
```

## 🚀 Migration Guide

### Migrating from Old Training Script

**Old**:
```bash
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --batch-size 32 \
    --epochs 3
```

**New (Memory Optimized)**:
```bash
python train_optimized.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --batch-size 32 \
    --use-streaming \
    --gradient-accumulation-steps 1 \
    --epochs 3
```

### Key Differences:
1. **Streaming enabled by default** in `train_optimized.py`
2. **Gradient accumulation** available
3. **Mixed precision** support
4. **8-bit optimizer** option
5. **Better memory monitoring**

## 📚 Additional Resources

### Files:
- `train_optimized.py` - Memory-optimized training script
- `streaming_data_loader.py` - Streaming data loader implementation
- `hf_dataset_loader.py` - Original data loader (updated with streaming)

### Documentation:
- JAX memory management: https://jax.readthedocs.io/en/latest/
- HuggingFace datasets streaming: https://huggingface.co/docs/datasets/stream
- Mixed precision training: https://pytorch.org/docs/stable/amp.html

---

## Summary

**Memory optimization is achieved through:**
1. ✅ **Streaming data loading** - Only load current batch
2. ✅ **Gradient accumulation** - Simulate large batches
3. ✅ **Mixed precision** - Use bfloat16/float16
4. ✅ **8-bit optimizer** - Compress optimizer states
5. ✅ **On-the-fly tokenization** - Don't store tokenized data
6. ✅ **Smaller buffers** - Reduce shuffle buffer size

**Result**: **Train models with 100x less memory!**

Use `train_optimized.py` for all training tasks going forward.
