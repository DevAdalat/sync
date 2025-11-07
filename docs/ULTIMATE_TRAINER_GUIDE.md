# Ultimate Transformer Trainer Guide

## Overview

The **Ultimate Transformer Trainer** (`train_ultimate.py`) is a comprehensive, all-in-one training solution that combines the best features from all existing trainers into a single, highly customizable script.

## 🌟 Key Features

### Model Flexibility
- **Model Presets**: Choose from nano, tiny, small, medium, large, or xlarge architectures
- **Custom Architecture**: Define your own model parameters (d_model, layers, heads, FFN size)
- **Parameter-Based Sizing**: Specify target parameter count and let the system optimize the architecture
- **Variable Model Sizes**: Create models from 1M to 1B+ parameters

### Advanced Optimizations
- **Flash Attention (Kvax)**: 2-3x faster attention for long sequences
- **RoPE**: Rotary Position Embeddings for better position encoding
- **RMSNorm**: More efficient normalization than LayerNorm  
- **SwiGLU**: Powerful activation function for FFN
- **LoRA**: Parameter-efficient fine-tuning
- **Mixed Precision**: bfloat16 training for TPU/GPU
- **Gradient Accumulation**: Train with larger effective batch sizes
- **Multi-Device Training**: Automatic data parallelism across TPU/GPU/CPU

### Memory Optimizations
- **Streaming Data Loading**: Minimal memory footprint
- **Efficient Shuffling**: Buffer-based shuffling for large datasets
- **Batch Size Control**: Fine-tune memory usage
- **Gradient Accumulation**: Simulate large batches with limited memory

### Training Features
- **Auto Vocab Size**: Automatically determine optimal vocabulary size
- **Learning Rate Scheduling**: Warmup + Cosine decay
- **Gradient Clipping**: Stable training
- **Validation During Training**: Monitor performance
- **Checkpoint Saving/Resuming**: Never lose progress
- **Cloud Storage**: S3, GCS, Azure Blob support
- **Real-time Logging**: Monitor training progress

## 🚀 Quick Start

### 1. Basic Training with Preset

```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset tiny \
  --auto-vocab-size \
  --epochs 3
```

### 2. Custom Architecture

```bash
python -m src.training.train_ultimate \
  --dataset-id "wikitext" \
  --dataset-config "wikitext-2-raw-v1" \
  --d-model 512 \
  --num-layers 8 \
  --num-heads 8 \
  --d-ff 2048 \
  --vocab-size 16000 \
  --seq-len 256 \
  --epochs 5
```

### 3. Parameter-Based Sizing

```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --target-params 10000000 \
  --prefer-depth \
  --epochs 5
```

## 📖 Detailed Usage

### Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset-id` | str | **required** | HuggingFace dataset ID |
| `--text-column` | str | `"text"` | Column containing text data |
| `--dataset-config` | str | `None` | Dataset configuration/subset |
| `--split` | str | `"train"` | Dataset split to use |
| `--val-split` | str | `None` | Validation split (optional) |
| `--max-train-examples` | int | `None` | Limit training examples |
| `--max-val-examples` | int | `None` | Limit validation examples |

### Tokenizer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--auto-vocab-size` | flag | `False` | Auto-determine optimal vocab size |
| `--vocab-size` | int | `16000` | Vocabulary size |
| `--min-vocab-size` | int | `5000` | Minimum vocab size (auto mode) |
| `--max-vocab-size` | int | `50000` | Maximum vocab size (auto mode) |
| `--tokenizer-train-examples` | int | `10000` | Examples for training tokenizer |
| `--retrain-tokenizer` | flag | `False` | Force retrain tokenizer |
| `--tokenizer-path` | str | `None` | Custom tokenizer path |

### Model Parameters

#### Model Presets

Use `--model-preset` to select from predefined architectures:

| Preset | Parameters | d_model | Layers | Heads | FFN | Description |
|--------|------------|---------|--------|-------|-----|-------------|
| `nano` | ~1M | 128 | 6 | 4 | 512 | Tiny model for testing |
| `tiny` | ~5M | 256 | 8 | 8 | 1024 | Small model for resource-constrained environments |
| `small` | ~50M | 512 | 12 | 8 | 2048 | Balanced model for general tasks |
| `medium` | ~125M | 768 | 12 | 12 | 3072 | Similar to GPT-2 Small / BERT Base |
| `large` | ~350M | 1024 | 24 | 16 | 4096 | Large model for high-quality generation |
| `xlarge` | ~700M | 1280 | 32 | 20 | 5120 | Extra large model for production use |

Example:
```bash
--model-preset medium
```

#### Parameter-Based Sizing

Use `--target-params` to specify desired parameter count:

```bash
--target-params 10000000  # 10M parameters
--prefer-depth             # Prefer deeper over wider
```

#### Custom Architecture

Manually specify all model dimensions:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--d-model` | int | `512` | Model hidden dimension |
| `--num-layers` | int | `6` | Number of transformer layers |
| `--num-heads` | int | `8` | Number of attention heads |
| `--d-ff` | int | `2048` | Feed-forward dimension |
| `--seq-len` | int | `256` | Maximum sequence length |
| `--dropout-rate` | float | `0.1` | Dropout rate |
| `--activation` | str | `"gelu"` | Activation function (relu/gelu/silu) |

#### Model Optimizations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--no-rmsnorm` | flag | `False` | Disable RMSNorm (use LayerNorm) |
| `--no-swiglu` | flag | `False` | Disable SwiGLU (use standard FFN) |
| `--no-rope` | flag | `False` | Disable RoPE (use learned positions) |
| `--no-flash-attention` | flag | `False` | Disable Flash Attention |
| `--use-lora` | flag | `False` | Enable LoRA fine-tuning |
| `--lora-rank` | int | `8` | LoRA rank |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | `3` | Number of training epochs |
| `--batch-size` | int | `32` | Training batch size |
| `--gradient-accumulation-steps` | int | `1` | Gradient accumulation steps |
| `--learning-rate` | float | `5e-4` | Peak learning rate |
| `--weight-decay` | float | `0.01` | Weight decay for AdamW |
| `--warmup-steps` | int | `500` | Learning rate warmup steps |
| `--grad-clip` | float | `1.0` | Gradient clipping norm |

### Data Loading Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-streaming` | flag | `True` | Use streaming data loader |
| `--shuffle-buffer-size` | int | `10000` | Shuffle buffer size |
| `--stride` | int | `None` | Sequence stride (default: seq_len) |

### Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-mixed-precision` | flag | `False` | Use bfloat16 precision |

### Checkpoint & Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output-dir` | str | `"output_ultimate"` | Output directory |
| `--log-every` | int | `10` | Log every N steps |
| `--save-every` | int | `1000` | Save checkpoint every N steps |
| `--eval-every` | int | `500` | Evaluate every N steps |
| `--resume-from` | str | `None` | Resume from checkpoint path |

### Cloud Storage (Optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cloud-provider` | str | `None` | Cloud provider (s3/gcs/azure) |
| `--cloud-bucket` | str | `None` | Cloud bucket/container name |
| `--cloud-region` | str | `None` | Cloud region |
| `--cloud-prefix` | str | `"checkpoints"` | Cloud storage prefix |
| `--azure-sas-token` | str | `None` | Azure SAS token |
| `--azure-account-name` | str | `None` | Azure storage account |

## 💡 Usage Examples

### Example 1: Quick Test with Tiny Model

Train a tiny model quickly for testing:

```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset nano \
  --auto-vocab-size \
  --epochs 2 \
  --batch-size 16 \
  --max-train-examples 10000 \
  --log-every 5
```

### Example 2: Medium Model with Validation

Train a medium-sized model with validation:

```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset medium \
  --val-split "validation" \
  --auto-vocab-size \
  --epochs 10 \
  --batch-size 64 \
  --eval-every 500 \
  --save-every 1000 \
  --output-dir output_medium_model
```

### Example 3: Custom Large Model

Train a custom large model:

```bash
python -m src.training.train_ultimate \
  --dataset-id "wikitext" \
  --dataset-config "wikitext-103-raw-v1" \
  --d-model 1024 \
  --num-layers 24 \
  --num-heads 16 \
  --d-ff 4096 \
  --vocab-size 32000 \
  --seq-len 512 \
  --epochs 20 \
  --batch-size 32 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --warmup-steps 1000
```

### Example 4: Memory-Optimized Training

Train with memory constraints:

```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset large \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --use-streaming \
  --shuffle-buffer-size 5000 \
  --epochs 5
```

### Example 5: TPU Training with Cloud Storage

Train on TPU with cloud checkpointing:

```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset large \
  --batch-size 128 \
  --use-mixed-precision \
  --cloud-provider gcs \
  --cloud-bucket my-tpu-checkpoints \
  --cloud-region us-central1 \
  --epochs 10
```

### Example 6: Resume Training

Resume from a previous checkpoint:

```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset medium \
  --resume-from output_ultimate/checkpoint_step_5000 \
  --epochs 10
```

### Example 7: Parameter-Based Sizing

Create a 100M parameter model:

```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --target-params 100000000 \
  --prefer-depth \
  --auto-vocab-size \
  --epochs 10 \
  --batch-size 64
```

### Example 8: LoRA Fine-Tuning

Fine-tune with LoRA:

```bash
python -m src.training.train_ultimate \
  --dataset-id "my-custom-dataset" \
  --model-preset medium \
  --use-lora \
  --lora-rank 16 \
  --learning-rate 1e-3 \
  --epochs 5
```

## 🎯 Model Selection Guide

### Choose by Task

**Text Classification / Short Sequences:**
- Use: `tiny` or `small` preset
- Sequence length: 128-256

**General Text Generation:**
- Use: `medium` preset
- Sequence length: 256-512

**High-Quality Generation / Long Context:**
- Use: `large` or `xlarge` preset
- Sequence length: 512-2048

**Research / Experimentation:**
- Use: Custom architecture or parameter-based sizing
- Adjust based on compute budget

### Choose by Compute

**Limited Resources (CPU/Small GPU):**
```bash
--model-preset nano
--batch-size 8
--gradient-accumulation-steps 4
```

**Medium Resources (Single GPU):**
```bash
--model-preset small
--batch-size 32
--gradient-accumulation-steps 2
```

**High Resources (Multi-GPU/TPU):**
```bash
--model-preset large
--batch-size 128
--use-mixed-precision
```

## 📊 Performance Tips

### Maximize Throughput

1. **Use appropriate batch size**:
   ```bash
   --batch-size 64  # Adjust based on GPU memory
   ```

2. **Enable flash attention**:
   ```bash
   # Flash attention is enabled by default
   # Provides 2-3x speedup for long sequences
   ```

3. **Use gradient accumulation**:
   ```bash
   --batch-size 16
   --gradient-accumulation-steps 4
   # Effective batch size = 64
   ```

4. **Enable mixed precision on TPU/GPU**:
   ```bash
   --use-mixed-precision
   ```

### Minimize Memory Usage

1. **Use streaming**:
   ```bash
   --use-streaming
   ```

2. **Reduce batch size with gradient accumulation**:
   ```bash
   --batch-size 8
   --gradient-accumulation-steps 8
   ```

3. **Smaller shuffle buffer**:
   ```bash
   --shuffle-buffer-size 5000
   ```

4. **Shorter sequences**:
   ```bash
   --seq-len 128
   ```

### Optimize Learning

1. **Proper warmup**:
   ```bash
   --warmup-steps 1000  # ~1-5% of total steps
   ```

2. **Learning rate tuning**:
   ```bash
   --learning-rate 5e-4  # Default, adjust based on model size
   ```

3. **Weight decay**:
   ```bash
   --weight-decay 0.01  # Regularization
   ```

4. **Gradient clipping**:
   ```bash
   --grad-clip 1.0  # Prevents exploding gradients
   ```

## 🔍 Monitoring Training

### Output Structure

```
output_ultimate/
├── tokenizer.json              # Trained tokenizer
├── model_config.json           # Model configuration
├── training_summary.json       # Training statistics
├── best_checkpoint/            # Best validation checkpoint
├── final_checkpoint/           # Final checkpoint
└── checkpoint_step_XXXX/       # Periodic checkpoints
```

### Training Logs

The trainer provides real-time logging:

```
Step    100 │ Loss: 3.4567 │ Tokens/sec:    45,678 │ Time:   123.4s
Step    200 │ Loss: 3.1234 │ Tokens/sec:    46,123 │ Time:   246.8s
```

### Validation Logs

When using validation:

```
  Validation
  ─────────────────────────────
  Validation loss: 2.8901 (50 batches)
  ✓ New best validation loss! Saving to: output_ultimate/best_checkpoint
```

## 🐛 Troubleshooting

### Out of Memory

**Problem**: GPU/TPU runs out of memory

**Solutions**:
1. Reduce batch size: `--batch-size 8`
2. Use gradient accumulation: `--gradient-accumulation-steps 4`
3. Reduce sequence length: `--seq-len 128`
4. Use streaming: `--use-streaming`
5. Smaller model: Use `--model-preset tiny`

### Slow Training

**Problem**: Training is too slow

**Solutions**:
1. Increase batch size: `--batch-size 128`
2. Use flash attention (enabled by default)
3. Enable mixed precision: `--use-mixed-precision`
4. Use multiple devices (automatic)
5. Larger shuffle buffer: `--shuffle-buffer-size 50000`

### Poor Convergence

**Problem**: Loss not decreasing

**Solutions**:
1. Increase warmup: `--warmup-steps 2000`
2. Reduce learning rate: `--learning-rate 1e-4`
3. Increase gradient clipping: `--grad-clip 5.0`
4. Check data quality
5. Train longer: `--epochs 20`

### Checkpoint Issues

**Problem**: Cannot resume from checkpoint

**Solutions**:
1. Use absolute path: `--resume-from /full/path/to/checkpoint`
2. Ensure checkpoint directory exists
3. Check cloud credentials (if using cloud storage)

## 🌐 Multi-Device Training

The trainer automatically detects and uses multiple devices:

### CPU
```
Backend: CPU
Devices: 1x cpu
⚠ Running on CPU - training will be slower
```

### Single GPU
```
Backend: GPU
Devices: 1x NVIDIA A100
✓ GPU Optimizations: CUDA acceleration enabled
```

### Multiple GPUs
```
Backend: GPU
Devices: 4x NVIDIA A100
✓ Multi-device training enabled!
✓ Data parallelism across 4 GPUs
Effective batch size: 128 (32 per device)
```

### TPU
```
Backend: TPU
Devices: 8x TPU v4
✓ TPU Optimizations: XLA compilation enabled
Consider using bfloat16 precision
```

## ☁️ Cloud Storage

### AWS S3

```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset medium \
  --cloud-provider s3 \
  --cloud-bucket my-s3-bucket \
  --cloud-region us-west-2 \
  --cloud-prefix my-model/checkpoints
```

### Google Cloud Storage

```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset medium \
  --cloud-provider gcs \
  --cloud-bucket my-gcs-bucket \
  --cloud-region us-central1
```

### Azure Blob Storage

```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset medium \
  --cloud-provider azure \
  --cloud-bucket my-container \
  --azure-account-name mystorageaccount \
  --azure-sas-token "sv=2021-06-08&ss=b&srt=sco&sp=rwdlac..."
```

## 📚 Advanced Topics

### Custom Datasets

Use any HuggingFace dataset:

```bash
--dataset-id "username/my-custom-dataset"
--text-column "content"  # Your text column name
--dataset-config "subset1"  # If dataset has configs
```

### Fine-Tuning Strategies

**Full Fine-Tuning:**
```bash
--model-preset medium
--learning-rate 1e-5
--epochs 3
```

**LoRA Fine-Tuning:**
```bash
--model-preset medium
--use-lora
--lora-rank 16
--learning-rate 1e-3
--epochs 5
```

### Hyperparameter Tuning

**Learning Rate:**
- Tiny models: `1e-3` to `5e-3`
- Small models: `5e-4` to `1e-3`
- Medium models: `1e-4` to `5e-4`
- Large models: `5e-5` to `1e-4`

**Batch Size:**
- Limited memory: `8-16`
- Moderate memory: `32-64`
- High memory: `128-256`
- TPU: `256-512`

**Warmup Steps:**
- Rule of thumb: 1-5% of total steps
- Minimum: `100`
- Typical: `500-2000`

## 🎓 Best Practices

1. **Start Small**: Begin with `--model-preset tiny` to verify everything works

2. **Use Auto Vocab Size**: Let the system determine optimal vocabulary

3. **Enable Flash Attention**: Provides significant speedup (enabled by default)

4. **Monitor Validation**: Use `--val-split` to track overfitting

5. **Save Frequently**: Use `--save-every 1000` to avoid losing progress

6. **Use Streaming**: Always use `--use-streaming` for large datasets

7. **Gradient Accumulation**: Use to simulate larger batches with limited memory

8. **Cloud Backups**: Use cloud storage for important training runs

9. **Resume Training**: Save checkpoints and resume if interrupted

10. **Experiment**: Try different presets and configurations

## 📖 Further Reading

- [Model Presets Guide](MODEL_PRESETS_GUIDE.md)
- [Flash Attention Guide](FLASH_ATTENTION_GUIDE.md)
- [Memory Optimization Guide](MEMORY_OPTIMIZATION_GUIDE.md)
- [Multi-Device Training Guide](MULTI_DEVICE_GUIDE.md)

## 🤝 Support

For issues or questions:
- Check the troubleshooting section
- Review example commands
- Consult other guides in the docs/ folder

---

**Happy Training! 🚀**
