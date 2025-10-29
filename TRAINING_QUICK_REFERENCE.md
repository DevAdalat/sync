# Training Script Quick Reference

## train_model.py - Unified Training Script

A single, comprehensive script that accepts all training parameters via command line or function call.

## Quick Start

### 1. Train on TinyStories-Instruct (Default Settings)
```bash
python train_model.py --dataset-id skeskinen/TinyStories-Instruct-hf
```
Trains a 1M parameter model for 3 epochs with batch size 32.

### 2. Custom Configuration
```bash
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 5000000 \
    --epochs 10 \
    --batch-size 64 \
    --learning-rate 1e-3
```

### 3. Quick Test (Fast)
```bash
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 500000 \
    --max-examples 1000 \
    --epochs 2
```

### 4. Train on WikiText-2
```bash
python train_model.py \
    --dataset-id wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --target-params 3000000 \
    --epochs 5
```

## Use as Python Function

```python
from train_model import train_model

results = train_model(
    dataset_id="skeskinen/TinyStories-Instruct-hf",
    target_params=5_000_000,
    epochs=10,
    batch_size=64,
    learning_rate=1e-3,
    output_dir="my_model"
)

print(f"Training completed!")
print(f"Final loss: {results['final_loss']:.4f}")
print(f"Best loss: {results['best_loss']:.4f}")
```

## Key Parameters

### Dataset
- `--dataset-id`: HuggingFace dataset ID (required)
- `--dataset-config`: Dataset configuration (optional)
- `--text-column`: Text column name (default: "text")
- `--split`: Dataset split (default: "train")

### Model
- `--target-params`: Target model size (default: 1,000,000)
- `--vocab-size`: Vocabulary size (default: 10,000)
- `--seq-len`: Sequence length (default: 128)

### Training
- `--epochs`: Number of epochs (default: 3)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 5e-4)
- `--weight-decay`: Weight decay (default: 0.01)

### Data Processing
- `--max-examples`: Limit examples for testing (default: all)
- `--stride`: Sequence stride (default: seq_len)
- `--tokenizer-train-examples`: Examples for tokenizer (default: 10,000)

### Output
- `--output-dir`: Output directory (default: "output")
- `--log-every`: Logging frequency (default: 10)
- `--seed`: Random seed (default: 42)

## Output Files

After training, these files are saved in `--output-dir`:

1. `best_checkpoint/` - Model parameters (weights)
2. `model_config.json` - Model architecture configuration
3. `tokenizer.json` - Trained tokenizer
4. `training_results.json` - Training statistics

## Training Examples

### Small Model (500K params, quick test)
```bash
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 500000 \
    --max-examples 500 \
    --epochs 2 \
    --batch-size 16
```
Time: 2-3 minutes

### Medium Model (3M params)
```bash
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 3000000 \
    --epochs 5 \
    --batch-size 32
```
Time: 30-60 minutes

### Large Model (10M params)
```bash
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 10000000 \
    --epochs 10 \
    --batch-size 64 \
    --seq-len 256 \
    --learning-rate 3e-4
```
Time: Several hours

### Very Large Model (50M params)
```bash
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 50000000 \
    --epochs 20 \
    --batch-size 128 \
    --seq-len 512 \
    --learning-rate 1e-4 \
    --vocab-size 20000
```
Time: Many hours

## Tips

1. **Start Small**: Test with `--max-examples 1000` first
2. **Memory Issues**: Reduce `--batch-size` or `--seq-len`
3. **Slow Training**: Increase `--batch-size` (if memory allows)
4. **Model Size**: 500K (tiny), 1M (small), 5M (medium), 50M+ (large)
5. **Learning Rate**: Larger models often need smaller learning rates (1e-4 to 3e-4)

## Model Architecture

The script automatically configures the model architecture to match the target parameter count using modern improvements:
- **RMSNorm** instead of LayerNorm (more efficient)
- **Rotary Position Embeddings (RoPE)** instead of learned positions
- **SwiGLU** activation in feed-forward networks (more powerful)

## For More Details

See `TRAIN_MODEL_GUIDE.py` for comprehensive documentation including:
- Detailed parameter explanations
- Troubleshooting guide
- Example workflows
- Training output explanation
- Best practices
