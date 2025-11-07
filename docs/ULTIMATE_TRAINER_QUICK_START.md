# Ultimate Trainer - Quick Start Guide

## 🚀 Fastest Way to Start Training

### 1-Line Training Command

```bash
python -m src.training.train_ultimate --dataset-id "roneneldan/TinyStories" --model-preset tiny --auto-vocab-size --epochs 3
```

That's it! This command will:
- ✅ Download TinyStories dataset
- ✅ Auto-determine optimal vocabulary size
- ✅ Train a tiny transformer model
- ✅ Save checkpoints and configuration
- ✅ Complete in a few minutes

## 📋 Common Use Cases

### Tiny Model (Testing)
```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset nano \
  --epochs 2 \
  --batch-size 16
```

### Small Model (Learning)
```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset small \
  --auto-vocab-size \
  --epochs 5
```

### Medium Model (Production)
```bash
python -m src.training.train_ultimate \
  --dataset-id "roneneldan/TinyStories" \
  --model-preset medium \
  --val-split "validation" \
  --auto-vocab-size \
  --epochs 10 \
  --save-every 1000
```

### Large Model (Research)
```bash
python -m src.training.train_ultimate \
  --dataset-id "wikitext" \
  --dataset-config "wikitext-103-raw-v1" \
  --model-preset large \
  --vocab-size 32000 \
  --seq-len 512 \
  --epochs 20 \
  --batch-size 64
```

## 🎯 Model Size Selection

| Preset | Parameters | Use Case | Training Time (estimate) |
|--------|------------|----------|-------------------------|
| `nano` | ~1M | Quick testing | Minutes |
| `tiny` | ~5M | Learning, prototyping | ~1 hour |
| `small` | ~50M | Small production tasks | ~3-5 hours |
| `medium` | ~125M | General production | ~8-12 hours |
| `large` | ~350M | High-quality generation | ~24-48 hours |
| `xlarge` | ~700M | Research, SOTA | Days |

## 💡 Key Parameters to Know

### Model Selection
```bash
--model-preset tiny          # Use preset (recommended)
# OR
--target-params 10000000     # Specify parameter count
# OR
--d-model 512 --num-layers 8 # Custom architecture
```

### Vocabulary
```bash
--auto-vocab-size           # Auto-determine (recommended)
# OR
--vocab-size 16000          # Manual specification
```

### Training Speed/Memory Tradeoff
```bash
# Fast, high memory:
--batch-size 128

# Slow, low memory:
--batch-size 8 --gradient-accumulation-steps 16
```

### Quality Improvements
```bash
--val-split "validation"    # Add validation
--epochs 10                 # Train longer
--warmup-steps 1000         # Better convergence
```

## 🔧 Common Customizations

### Use Your Own Dataset
```bash
--dataset-id "username/my-dataset" \
--text-column "my_text_column"
```

### Adjust Sequence Length
```bash
--seq-len 128    # Short sequences (faster, less memory)
--seq-len 512    # Long sequences (better quality, more memory)
```

### Save More Frequently
```bash
--save-every 500    # Save every 500 steps
--log-every 10      # Log every 10 steps
```

### Resume Training
```bash
--resume-from output_ultimate/checkpoint_step_5000
```

## 📊 Output Files

After training completes, find these files in `output_ultimate/`:

```
output_ultimate/
├── tokenizer.json              # Your trained tokenizer
├── model_config.json           # Model architecture config
├── training_summary.json       # Training statistics
├── best_checkpoint/            # Best model (if using validation)
├── final_checkpoint/           # Final model
└── checkpoint_step_XXXX/       # Periodic checkpoints
```

## 🐛 Quick Troubleshooting

### "Out of Memory"
```bash
# Solution: Reduce batch size
--batch-size 8 --gradient-accumulation-steps 4
```

### "Training Too Slow"
```bash
# Solution: Increase batch size (if you have memory)
--batch-size 128
```

### "Loss Not Decreasing"
```bash
# Solution: Adjust learning rate and warmup
--learning-rate 1e-4 --warmup-steps 1000
```

## 📚 Next Steps

1. **Start with tiny model**: Test that everything works
2. **Scale up gradually**: Move to small → medium → large
3. **Add validation**: Monitor overfitting with `--val-split`
4. **Save checkpoints**: Use `--save-every` to avoid losing progress
5. **Read full guide**: Check `ULTIMATE_TRAINER_GUIDE.md` for all options

## 🎓 Pro Tips

1. **Always use `--auto-vocab-size`** - It determines the best vocabulary size for your dataset
   
2. **Start small** - Use `--model-preset nano` first to verify your setup works

3. **Use streaming** - It's enabled by default and saves memory

4. **Monitor validation** - Add `--val-split "validation"` to track overfitting

5. **Save checkpoints** - Training can be interrupted, use `--save-every 1000`

6. **Gradient accumulation** - Simulate large batches: `--batch-size 8 --gradient-accumulation-steps 8`

## ⚡ Performance Checklist

- [ ] Used appropriate `--batch-size` for your GPU
- [ ] Enabled flash attention (on by default)
- [ ] Using `--use-streaming` for large datasets
- [ ] Set `--warmup-steps` (typically 500-2000)
- [ ] Chosen appropriate `--seq-len` for your task
- [ ] Using validation split for monitoring
- [ ] Saving checkpoints regularly

## 🔥 Ready to Train?

Pick a command from above and start training! All the import errors you see are normal - they'll resolve once you install dependencies with:

```bash
pip install -r requirements.txt
```

Then run your training command and watch the magic happen! 🎉

---

**For detailed documentation**, see [ULTIMATE_TRAINER_GUIDE.md](ULTIMATE_TRAINER_GUIDE.md)
