"""
train_model.py - Simple Training Script Guide
=============================================

This is a streamlined training script that accepts all training parameters
via command line arguments or can be imported as a Python function.

USAGE EXAMPLES
==============

1. Basic Training with Default Parameters
------------------------------------------
python train_model.py --dataset-id skeskinen/TinyStories-Instruct-hf

This will train a 1M parameter model for 3 epochs with default settings.


2. Custom Model Size and Training Duration
-------------------------------------------
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 5000000 \
    --epochs 10 \
    --batch-size 64 \
    --learning-rate 1e-3


3. Train on WikiText-2
----------------------
python train_model.py \
    --dataset-id wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --target-params 3000000 \
    --epochs 5


4. Quick Test with Small Model
-------------------------------
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 500000 \
    --max-examples 1000 \
    --epochs 2 \
    --batch-size 16


5. Large Model Training
-----------------------
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 50000000 \
    --epochs 20 \
    --batch-size 128 \
    --seq-len 256 \
    --learning-rate 3e-4 \
    --vocab-size 20000


USING AS A PYTHON FUNCTION
===========================

You can also import and call the train_model function directly:

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


ALL AVAILABLE PARAMETERS
=========================

Dataset Configuration:
  --dataset-id          HuggingFace dataset ID (REQUIRED)
                        Examples: 
                          - skeskinen/TinyStories-Instruct-hf
                          - wikitext
                          - roneneldan/TinyStories
  
  --dataset-config      Dataset configuration/subset (optional)
                        Example: wikitext-2-raw-v1
  
  --text-column         Column name containing text data (default: "text")
  
  --split               Dataset split to use (default: "train")

Model Configuration:
  --target-params       Target number of model parameters (default: 1,000,000)
                        Examples: 500000, 1000000, 5000000, 50000000
  
  --vocab-size          Vocabulary size for tokenizer (default: 10,000)
  
  --seq-len             Sequence length (default: 128)
  
  --prefer-depth        Prefer deeper models over wider ones (default: True)

Training Configuration:
  --epochs              Number of training epochs (default: 3)
  
  --batch-size          Training batch size (default: 32)
  
  --learning-rate       Initial learning rate (default: 5e-4)
  
  --weight-decay        Weight decay for AdamW optimizer (default: 0.01)
  
  --warmup-steps        Number of warmup steps (default: 100)
  
  --grad-clip           Gradient clipping norm (default: 1.0)

Data Processing:
  --max-examples        Maximum examples to use from dataset (default: all)
                        Use for quick testing with limited data
  
  --stride              Stride for creating sequences (default: seq_len)
                        Lower stride creates more overlapping sequences
  
  --tokenizer-train-examples  
                        Number of examples to train tokenizer on (default: 10,000)
  
  --retrain-tokenizer   Retrain tokenizer even if one exists

Output and Logging:
  --output-dir          Output directory for checkpoints (default: "output")
  
  --log-every           Log every N steps (default: 10)
  
  --seed                Random seed (default: 42)


OUTPUT FILES
============

After training, the following files are saved in the output directory:

1. best_checkpoint/
   - Model parameters (weights)
   - Load with: checkpointer.restore(checkpoint_path)

2. model_config.json
   - Model architecture configuration
   - Contains: vocab_size, d_model, num_layers, num_heads, etc.

3. tokenizer.json
   - Trained tokenizer
   - Load with: Tokenizer.from_file(tokenizer_path)

4. training_results.json
   - Training statistics
   - Contains: final_loss, best_loss, total_steps, etc.


TRAINING OUTPUT
===============

The script provides detailed progress information:

STEP 1: LOADING DATASET
  - Shows dataset ID, config, split, and text column
  - Reports number of examples loaded

STEP 2: TOKENIZER SETUP
  - Loads existing or trains new tokenizer
  - Shows vocabulary size

STEP 3: MODEL CREATION
  - Reports model architecture (layers, hidden size, heads, FFN size)
  - Shows actual parameter count vs target

STEP 4: PREPARING TRAINING DATA
  - Shows number of sequences and batches created
  - Reports total tokens

STEP 5: TRAINING SETUP
  - Shows optimizer configuration
  - Reports total training steps

STEP 6: TRAINING
  - Real-time progress updates every N steps
  - Shows: step number, loss, tokens/sec, elapsed time
  - Epoch summaries with average loss and throughput
  - Automatically saves best checkpoint

TRAINING COMPLETE
  - Final statistics: total time, steps, tokens processed
  - Tokens per second throughput
  - Final and best loss values
  - Output file locations


TIPS AND BEST PRACTICES
========================

1. Start Small
   - Begin with small models (500K-1M params) and limited data (--max-examples 1000)
   - Verify everything works before scaling up

2. Model Size Selection
   - 500K-1M params: Good for testing and small tasks
   - 3M-10M params: Small models for specific domains
   - 50M-100M params: Medium-sized models with good performance
   - 500M+ params: Large models (requires significant compute)

3. Batch Size and Memory
   - Larger batch sizes train faster but use more memory
   - If you get OOM errors, reduce batch_size or seq_len
   - Typical ranges: 16-128

4. Learning Rate
   - Default 5e-4 works well for most cases
   - For larger models, try 1e-4 to 3e-4
   - For smaller models or fine-tuning, try 1e-3 to 3e-3

5. Sequence Length
   - Longer sequences capture more context but use more memory
   - Common values: 64, 128, 256, 512, 1024
   - Start with 128 and adjust based on your data

6. Tokenizer
   - Vocab size 10K-20K works well for most text
   - Larger vocab (30K-50K) for multilingual or diverse text
   - The tokenizer is saved and reused across runs

7. Monitoring Training
   - Watch for decreasing loss (good)
   - If loss stops decreasing, training may be done
   - If loss increases, learning rate may be too high


EXAMPLE WORKFLOWS
=================

Quick Test (2-3 minutes):
```bash
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 500000 \
    --max-examples 500 \
    --epochs 2 \
    --batch-size 16
```

Small Model Training (30 minutes - 1 hour):
```bash
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 3000000 \
    --epochs 5 \
    --batch-size 32
```

Medium Model Training (several hours):
```bash
python train_model.py \
    --dataset-id skeskinen/TinyStories-Instruct-hf \
    --target-params 10000000 \
    --epochs 10 \
    --batch-size 64 \
    --seq-len 256 \
    --learning-rate 3e-4
```

Large Model Training (many hours/days):
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


TROUBLESHOOTING
===============

1. "Out of Memory" Error
   - Reduce --batch-size (e.g., 32 → 16 → 8)
   - Reduce --seq-len (e.g., 256 → 128 → 64)
   - Reduce --target-params (smaller model)

2. Slow Training
   - Increase --batch-size (if memory allows)
   - Reduce --max-examples for testing
   - Use --log-every 50 or 100 for less frequent logging

3. Loss Not Decreasing
   - Increase --epochs (train longer)
   - Adjust --learning-rate (try 1e-3 or 1e-4)
   - Check if dataset is appropriate for language modeling

4. Import Errors
   - Make sure you're in the right directory
   - Install requirements: pip install -r requirements.txt
   - Check Python version: python --version (3.8+)


For more information, see the source code: train_model.py
"""

if __name__ == "__main__":
    print(__doc__)
