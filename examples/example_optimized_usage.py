"""
Example: Using the optimized prepare_sequences in your training code

This shows how to integrate the optimized methods into your existing workflow.
"""

from hf_dataset_loader import HFDatasetLoader
from tokenizers import Tokenizer
import jax
import jax.numpy as jnp

# Initialize dataset loader
loader = HFDatasetLoader(
    dataset_id="iohadrubin/wikitext-103-raw-v1",
    text_column="text",
    split="train",
    streaming=True
)

# Train or load tokenizer
tokenizer = loader.train_tokenizer(vocab_size=5000, max_examples=1000)
# Or load existing: tokenizer = Tokenizer.from_file("char_tokenizer.json")

print("\n" + "="*70)
print("METHOD 1: Optimized prepare_sequences (backward compatible)")
print("="*70)

# This works EXACTLY like before, but is now much faster!
inputs, targets = loader.prepare_sequences(
    tokenizer=tokenizer,
    seq_len=128,
    stride=64,
    max_examples=1000
)

print(f"✓ Created {len(inputs)} sequences")
print(f"✓ Type: {type(inputs)} (list of lists)")
print(f"✓ First input: {inputs[0][:10]}...")

# Convert to JAX arrays for training (if needed)
inputs_jax = jnp.array(inputs, dtype=jnp.int32)
targets_jax = jnp.array(targets, dtype=jnp.int32)
print(f"✓ JAX shape: {inputs_jax.shape}")


print("\n" + "="*70)
print("METHOD 2: prepare_sequences_fast (returns JAX arrays directly)")
print("="*70)

# This is even faster - returns JAX arrays directly!
inputs_fast, targets_fast = loader.prepare_sequences_fast(
    tokenizer=tokenizer,
    seq_len=128,
    stride=64,
    max_examples=1000,
    return_jax=True  # Returns JAX arrays directly
)

print(f"✓ Created {len(inputs_fast)} sequences")
print(f"✓ Type: {type(inputs_fast)} (JAX array)")
print(f"✓ Shape: {inputs_fast.shape}")
print(f"✓ First input: {inputs_fast[0][:10]}")


print("\n" + "="*70)
print("METHOD 3: Batch iterator (recommended for large datasets)")
print("="*70)

# For large datasets, use batch iterator to avoid loading everything at once
batch_iterator = loader.create_batch_iterator(
    tokenizer=tokenizer,
    batch_size=32,
    seq_len=128,
    stride=64,
    max_examples=1000,
    shuffle=True,
    num_workers=None  # Auto-detect CPU count
)

# Iterate over batches
for i, batch in enumerate(batch_iterator):
    if i == 0:  # Just show first batch
        print(f"✓ Batch {i}:")
        print(f"  - input_ids shape: {batch['input_ids'].shape}")
        print(f"  - labels shape: {batch['labels'].shape}")
        print(f"  - Type: {type(batch['input_ids'])}")
        break


print("\n" + "="*70)
print("PERFORMANCE TIPS")
print("="*70)
print("""
1. For small datasets (<10K tokens):
   - Use prepare_sequences() with use_gpu=False
   - Single threading is often sufficient

2. For medium datasets (10K-1M tokens):
   - Use prepare_sequences() with default settings
   - Multi-threading + GPU gives best performance

3. For large datasets (>1M tokens):
   - Use prepare_sequences_fast() for maximum speed
   - Or use create_batch_iterator() to avoid memory issues

4. For training loops:
   - Use create_batch_iterator() directly
   - It now uses the optimized prepare_sequences_fast() internally!
""")

print("="*70)
print("✓ All methods work! Choose based on your needs.")
print("="*70)
