"""
Memory analysis script to identify why prepare_sequences uses so much RAM.

This script analyzes the memory usage at each step of the process.
"""

import os

import psutil

from hf_dataset_loader import HFDatasetLoader


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024


def analyze_memory_usage():
    """Analyze memory usage step by step"""

    print("=" * 70)
    print("MEMORY ANALYSIS: prepare_sequences")
    print("=" * 70)

    # Initial memory
    initial_mem = get_memory_usage()
    print(f"Initial memory: {initial_mem:.2f} GB")

    # Load dataset
    print("\n1. Loading dataset...")
    loader = HFDatasetLoader(
        dataset_id="iohadrubin/wikitext-103-raw-v1",
        text_column="text",
        split="train",
        streaming=True,  # Important for memory efficiency
    )
    after_dataset_mem = get_memory_usage()
    print(
        f"   Memory after dataset load: {after_dataset_mem:.2f} GB (+{after_dataset_mem - initial_mem:.2f} GB)"
    )

    # Train tokenizer
    print("\n2. Training tokenizer...")
    tokenizer = loader.train_tokenizer(vocab_size=5000, max_examples=1000)
    after_tokenizer_mem = get_memory_usage()
    print(
        f"   Memory after tokenizer: {after_tokenizer_mem:.2f} GB (+{after_tokenizer_mem - after_dataset_mem:.2f} GB)"
    )

    # Test parameters
    seq_len = 128
    stride = 64
    max_examples = 10000

    print("\n3. Testing with parameters:")
    print(f"   - seq_len: {seq_len}")
    print(f"   - stride: {stride}")
    print(f"   - max_examples: {max_examples}")

    # Step 1: Get text data
    print("\n4. Loading text data...")
    texts = loader.get_text_data(max_examples=max_examples)
    after_texts_mem = get_memory_usage()
    print(
        f"   Memory after loading texts: {after_texts_mem:.2f} GB (+{after_texts_mem - after_tokenizer_mem:.2f} GB)"
    )
    print(f"   Number of texts: {len(texts)}")
    print(
        f"   Average text length: {sum(len(t) for t in texts) / len(texts):.0f} chars"
    )

    # Step 2: Tokenization (simulate the problematic part)
    print("\n5. Tokenization (current method)...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text).ids
        all_tokens.extend(tokens)
    after_tokens_mem = get_memory_usage()
    print(
        f"   Memory after tokenization: {after_tokens_mem:.2f} GB (+{after_tokens_mem - after_texts_mem:.2f} GB)"
    )
    print(f"   Total tokens: {len(all_tokens)}")
    print(
        f"   Memory per token: {(after_tokens_mem - after_texts_mem) * 1024 / len(all_tokens):.2f} MB"
    )

    # Step 3: Calculate expected sequence count
    print("\n6. Sequence creation analysis...")
    num_sequences = (len(all_tokens) - seq_len) // stride + 1
    print(f"   Expected sequences: {num_sequences}")
    print(
        f"   Memory needed for sequences: {num_sequences * seq_len * 2 * 4 / 1024 / 1024:.2f} MB"
    )
    print("   (inputs + targets, each token = 4 bytes)")

    # Step 4: Create sequences (current method)
    print("\n7. Creating sequences (current method)...")
    inputs = []
    targets = []
    for i in range(0, len(all_tokens) - seq_len, stride):
        input_seq = all_tokens[i : i + seq_len]
        target_seq = all_tokens[i + 1 : i + seq_len + 1]
        inputs.append(input_seq)
        targets.append(target_seq)
    after_sequences_mem = get_memory_usage()
    print(
        f"   Memory after sequences: {after_sequences_mem:.2f} GB (+{after_sequences_mem - after_tokens_mem:.2f} GB)"
    )

    # Step 5: Convert to JAX arrays
    print("\n8. Converting to JAX arrays...")
    import jax.numpy as jnp

    inputs_jax = jnp.array(inputs, dtype=jnp.int32)
    targets_jax = jnp.array(targets, dtype=jnp.int32)
    after_jax_mem = get_memory_usage()
    print(
        f"   Memory after JAX arrays: {after_jax_mem:.2f} GB (+{after_jax_mem - after_sequences_mem:.2f} GB)"
    )

    # Summary
    print("\n" + "=" * 70)
    print("MEMORY USAGE SUMMARY")
    print("=" * 70)
    print(f"Initial:                {initial_mem:.2f} GB")
    print(f"After dataset:          {after_dataset_mem:.2f} GB")
    print(f"After tokenizer:        {after_tokenizer_mem:.2f} GB")
    print(f"After texts:            {after_texts_mem:.2f} GB")
    print(f"After tokenization:     {after_tokens_mem:.2f} GB")
    print(f"After sequences:        {after_sequences_mem:.2f} GB")
    print(f"After JAX arrays:       {after_jax_mem:.2f} GB")
    print(f"Total increase:         {after_jax_mem - initial_mem:.2f} GB")
    print("Dataset size:           1.33 GB")
    print(f"Memory blowup factor:   {(after_jax_mem - initial_mem) / 1.33:.1f}x")

    # Identify problems
    print("\n" + "=" * 70)
    print("MEMORY ISSUES IDENTIFIED")
    print("=" * 70)
    print("1. All texts loaded into memory at once")
    print("2. All tokens stored in single list (duplicates memory)")
    print("3. Overlapping sequences create multiple copies of same data")
    print("4. Lists + JAX arrays = duplicate storage")
    print("5. No cleanup of intermediate variables")

    return {
        "initial": initial_mem,
        "final": after_jax_mem,
        "blowup_factor": (after_jax_mem - initial_mem) / 1.33,
    }


if __name__ == "__main__":
    analyze_memory_usage()
