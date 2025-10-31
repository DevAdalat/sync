"""
Simple memory test to identify the 34x memory blowup issue.
"""

import os


def get_memory_mb():
    """Get current memory usage in MB"""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0


def test_memory_issue():
    """Test memory usage step by step"""

    print("=" * 60)
    print("MEMORY ISSUE ANALYSIS")
    print("=" * 60)

    # Test with a small dataset first
    print("\n1. Testing with small dataset...")

    from hf_dataset_loader import HFDatasetLoader

    loader = HFDatasetLoader(
        dataset_id="iohadrubin/wikitext-103-raw-v1",
        text_column="text",
        split="train",
        streaming=True,
    )

    # Train small tokenizer
    tokenizer = loader.train_tokenizer(vocab_size=1000, max_examples=100)

    # Test parameters
    seq_len = 64
    stride = 32
    max_examples = 500

    print(
        f"Parameters: seq_len={seq_len}, stride={stride}, max_examples={max_examples}"
    )

    # Step 1: Get texts
    print("\n2. Loading texts...")
    mem_before = get_memory_mb()
    texts = loader.get_text_data(max_examples=max_examples)
    mem_after_texts = get_memory_mb()
    print(f"   Texts loaded: {len(texts)}")
    print(
        f"   Memory: {mem_after_texts:.1f} MB (+{mem_after_texts - mem_before:.1f} MB)"
    )

    # Step 2: Tokenize
    print("\n3. Tokenizing...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text).ids
        all_tokens.extend(tokens)
    mem_after_tokens = get_memory_mb()
    print(f"   Tokens: {len(all_tokens)}")
    print(
        f"   Memory: {mem_after_tokens:.1f} MB (+{mem_after_tokens - mem_after_texts:.1f} MB)"
    )

    # Step 3: Calculate expected sequences
    num_sequences = (len(all_tokens) - seq_len) // stride + 1
    expected_memory_mb = (
        num_sequences * seq_len * 2 * 4 / 1024 / 1024
    )  # 2 arrays, 4 bytes per int32
    print("\n4. Sequence analysis:")
    print(f"   Expected sequences: {num_sequences}")
    print(f"   Expected memory for sequences: {expected_memory_mb:.1f} MB")

    # Step 4: Create sequences (problematic part)
    print("\n5. Creating sequences (this is where memory blows up)...")
    inputs = []
    targets = []

    for i in range(0, len(all_tokens) - seq_len, stride):
        input_seq = all_tokens[i : i + seq_len]
        target_seq = all_tokens[i + 1 : i + seq_len + 1]
        inputs.append(input_seq)
        targets.append(target_seq)

        # Show memory at intervals
        if i % 1000 == 0 and i > 0:
            current_mem = get_memory_mb()
            print(
                f"   Progress: {i // stride}/{num_sequences} sequences, Memory: {current_mem:.1f} MB"
            )

    mem_after_sequences = get_memory_mb()
    print(
        f"   Final memory: {mem_after_sequences:.1f} MB (+{mem_after_sequences - mem_after_tokens:.1f} MB)"
    )

    # Analysis
    print("\n" + "=" * 60)
    print("MEMORY ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Texts memory:          {mem_after_texts - mem_before:.1f} MB")
    print(f"Tokens memory:         {mem_after_tokens - mem_after_texts:.1f} MB")
    print(f"Sequences memory:      {mem_after_sequences - mem_after_tokens:.1f} MB")
    print(f"Expected sequences:     {expected_memory_mb:.1f} MB")
    print(
        f"Actual vs expected:     {(mem_after_sequences - mem_after_tokens) / expected_memory_mb:.1f}x"
    )

    # Identify the problem
    print("\nPROBLEM IDENTIFIED:")
    print("1. Overlapping sequences create duplicate data storage")
    print("2. Each sequence stores a full copy of tokens")
    print(f"3. With stride=32, each token appears in ~{seq_len // stride} sequences")
    print(f"4. Memory blowup factor: ~{seq_len // stride}x from overlapping")

    return {
        "texts_memory": mem_after_texts - mem_before,
        "tokens_memory": mem_after_tokens - mem_after_texts,
        "sequences_memory": mem_after_sequences - mem_after_tokens,
        "expected_memory": expected_memory_mb,
        "blowup_factor": (mem_after_sequences - mem_after_tokens) / expected_memory_mb,
    }


if __name__ == "__main__":
    test_memory_issue()
