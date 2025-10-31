"""
Test the memory-efficient fix for prepare_sequences.
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


def test_memory_fix():
    """Test the memory-efficient version"""

    print("=" * 60)
    print("TESTING MEMORY FIX")
    print("=" * 60)

    from src.data.hf_dataset_loader import HFDatasetLoader

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

    # Test memory-efficient version
    print("\nTesting memory-efficient version...")
    mem_before = get_memory_mb()

    inputs, targets = loader.prepare_sequences(
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        max_examples=max_examples,
        memory_efficient=True,  # Use the new memory-efficient version
        use_gpu=False,
    )

    mem_after = get_memory_mb()

    print(f"Memory used: {mem_after - mem_before:.1f} MB")
    print(f"Sequences created: {len(inputs)}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Data type: {type(inputs)}")

    # Calculate expected memory
    expected_memory_mb = len(inputs) * seq_len * 2 * 4 / 1024 / 1024
    print(f"Expected memory: {expected_memory_mb:.1f} MB")
    print(f"Memory efficiency: {(mem_after - mem_before) / expected_memory_mb:.2f}x")

    # Test if data is correct
    print("\nData validation:")
    print(f"First input: {inputs[0][:10]}")
    print(f"First target: {targets[0][:10]}")
    print(
        f"Input equals target shifted by 1: {jnp.allclose(inputs[0][1:], targets[0][:-1])}"
    )

    return {
        "memory_used": mem_after - mem_before,
        "expected_memory": expected_memory_mb,
        "efficiency": (mem_after - mem_before) / expected_memory_mb,
        "sequences": len(inputs),
    }


if __name__ == "__main__":
    import jax.numpy as jnp

    test_memory_fix()
