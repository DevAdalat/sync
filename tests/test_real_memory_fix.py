"""
Real test of the memory fix with actual dataset loading.
This will test the exact scenario the user experienced.
"""

import os
import time


def get_memory_gb():
    """Get current memory usage in GB"""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024
    except ImportError:
        return 0


def test_real_memory_fix():
    """Test memory fix with real dataset"""

    print("=" * 70)
    print("REAL MEMORY FIX TEST")
    print("=" * 70)

    from src.data.hf_dataset_loader import HFDatasetLoader

    # Test with realistic parameters
    seq_len = 128
    stride = 64
    max_examples = 2000  # Reasonable test size

    print("Test parameters:")
    print(f"  - seq_len: {seq_len}")
    print(f"  - stride: {stride}")
    print(f"  - max_examples: {max_examples}")

    # Load dataset
    print("\n1. Loading dataset...")
    mem_before = get_memory_gb()

    loader = HFDatasetLoader(
        dataset_id="iohadrubin/wikitext-103-raw-v1",
        text_column="text",
        split="train",
        streaming=True,
    )

    # Train tokenizer
    print("2. Training tokenizer...")
    tokenizer = loader.train_tokenizer(vocab_size=2000, max_examples=500)

    # Test memory-efficient version (NEW DEFAULT)
    print("\n3. Testing MEMORY-EFFICIENT version (new default)...")
    mem_before_efficient = get_memory_gb()

    start_time = time.time()
    inputs_eff, targets_eff = loader.prepare_sequences(
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        max_examples=max_examples,
        memory_efficient=True,  # New memory-efficient version
        use_gpu=False,
    )
    time_efficient = time.time() - start_time
    mem_after_efficient = get_memory_gb()

    print(f"   ✓ Time: {time_efficient:.2f}s")
    print(f"   ✓ Memory used: {mem_after_efficient - mem_before_efficient:.2f} GB")
    print(f"   ✓ Sequences created: {len(inputs_eff)}")
    print(f"   ✓ Input shape: {inputs_eff.shape}")
    print(f"   ✓ Data type: {type(inputs_eff)}")

    # Test original version (OLD PROBLEMATIC)
    print("\n4. Testing ORIGINAL version (problematic)...")
    mem_before_original = get_memory_gb()

    start_time = time.time()
    inputs_orig, targets_orig = loader.prepare_sequences(
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        max_examples=max_examples,
        memory_efficient=False,  # Original memory-wasting version
        use_gpu=False,
    )
    time_original = time.time() - start_time
    mem_after_original = get_memory_gb()

    print(f"   ✓ Time: {time_original:.2f}s")
    print(f"   ✓ Memory used: {mem_after_original - mem_before_original:.2f} GB")
    print(f"   ✓ Sequences created: {len(inputs_orig)}")
    print(f"   ✓ Input type: {type(inputs_orig)}")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    mem_efficient_used = mem_after_efficient - mem_before_efficient
    mem_original_used = mem_after_original - mem_before_original

    print("Memory-efficient version:")
    print(f"  - Memory: {mem_efficient_used:.2f} GB")
    print(f"  - Time: {time_efficient:.2f}s")
    print(f"  - Data type: {type(inputs_eff)}")

    print("\nOriginal version:")
    print(f"  - Memory: {mem_original_used:.2f} GB")
    print(f"  - Time: {time_original:.2f}s")
    print(f"  - Data type: {type(inputs_orig)}")

    if mem_original_used > 0:
        reduction = mem_original_used / mem_efficient_used
        print(f"\nMemory reduction: {reduction:.1f}x")
        print(f"Memory saved: {mem_original_used - mem_efficient_used:.2f} GB")

    # Verify data integrity
    print("\nData integrity check:")
    if hasattr(inputs_eff, "shape") and hasattr(inputs_orig, "__len__"):
        print(f"  - Same number of sequences: {len(inputs_eff) == len(inputs_orig)}")
        if len(inputs_eff) > 0 and len(inputs_orig) > 0:
            # Convert to same type for comparison
            import numpy as np

            if hasattr(inputs_eff, "numpy"):
                eff_first = inputs_eff[0].numpy()
            else:
                eff_first = inputs_eff[0]

            orig_first = np.array(inputs_orig[0])
            print(
                f"  - First sequence matches: {np.array_equal(eff_first, orig_first)}"
            )

    # Clean up
    del inputs_eff, targets_eff
    import gc

    gc.collect()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if mem_original_used > mem_efficient_used * 1.5:
        print("✅ MEMORY FIX WORKING! Significant memory reduction achieved.")
    else:
        print("⚠️  Memory reduction may be smaller than expected.")

    if time_efficient <= time_original * 1.2:
        print("✅ Performance maintained! No significant speed loss.")
    else:
        print("⚠️  Performance may be impacted.")

    return {
        "memory_efficient": mem_efficient_used,
        "memory_original": mem_original_used,
        "time_efficient": time_efficient,
        "time_original": time_original,
        "reduction": mem_original_used / mem_efficient_used
        if mem_efficient_used > 0
        else 1,
    }


if __name__ == "__main__":
    try:
        results = test_real_memory_fix()
        print(f"\n🎯 Final memory reduction: {results['reduction']:.1f}x")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
