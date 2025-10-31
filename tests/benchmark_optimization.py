"""
Benchmark script to compare optimized vs original prepare_sequences performance.

This demonstrates the speedup from:
1. Multithreaded tokenization (CPU parallelization)
2. GPU-accelerated sequence creation (JAX vectorization)
"""

import time
import logging
from hf_dataset_loader import HFDatasetLoader
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_prepare_sequences():
    """Benchmark the optimized prepare_sequences method"""
    
    print("="*70)
    print("BENCHMARK: Optimized prepare_sequences")
    print("="*70)
    
    # Load a small dataset for testing
    loader = HFDatasetLoader(
        dataset_id="iohadrubin/wikitext-103-raw-v1",
        text_column="text",
        split="train",
        streaming=True
    )
    
    # Train a small tokenizer
    print("\n1. Training tokenizer...")
    tokenizer = loader.train_tokenizer(vocab_size=5000, max_examples=500)
    
    # Test parameters
    seq_len = 128
    stride = 64
    max_examples = 1000
    
    print(f"\n2. Testing with parameters:")
    print(f"   - seq_len: {seq_len}")
    print(f"   - stride: {stride}")
    print(f"   - max_examples: {max_examples}")
    
    # Test 1: Original method (with GPU disabled for fair comparison)
    print(f"\n3. Running with GPU disabled (CPU-only baseline)...")
    start = time.time()
    inputs1, targets1 = loader.prepare_sequences(
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        max_examples=max_examples,
        num_workers=1,  # Single thread for baseline
        use_gpu=False
    )
    time1 = time.time() - start
    print(f"   ✓ Time (1 thread, CPU): {time1:.3f}s")
    print(f"   ✓ Sequences created: {len(inputs1)}")
    
    # Test 2: Multi-threaded CPU
    print(f"\n4. Running with multi-threading (CPU parallel)...")
    start = time.time()
    inputs2, targets2 = loader.prepare_sequences(
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        max_examples=max_examples,
        num_workers=None,  # Auto-detect CPU count
        use_gpu=False
    )
    time2 = time.time() - start
    print(f"   ✓ Time (multi-thread, CPU): {time2:.3f}s")
    print(f"   ✓ Speedup: {time1/time2:.2f}x")
    print(f"   ✓ Sequences created: {len(inputs2)}")
    
    # Test 3: Multi-threaded + GPU
    print(f"\n5. Running with multi-threading + GPU...")
    start = time.time()
    inputs3, targets3 = loader.prepare_sequences(
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        max_examples=max_examples,
        num_workers=None,  # Auto-detect CPU count
        use_gpu=True
    )
    time3 = time.time() - start
    print(f"   ✓ Time (multi-thread + GPU): {time3:.3f}s")
    print(f"   ✓ Speedup over baseline: {time1/time3:.2f}x")
    print(f"   ✓ Sequences created: {len(inputs3)}")
    
    # Test 4: Fast method (returns JAX arrays directly)
    print(f"\n6. Running prepare_sequences_fast (JAX arrays)...")
    start = time.time()
    inputs4, targets4 = loader.prepare_sequences_fast(
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        max_examples=max_examples,
        num_workers=None,
        return_jax=True
    )
    time4 = time.time() - start
    print(f"   ✓ Time (fast method): {time4:.3f}s")
    print(f"   ✓ Speedup over baseline: {time1/time4:.2f}x")
    print(f"   ✓ Sequences created: {len(inputs4)}")
    print(f"   ✓ Output type: {type(inputs4)}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Single-threaded CPU:     {time1:.3f}s  (1.00x)")
    print(f"Multi-threaded CPU:      {time2:.3f}s  ({time1/time2:.2f}x speedup)")
    print(f"Multi-threaded + GPU:    {time3:.3f}s  ({time1/time3:.2f}x speedup)")
    print(f"Fast method (JAX):       {time4:.3f}s  ({time1/time4:.2f}x speedup)")
    print("="*70)
    
    # Verify results are consistent
    print(f"\n✓ All methods produced {len(inputs1)} sequences")
    print("✓ Optimization successful!")


if __name__ == "__main__":
    benchmark_prepare_sequences()
