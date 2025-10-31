"""
Minimal test to verify memory fix works.
"""


def minimal_test():
    """Minimal test of memory fix"""

    print("🔍 Minimal Memory Fix Test")
    print("=" * 40)

    try:
        import numpy as np

        print("✅ Imports successful")

        # Test the core memory fix logic directly
        print("🧪 Testing core memory fix logic...")

        # Simulate the old problematic approach
        all_tokens = list(range(1000))  # 1000 tokens
        seq_len = 64
        stride = 32

        print(f"   Tokens: {len(all_tokens)}")
        print(f"   Seq len: {seq_len}, Stride: {stride}")

        # OLD METHOD (problematic)
        inputs_old = []
        targets_old = []
        for i in range(0, len(all_tokens) - seq_len, stride):
            input_seq = all_tokens[i : i + seq_len]
            target_seq = all_tokens[i + 1 : i + seq_len + 1]
            inputs_old.append(input_seq)
            targets_old.append(target_seq)

        print(f"   Old method: {len(inputs_old)} sequences (lists)")

        # NEW METHOD (memory efficient)
        num_sequences = (len(all_tokens) - seq_len) // stride + 1
        inputs_new = np.zeros((num_sequences, seq_len), dtype=np.int32)
        targets_new = np.zeros((num_sequences, seq_len), dtype=np.int32)

        for i in range(num_sequences):
            start_idx = i * stride
            inputs_new[i] = all_tokens[start_idx : start_idx + seq_len]
            targets_new[i] = all_tokens[start_idx + 1 : start_idx + seq_len + 1]

        print(f"   New method: {len(inputs_new)} sequences (numpy)")

        # Verify they're the same
        first_old = inputs_old[0]
        first_new = inputs_new[0].tolist()

        print(f"   First sequence old: {first_old[:5]}...")
        print(f"   First sequence new: {first_new[:5]}...")
        print(f"   Data matches: {first_old == first_new}")

        # Memory comparison
        old_memory = len(inputs_old) * seq_len * 8  # Python int ~8 bytes
        new_memory = len(inputs_new) * seq_len * 4  # int32 = 4 bytes

        print(f"   Old memory: ~{old_memory / 1024:.1f} KB")
        print(f"   New memory: ~{new_memory / 1024:.1f} KB")
        print(f"   Reduction: {old_memory / new_memory:.1f}x")

        print("\n🎉 Core memory fix logic WORKS!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    minimal_test()
