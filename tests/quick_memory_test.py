"""
Quick test to verify memory fix is working.
"""

def quick_memory_test():
    """Quick test of memory fix"""
    
    print("🧪 Quick Memory Fix Test")
    print("="*50)
    
    from hf_dataset_loader import HFDatasetLoader
    
    # Small test to verify fix works
    print("1. Loading dataset...")
    loader = HFDatasetLoader(
        dataset_id="iohadrubin/wikitext-103-raw-v1",
        text_column="text",
        split="train",
        streaming=True
    )
    
    print("2. Training tiny tokenizer...")
    tokenizer = loader.train_tokenizer(vocab_size=500, max_examples=100)
    
    print("3. Testing memory-efficient version...")
    try:
        inputs, targets = loader.prepare_sequences(
            tokenizer=tokenizer,
            seq_len=32,
            stride=16,
            max_examples=200,
            memory_efficient=True,
            use_gpu=False
        )
        print(f"   ✅ Memory-efficient version WORKS!")
        print(f"   ✅ Created {len(inputs)} sequences")
        print(f"   ✅ Input shape: {inputs.shape}")
        print(f"   ✅ Data type: {type(inputs)}")
        
        # Test data integrity
        print(f"   ✅ First input: {inputs[0][:5]}")
        print(f"   ✅ First target: {targets[0][:5]}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Memory-efficient version FAILED: {e}")
        return False

if __name__ == "__main__":
    success = quick_memory_test()
    if success:
        print("\n🎉 Memory fix is WORKING!")
    else:
        print("\n💥 Memory fix has ISSUES!")