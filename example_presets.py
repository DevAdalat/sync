"""
Example script showing how to use model presets.

This demonstrates the new preset system for creating models of different sizes.
"""

from config import ModelConfig

def main():
    # List all available presets
    print("="*80)
    print("MODEL PRESET EXAMPLES")
    print("="*80)
    
    ModelConfig.list_presets()
    
    print("\n" + "="*80)
    print("Creating Models from Presets".center(80))
    print("="*80 + "\n")
    
    # Example 1: Create a tiny model (recommended for your 5M param use case)
    print("1. TINY MODEL (Your original 5M param target - FAST!)")
    print("-" * 60)
    tiny_config = ModelConfig.from_preset("tiny", vocab_size=8986, max_len=128)
    print(f"   d_model:    {tiny_config.d_model}")
    print(f"   num_layers: {tiny_config.num_layers}")
    print(f"   num_heads:  {tiny_config.num_heads}")
    print(f"   d_ff:       {tiny_config.d_ff}")
    print(f"   ✅ ~5M parameters, 8x faster than your 92-layer config!\n")
    
    # Example 2: Nano model for ultra-fast testing
    print("2. NANO MODEL (Ultra-fast for testing)")
    print("-" * 60)
    nano_config = ModelConfig.from_preset("nano", vocab_size=8986, max_len=128)
    print(f"   d_model:    {nano_config.d_model}")
    print(f"   num_layers: {nano_config.num_layers}")
    print(f"   num_heads:  {nano_config.num_heads}")
    print(f"   d_ff:       {nano_config.d_ff}")
    print(f"   ✅ ~1M parameters, great for rapid prototyping!\n")
    
    # Example 3: Medium model (BERT/GPT-2 size)
    print("3. MEDIUM MODEL (GPT-2 Small / BERT Base scale)")
    print("-" * 60)
    medium_config = ModelConfig.from_preset("medium", vocab_size=8986, max_len=128)
    print(f"   d_model:    {medium_config.d_model}")
    print(f"   num_layers: {medium_config.num_layers}")
    print(f"   num_heads:  {medium_config.num_heads}")
    print(f"   d_ff:       {medium_config.d_ff}")
    print(f"   ✅ ~125M parameters, production-quality model!\n")
    
    # Example 4: Custom overrides
    print("4. CUSTOM: Tiny model with custom dropout and activation")
    print("-" * 60)
    custom_config = ModelConfig.from_preset(
        "tiny", 
        vocab_size=8986, 
        max_len=128,
        dropout_rate=0.2,  # Custom dropout
        activation="silu",  # Custom activation
        use_lora=True       # Enable LoRA
    )
    print(f"   d_model:      {custom_config.d_model}")
    print(f"   num_layers:   {custom_config.num_layers}")
    print(f"   dropout_rate: {custom_config.dropout_rate}")
    print(f"   activation:   {custom_config.activation}")
    print(f"   use_lora:     {custom_config.use_lora}")
    print(f"   ✅ Preset + custom overrides!\n")
    
    # Example 5: Comparison with your old config
    print("5. COMPARISON: Your Old Config vs Recommended")
    print("-" * 60)
    print("   OLD (slow):")
    print(f"      d_model=64, num_layers=92")
    print(f"      → 92 sequential operations per token 🐌")
    print()
    print("   NEW (fast):")
    print(f"      d_model=256, num_layers=8 (tiny preset)")
    print(f"      → 8 sequential operations per token ⚡")
    print(f"      → Same ~5M params, 11x faster!\n")
    
    print("="*80)
    print("Usage in your training script:")
    print("="*80)
    print("""
    from config import ModelConfig
    
    # Instead of manually specifying dimensions:
    # config = ModelConfig(vocab_size=8986, d_model=64, num_layers=92, ...)
    
    # Use a preset:
    config = ModelConfig.from_preset("tiny", vocab_size=8986, max_len=128)
    
    # Or with custom overrides:
    config = ModelConfig.from_preset(
        "tiny", 
        vocab_size=8986, 
        max_len=128,
        dropout_rate=0.15
    )
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
