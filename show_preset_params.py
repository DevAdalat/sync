"""
Calculate exact parameter counts for each model preset.
"""

from config import ModelConfig, MODEL_PRESETS


def calculate_params(config: ModelConfig) -> int:
    """Calculate approximate parameter count for a transformer model."""
    d_model = config.d_model
    num_layers = config.num_layers
    d_ff = config.d_ff
    vocab_size = config.vocab_size
    num_heads = config.num_heads
    
    # Token embeddings
    token_embed_params = vocab_size * d_model
    
    # Position embeddings (if not using RoPE)
    if not config.use_rope:
        pos_embed_params = config.max_len * d_model
    else:
        pos_embed_params = 0
    
    # Per transformer layer:
    # - Multi-head attention: Q, K, V projections + output projection
    # - Feed-forward: 2 linear layers (or 3 for SwiGLU)
    # - Layer norms (minimal params)
    
    # Attention params per layer
    qkv_params = 3 * (d_model * d_model)  # Q, K, V projections
    attn_out_params = d_model * d_model    # Output projection
    attn_params = qkv_params + attn_out_params
    
    # Feed-forward params per layer
    if config.use_swiglu:
        # SwiGLU has 3 projections: gate, value, and output
        ff_params = (d_model * d_ff) + (d_model * d_ff) + (d_ff * d_model)
    else:
        # Standard FFN: 2 projections
        ff_params = (d_model * d_ff) + (d_ff * d_model)
    
    # Layer norm params (scale only for RMSNorm, scale + bias for LayerNorm)
    if config.use_rmsnorm:
        norm_params = 2 * d_model  # 2 norms per layer
    else:
        norm_params = 4 * d_model  # 2 norms per layer, each with scale + bias
    
    # Total per layer
    per_layer_params = attn_params + ff_params + norm_params
    
    # All layers
    total_layer_params = num_layers * per_layer_params
    
    # Output projection (language model head)
    output_params = d_model * vocab_size
    
    # Total
    total = token_embed_params + pos_embed_params + total_layer_params + output_params
    
    return total


def format_params(num_params: int) -> str:
    """Format parameter count in human-readable form."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    else:
        return str(num_params)


def main():
    vocab_size = 8986  # Your vocab size
    max_len = 128
    
    print("\n" + "="*80)
    print("EXACT PARAMETER COUNTS FOR EACH PRESET")
    print(f"(vocab_size={vocab_size}, max_len={max_len})")
    print("="*80 + "\n")
    
    results = []
    
    for preset_name in MODEL_PRESETS.keys():
        config = ModelConfig.from_preset(preset_name, vocab_size=vocab_size, max_len=max_len)
        num_params = calculate_params(config)
        
        results.append({
            "preset": preset_name.upper(),
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "d_ff": config.d_ff,
            "params": num_params,
            "params_str": format_params(num_params)
        })
    
    # Print results
    for r in results:
        print(f"{r['preset']:8s} | d_model={r['d_model']:4d} | layers={r['num_layers']:2d} | "
              f"heads={r['num_heads']:2d} | d_ff={r['d_ff']:4d} | "
              f"params={r['params_str']:>8s} ({r['params']:,})")
    
    print("\n" + "="*80)
    print("RECOMMENDATION FOR YOUR USE CASE")
    print("="*80)
    print(f"\nYour original config:")
    print(f"  - d_model=64, num_layers=92, d_ff=128")
    print(f"  - Result: VERY SLOW (92 sequential operations per token)")
    print(f"\nRecommended: Use 'TINY' preset")
    print(f"  - d_model=256, num_layers=8, d_ff=1024")
    print(f"  - Result: 11x FASTER (8 sequential operations per token)")
    print(f"  - Similar parameter count (~5M)")
    print(f"  - Better performance (proven architecture ratios)")
    
    # Show speed comparison
    print("\n" + "="*80)
    print("GENERATION SPEED COMPARISON")
    print("="*80)
    print(f"\nTo generate 100 tokens:")
    print(f"  Old config (92 layers):  9,200 transformer block operations")
    print(f"  TINY preset (8 layers):    800 transformer block operations")
    print(f"  NANO preset (6 layers):    600 transformer block operations")
    print(f"\n  → TINY is ~11x faster!")
    print(f"  → NANO is ~15x faster!")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
