"""
Model sizing utility to automatically configure model architecture 
to match a target parameter count.
"""

import math
from config import ModelConfig
from typing import Tuple, Optional


def calculate_model_params(config: ModelConfig, use_swiglu=True, use_rope=True, use_rmsnorm=True) -> int:
    """Calculate total parameters for a transformer model with enhanced architecture."""
    
    # Embedding parameters
    token_embed = config.vocab_size * config.d_model
    
    # Positional embedding (only if not using RoPE)
    pos_embed = 0 if use_rope else config.max_len * config.d_model
    
    # Attention parameters per layer
    # Q, K, V projections + output projection
    attn_params = 4 * (config.d_model * config.d_model + config.d_model)
    
    # Feed-forward parameters per layer
    if use_swiglu:
        # SwiGLU uses 3 linear layers: gate, value, and output
        # gate: d_model -> d_ff
        # value: d_model -> d_ff  
        # output: d_ff -> d_model
        ff_params = 2 * (config.d_model * config.d_ff + config.d_ff) + \
                    (config.d_ff * config.d_model + config.d_model)
    else:
        # Standard FFN: Two linear layers: d_model -> d_ff -> d_model
        ff_params = (config.d_model * config.d_ff + config.d_ff) + \
                    (config.d_ff * config.d_model + config.d_model)
    
    # Normalization parameters per layer (2 norms per block)
    if use_rmsnorm:
        # RMSNorm only has scale parameter (no bias)
        ln_params = 2 * config.d_model
    else:
        # LayerNorm has scale and bias
        ln_params = 2 * (config.d_model * 2)
    
    # Total per layer
    layer_params = attn_params + ff_params + ln_params
    
    # Final norm before output
    final_norm = config.d_model if use_rmsnorm else config.d_model * 2
    
    # Output projection
    output_params = config.d_model * config.vocab_size + config.vocab_size
    
    # Total
    total = token_embed + pos_embed + (layer_params * config.num_layers) + final_norm + output_params
    
    return total


def find_optimal_config(
    target_params: int,
    vocab_size: int,
    max_len: int = 512,
    min_d_model: int = 64,
    max_d_model: int = 2048,
    prefer_depth: bool = True
) -> ModelConfig:
    """
    Find optimal model configuration to match target parameter count.
    
    Args:
        target_params: Target number of parameters
        vocab_size: Vocabulary size
        max_len: Maximum sequence length
        min_d_model: Minimum model dimension
        max_d_model: Maximum model dimension
        prefer_depth: If True, prefer more layers over wider models
    
    Returns:
        ModelConfig with approximately target_params parameters
    """
    
    best_config = None
    best_diff = float('inf')
    
    # Try different configurations
    for d_model in range(min_d_model, max_d_model + 1, 32):
        # Ensure d_model is divisible by common head counts
        if d_model % 8 != 0:
            continue
            
        # Try different number of heads
        for num_heads in [4, 8, 12, 16]:
            if d_model % num_heads != 0:
                continue
            
            # Try different d_ff ratios (typically 2-4x d_model)
            for ff_ratio in [2, 3, 4]:
                d_ff = d_model * ff_ratio
                
                # Calculate max possible layers with this config (with SwiGLU and RoPE)
                # Start with estimate and refine
                base_params = vocab_size * d_model  # No pos_embed with RoPE
                attn_params = 4 * (d_model * d_model + d_model)
                # SwiGLU: 3 linear layers
                ff_params = 2 * (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
                # RMSNorm: only scale
                ln_params = 2 * d_model
                layer_params = attn_params + ff_params + ln_params
                output_params = d_model * vocab_size + vocab_size + d_model  # +final norm
                
                available_for_layers = target_params - base_params - output_params
                estimated_layers = max(1, int(available_for_layers / layer_params))
                
                # Try layers around the estimate
                for num_layers in range(max(1, estimated_layers - 2), estimated_layers + 3):
                    config = ModelConfig(
                        vocab_size=vocab_size,
                        d_model=d_model,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        d_ff=d_ff,
                        max_len=max_len,
                        dropout_rate=0.1,
                        activation="gelu",
                        use_lora=False,
                        lora_rank=8
                    )
                    
                    params = calculate_model_params(config, use_swiglu=True, use_rope=True, use_rmsnorm=True)
                    diff = abs(params - target_params)
                    
                    # Prefer configurations closer to target
                    # If prefer_depth, slightly favor deeper models when diff is similar
                    if prefer_depth and diff < best_diff * 1.1:
                        if best_config is None or num_layers > best_config.num_layers:
                            best_diff = diff
                            best_config = config
                    elif diff < best_diff:
                        best_diff = diff
                        best_config = config
    
    if best_config is None:
        # Fallback to a simple config
        best_config = ModelConfig(
            vocab_size=vocab_size,
            d_model=min_d_model,
            num_heads=4,
            num_layers=2,
            d_ff=min_d_model * 2,
            max_len=max_len,
            dropout_rate=0.1,
            activation="gelu",
            use_lora=False,
            lora_rank=8
        )
    
    actual_params = calculate_model_params(best_config, use_swiglu=True, use_rope=True, use_rmsnorm=True)
    print(f"\nTarget parameters: {target_params:,}")
    print(f"Actual parameters: {actual_params:,}")
    print(f"Difference: {abs(actual_params - target_params):,} ({abs(actual_params - target_params) / target_params * 100:.2f}%)")
    print(f"\nModel configuration:")
    print(f"  - d_model: {best_config.d_model}")
    print(f"  - num_layers: {best_config.num_layers}")
    print(f"  - num_heads: {best_config.num_heads}")
    print(f"  - d_ff: {best_config.d_ff}")
    print(f"  - vocab_size: {best_config.vocab_size}")
    print(f"  - max_len: {best_config.max_len}")
    
    return best_config


def create_model_from_params(
    target_params: int,
    vocab_size: int,
    max_len: int = 512,
    **kwargs
) -> ModelConfig:
    """
    Create a model configuration from target parameter count.
    
    Args:
        target_params: Target number of parameters (e.g., 1000000 for 1M)
        vocab_size: Vocabulary size
        max_len: Maximum sequence length
        **kwargs: Additional arguments to pass to find_optimal_config
    
    Returns:
        ModelConfig optimized for target parameter count
    """
    return find_optimal_config(target_params, vocab_size, max_len, **kwargs)


if __name__ == "__main__":
    # Test with different sizes
    print("=" * 60)
    print("Testing 1M parameter model:")
    print("=" * 60)
    config_1m = create_model_from_params(1_000_000, vocab_size=256, max_len=100)
    
    print("\n" + "=" * 60)
    print("Testing 5M parameter model:")
    print("=" * 60)
    config_5m = create_model_from_params(5_000_000, vocab_size=256, max_len=100)
    
    print("\n" + "=" * 60)
    print("Testing 10M parameter model:")
    print("=" * 60)
    config_10m = create_model_from_params(10_000_000, vocab_size=256, max_len=100)
