"""
Example: Create a model with a specific parameter count

This script demonstrates how to create transformer models with exact parameter counts.
"""

from src.models.model import ProductionTransformer
from tests.model_sizing import create_model_from_params
import jax
import jax.numpy as jnp


def create_model_example(target_params=1_000_000):
    """Create a model with target parameter count"""
    
    print(f"\n{'='*70}")
    print(f"Creating a model with approximately {target_params:,} parameters")
    print(f"{'='*70}\n")
    
    # Create model config automatically based on target parameter count
    config = create_model_from_params(
        target_params=target_params,
        vocab_size=256,  # Your vocabulary size
        max_len=100,     # Maximum sequence length
        prefer_depth=True  # Prefer deeper models over wider ones
    )
    
    # Create the model
    model = ProductionTransformer(config=config)
    
    # Initialize with random parameters
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 50), dtype=jnp.int32)  # batch_size=1, seq_len=50
    params = model.init(rng, dummy_input, deterministic=True)
    
    # Test the model
    output = model.apply(params, dummy_input, deterministic=True)
    print(f"\nModel ready!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    return model, params, config


if __name__ == "__main__":
    # Example 1: Create a 1 million parameter model
    print("\n" + "="*70)
    print("EXAMPLE 1: 1 Million Parameter Model")
    print("="*70)
    model_1m, params_1m, config_1m = create_model_example(1_000_000)
    
    # Example 2: Create a 5 million parameter model
    print("\n" + "="*70)
    print("EXAMPLE 2: 5 Million Parameter Model")
    print("="*70)
    model_5m, params_5m, config_5m = create_model_example(5_000_000)
    
    # Example 3: Custom parameter count
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom 2.5 Million Parameter Model")
    print("="*70)
    model_custom, params_custom, config_custom = create_model_example(2_500_000)
    
    print("\n" + "="*70)
    print("All models created successfully!")
    print("="*70)
    print("\nYou can now use these models for training or inference.")
    print("The model architecture includes modern improvements:")
    print("  ✓ RMSNorm (more efficient than LayerNorm)")
    print("  ✓ SwiGLU activation (more powerful than GELU)")
    print("  ✓ Rotary Position Embeddings (better than learned embeddings)")
