"""
Test script to create and verify models with specific parameter counts.
This script tests 1M, 3M, and 5M parameter models.
"""

import jax
import jax.numpy as jnp
from src.models.model import ProductionTransformer
from model_sizing import create_model_from_params, calculate_model_params
import argparse


def count_params_jax(params):
    """Count parameters in JAX model params"""
    total = 0
    for key, value in params.items():
        if isinstance(value, dict):
            total += count_params_jax(value)
        else:
            total += value.size
    return total


def test_model(target_params, vocab_size=256, max_len=100, batch_size=4, seq_len=32):
    """Test a model with target parameter count"""
    
    print("\n" + "="*70)
    print(f"Testing {target_params:,} parameter model")
    print("="*70)
    
    # Create model config for target parameter count
    config = create_model_from_params(
        target_params=target_params,
        vocab_size=vocab_size,
        max_len=max_len,
        prefer_depth=True
    )
    
    # Calculate theoretical parameters
    theoretical_params = calculate_model_params(config, use_swiglu=True, use_rope=True, use_rmsnorm=True)
    
    # Create model
    model = ProductionTransformer(config=config)
    
    # Initialize model with dummy input
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    print("\nInitializing model...")
    params = model.init(rng, dummy_input, deterministic=True)
    
    # Count actual parameters
    actual_params = count_params_jax(params)
    
    print(f"\n{'='*70}")
    print(f"Parameter Count Summary:")
    print(f"{'='*70}")
    print(f"Target parameters:      {target_params:>15,}")
    print(f"Theoretical parameters: {theoretical_params:>15,}")
    print(f"Actual JAX parameters:  {actual_params:>15,}")
    print(f"Difference from target: {abs(actual_params - target_params):>15,} ({abs(actual_params - target_params) / target_params * 100:>6.2f}%)")
    print(f"{'='*70}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    output = model.apply(params, dummy_input, deterministic=True)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {vocab_size})")
    
    assert output.shape == (batch_size, seq_len, vocab_size), \
        f"Output shape mismatch! Got {output.shape}, expected ({batch_size}, {seq_len}, {vocab_size})"
    
    print("✓ Forward pass successful!")
    
    # Test backward pass (gradient computation)
    print("\nTesting backward pass...")
    
    def loss_fn(params, inputs, labels):
        logits = model.apply(params, inputs, deterministic=True)
        # Simple cross-entropy loss
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.sum(jax.nn.one_hot(labels, vocab_size) * log_probs, axis=-1))
        return loss
    
    dummy_labels = jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)
    
    loss, grads = jax.value_and_grad(loss_fn)(params, dummy_input, dummy_labels)
    print(f"Loss: {loss:.4f}")
    print(f"Gradients computed successfully!")
    print("✓ Backward pass successful!")
    
    # Memory estimate
    param_memory_mb = actual_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"\nEstimated parameter memory: {param_memory_mb:.2f} MB")
    
    return {
        'config': config,
        'target_params': target_params,
        'theoretical_params': theoretical_params,
        'actual_params': actual_params,
        'model': model,
        'params': params
    }


def main():
    parser = argparse.ArgumentParser(description="Test model parameter sizing")
    parser.add_argument('--target-params', type=int, default=None,
                       help='Target parameter count (e.g., 1000000 for 1M)')
    parser.add_argument('--vocab-size', type=int, default=256,
                       help='Vocabulary size')
    parser.add_argument('--max-len', type=int, default=100,
                       help='Maximum sequence length')
    parser.add_argument('--test-all', action='store_true',
                       help='Test all predefined sizes (1M, 3M, 5M)')
    
    args = parser.parse_args()
    
    if args.test_all:
        # Test multiple sizes
        test_sizes = [1_000_000, 3_000_000, 5_000_000]
        results = []
        
        for size in test_sizes:
            result = test_model(
                target_params=size,
                vocab_size=args.vocab_size,
                max_len=args.max_len
            )
            results.append(result)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY OF ALL TESTS")
        print("="*70)
        print(f"{'Target':<12} {'Theoretical':<15} {'Actual':<15} {'Difference':<12} {'Error %':<10}")
        print("-"*70)
        for r in results:
            diff = abs(r['actual_params'] - r['target_params'])
            error_pct = diff / r['target_params'] * 100
            print(f"{r['target_params']:>11,} {r['theoretical_params']:>14,} {r['actual_params']:>14,} {diff:>11,} {error_pct:>9.2f}%")
        print("="*70)
        
    elif args.target_params:
        # Test specific size
        test_model(
            target_params=args.target_params,
            vocab_size=args.vocab_size,
            max_len=args.max_len
        )
    else:
        print("Please specify --target-params or --test-all")
        parser.print_help()


if __name__ == "__main__":
    main()
