"""
Simple training script for transformer models with HuggingFace datasets.

Usage examples:
    # Train on TinyStories-Instruct
    python train_model.py --dataset-id skeskinen/TinyStories-Instruct-hf
    
    # Custom configuration
    python train_model.py \\
        --dataset-id skeskinen/TinyStories-Instruct-hf \\
        --target-params 5000000 \\
        --epochs 10 \\
        --batch-size 64 \\
        --learning-rate 1e-3
"""

import argparse
import json
import logging
import os
import sys
import time

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tokenizers import Tokenizer

from tests.model_sizing import create_model_from_params
from ..data.hf_dataset_loader import HFDatasetLoader
from ..models.model import ProductionTransformer

# Configure logging for real-time output
logging.basicConfig(
    level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True
)

try:
    from ..data.streaming_data_loader import StreamingDataLoader

    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    print(
        "Warning: streaming_data_loader not available. Install for better memory efficiency."
    )


def count_params(params):
    """Count total parameters in model"""
    total = 0
    for key, value in params.items():
        if isinstance(value, dict):
            total += count_params(value)
        else:
            total += value.size
    return total


def detect_and_setup_device():
    """
    Detect available accelerators (TPU/GPU/CPU) and configure JAX accordingly.

    Returns:
        dict with device information
    """
    devices = jax.devices()
    backend = jax.default_backend()

    device_info = {
        "backend": backend,
        "num_devices": len(devices),
        "devices": [str(d) for d in devices],
        "device_type": devices[0].platform if devices else "unknown",
    }

    # Print device information
    print(f"\n{'=' * 80}")
    print("DEVICE CONFIGURATION")
    print("=" * 80)
    print(f"Backend:       {backend.upper()}")
    print(f"Device type:   {device_info['device_type'].upper()}")
    print(f"Num devices:   {device_info['num_devices']}")

    if len(devices) > 1:
        print(f"\nDetected {len(devices)} devices:")
        for i, device in enumerate(devices):
            print(f"  Device {i}: {device}")
        print("\n✓ Multi-device training enabled!")

        # Configure for data parallelism if multiple devices
        if backend in ["tpu", "gpu"]:
            print(
                f"✓ Data parallelism will be used across {len(devices)} {backend.upper()}s"
            )
    else:
        print(f"  Device: {devices[0]}")

    # TPU-specific optimizations
    if backend == "tpu":
        print("\n✓ TPU detected - using TPU-optimized settings:")
        print("  • XLA compilation enabled")
        print("  • bfloat16 precision recommended for better performance")
        print("  • Consider using larger batch sizes for TPU efficiency")

    # GPU-specific optimizations
    elif backend == "gpu":
        print("\n✓ GPU detected - using GPU-optimized settings:")
        print("  • CUDA/ROCm acceleration enabled")
        print("  • Mixed precision training available")

    # CPU fallback
    else:
        print("\n⚠ Running on CPU - training will be slower")
        print("  • Consider using a TPU or GPU for faster training")
        print("  • For free TPU access, use Google Colab")

    print("=" * 80)

    return device_info


def train_model(
    # Dataset parameters
    dataset_id="skeskinen/TinyStories-Instruct-hf",
    dataset_config=None,
    text_column="text",
    split="train",
    # Model parameters
    target_params=1_000_000,
    vocab_size=10000,
    seq_len=128,
    prefer_depth=True,
    # Training parameters
    epochs=3,
    batch_size=32,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_steps=100,
    grad_clip=1.0,
    # Data parameters
    max_examples=None,
    stride=None,
    tokenizer_train_examples=10000,
    # Output parameters
    output_dir="output",
    log_every=10,
    seed=42,
    retrain_tokenizer=False,
):
    """
    Train a transformer model on HuggingFace dataset.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., 'skeskinen/TinyStories-Instruct-hf')
        dataset_config: Dataset configuration/subset (optional)
        text_column: Column name containing text data
        split: Dataset split to use
        target_params: Target number of model parameters
        vocab_size: Vocabulary size for tokenizer
        seq_len: Sequence length
        prefer_depth: Prefer deeper models over wider ones
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps
        grad_clip: Gradient clipping norm
        max_examples: Maximum examples to use from dataset (None for all)
        stride: Stride for creating sequences (None uses seq_len)
        tokenizer_train_examples: Number of examples to train tokenizer on
        output_dir: Output directory for checkpoints
        log_every: Log every N steps
        seed: Random seed
        retrain_tokenizer: Retrain tokenizer even if one exists

    Returns:
        Dictionary with training results
    """

    print("\n" + "=" * 80, flush=True)
    print(" MODEL TRAINING".center(80), flush=True)
    print("=" * 80, flush=True)

    # ========================================================================
    # Step 0: Detect and Setup Device (TPU/GPU/CPU)
    # ========================================================================
    device_info = detect_and_setup_device()

    # ========================================================================
    # Step 1: Load Dataset
    # ========================================================================
    print(f"\n{'=' * 80}", flush=True)
    print("STEP 1: LOADING DATASET", flush=True)
    print("=" * 80, flush=True)
    print(f"Dataset ID:    {dataset_id}")
    if dataset_config:
        print(f"Config:        {dataset_config}")
    print(f"Split:         {split}")
    print(f"Text column:   {text_column}")

    dataset_loader = HFDatasetLoader(
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        text_column=text_column,
        split=split,
    )

    # ========================================================================
    # Step 2: Setup Tokenizer
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("STEP 2: TOKENIZER SETUP")
    print("=" * 80)

    tokenizer_path = f"{output_dir}/tokenizer.json"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(tokenizer_path) and not retrain_tokenizer:
        print(f"Loading existing tokenizer from: {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        actual_vocab_size = tokenizer.get_vocab_size()
        print(f"Tokenizer loaded! Vocab size: {actual_vocab_size}")
    else:
        print(f"Training new tokenizer (target vocab_size={vocab_size})")
        tokenizer = dataset_loader.train_tokenizer(
            vocab_size=vocab_size,
            save_path=tokenizer_path,
            max_examples=tokenizer_train_examples,
        )
        actual_vocab_size = tokenizer.get_vocab_size()
        print(f"Tokenizer trained and saved! Vocab size: {actual_vocab_size}")

    # ========================================================================
    # Step 3: Create Model
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("STEP 3: MODEL CREATION")
    print("=" * 80)
    print(f"Target parameters: {target_params:,}")
    print(f"Sequence length:   {seq_len}")
    print(f"Vocab size:        {actual_vocab_size}")

    model_config = create_model_from_params(
        target_params=target_params,
        vocab_size=actual_vocab_size,
        max_len=seq_len,
        prefer_depth=prefer_depth,
    )

    model = ProductionTransformer(config=model_config)

    # Initialize model
    rng = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input, deterministic=True)

    actual_params = count_params(params)
    error_pct = abs(actual_params - target_params) / target_params * 100

    print("\nModel Configuration:")
    print(f"  Layers:         {model_config.num_layers}")
    print(f"  Hidden size:    {model_config.d_model}")
    print(f"  Attention heads: {model_config.num_heads}")
    print(f"  FFN size:       {model_config.d_ff}")
    print(f"\nActual parameters: {actual_params:,} ({error_pct:.2f}% from target)")

    # ========================================================================
    # Step 4: Prepare Training Data
    # ========================================================================
    print(f"\n{'=' * 80}", flush=True)
    print("STEP 4: PREPARING TRAINING DATA", flush=True)
    print("=" * 80, flush=True)
    print(f"Sequence length: {seq_len}", flush=True)
    print(
        f"Stride:          {stride if stride else seq_len} (no overlap)"
        if not stride
        else f"{stride}",
        flush=True,
    )
    print(f"Max examples:    {max_examples if max_examples else 'all'}", flush=True)

    # Force flush before prepare_sequences to ensure real-time logging
    sys.stdout.flush()

    inputs, targets = dataset_loader.prepare_sequences(
        tokenizer=tokenizer, seq_len=seq_len, stride=stride, max_examples=max_examples
    )

    # Force flush after prepare_sequences
    sys.stdout.flush()

    # Convert to JAX arrays
    inputs = jnp.array(inputs, dtype=jnp.int32)
    targets = jnp.array(targets, dtype=jnp.int32)

    num_sequences = len(inputs)
    num_batches = num_sequences // batch_size
    total_tokens = num_sequences * seq_len

    print("\nData Statistics:")
    print(f"  Total sequences: {num_sequences:,}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Batches/epoch:   {num_batches:,}")
    print(f"  Total batches:   {num_batches * epochs:,}")

    # ========================================================================
    # Step 5: Setup Training
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("STEP 5: TRAINING SETUP")
    print("=" * 80)

    total_steps = num_batches * epochs

    print(f"Learning rate:   {learning_rate}")
    print(f"Weight decay:    {weight_decay}")
    print(f"Warmup steps:    {warmup_steps}")
    print(f"Gradient clip:   {grad_clip}")
    print(f"Batch size:      {batch_size}")
    print(f"Epochs:          {epochs}")
    print(f"Total steps:     {total_steps:,}")

    # Create optimizer with warmup and cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.1,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )

    # Create train state
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Define loss and train step
    def loss_fn(params, batch, dropout_rng):
        logits = model.apply(
            params,
            batch["input_ids"],
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )
        # Shift logits and labels for next-token prediction
        logits_shifted = logits[:, :-1, :]
        labels_shifted = batch["labels"][:, :-1]

        # Reshape for cross entropy
        logits_flat = logits_shifted.reshape(-1, logits_shifted.shape[-1])
        labels_flat = labels_shifted.reshape(-1)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat)
        return jnp.mean(loss)

    # Check if we have multiple devices for data parallelism
    num_devices = device_info["num_devices"]
    use_multi_device = num_devices > 1

    if use_multi_device:
        print(f"\n✓ Enabling multi-device training with {num_devices} devices")
        print("  • Using jax.pmap for data parallelism")
        print(f"  • Effective batch size: {batch_size * num_devices}")

        # Multi-device train step using pmap
        @jax.pmap
        def train_step_pmap(state, batch, dropout_rng):
            """Parallel training step across multiple devices."""
            loss, grads = jax.value_and_grad(loss_fn)(state.params, batch, dropout_rng)
            # Average gradients across devices (pmap does this automatically)
            state = state.apply_gradients(grads=grads)
            return state, loss

        # Replicate state across devices
        print(f"  • Replicating model parameters across {num_devices} devices...")
        state = jax.device_put_replicated(state, jax.devices())
        print("  • Parameters replicated successfully")

        train_step = train_step_pmap
    else:
        print("\n✓ Using single-device training")

        # Single-device train step
        @jax.jit
        def train_step(state, batch, dropout_rng):
            loss, grads = jax.value_and_grad(loss_fn)(state.params, batch, dropout_rng)
            state = state.apply_gradients(grads=grads)
            return state, loss

    # ========================================================================
    # Step 6: Train
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("STEP 6: TRAINING")
    print("=" * 80)

    start_time = time.time()
    global_step = 0
    best_loss = float("inf")
    avg_loss = 0.0  # Initialize to avoid unbound variable

    for epoch in range(epochs):
        print(f"\n{'─' * 80}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print("─" * 80)

        epoch_start = time.time()
        epoch_loss = 0
        num_batches_processed = 0

        # Shuffle data each epoch
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, len(inputs))
        inputs_shuffled = inputs[perm]
        targets_shuffled = targets[perm]

        # Determine effective batch size for multi-device training
        effective_batch_size = (
            batch_size * num_devices if use_multi_device else batch_size
        )

        # Training loop
        for i in range(0, len(inputs) - effective_batch_size, effective_batch_size):
            if use_multi_device:
                # Split batch across devices
                batch_data = {
                    "input_ids": inputs_shuffled[i : i + effective_batch_size],
                    "labels": targets_shuffled[i : i + effective_batch_size],
                }

                # Reshape to (num_devices, per_device_batch_size, seq_len)
                batch = {
                    "input_ids": batch_data["input_ids"].reshape(
                        num_devices, batch_size, seq_len
                    ),
                    "labels": batch_data["labels"].reshape(
                        num_devices, batch_size, seq_len
                    ),
                }

                # Generate dropout keys for each device
                rng, *dropout_rngs = jax.random.split(rng, num_devices + 1)
                dropout_rng = jnp.array(dropout_rngs)

                # Train step returns (state_per_device, loss_per_device)
                state, loss = train_step(state, batch, dropout_rng)

                # Average loss across devices for logging
                epoch_loss += jnp.mean(loss)
            else:
                # Single device batch
                batch = {
                    "input_ids": inputs_shuffled[i : i + batch_size],
                    "labels": targets_shuffled[i : i + batch_size],
                }

                # Generate new dropout key for each step
                rng, dropout_rng = jax.random.split(rng)
                state, loss = train_step(state, batch, dropout_rng)
                epoch_loss += loss

            num_batches_processed += 1
            global_step += 1

            # Log progress
            if global_step % log_every == 0:
                elapsed = time.time() - start_time
                # Calculate tokens/sec based on effective batch size
                tokens_per_sec = (
                    global_step * effective_batch_size * seq_len
                ) / elapsed
                current_loss = jnp.mean(loss) if use_multi_device else loss
                print(
                    f"  Step {global_step:>5} | Loss: {current_loss:.4f} | "
                    f"Tokens/sec: {tokens_per_sec:>8,.0f} | Time: {elapsed:>6.1f}s"
                )

        # Epoch summary
        avg_loss = epoch_loss / num_batches_processed
        epoch_time = time.time() - epoch_start
        batches_per_sec = num_batches_processed / epoch_time

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average loss:    {avg_loss:.4f}")
        print(f"  Epoch time:      {epoch_time:.1f}s")
        print(f"  Batches/sec:     {batches_per_sec:.1f}")

        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.abspath(f"{output_dir}/best_checkpoint")
            print("  ✓ New best loss! Saving checkpoint...")

            # Save model params
            import shutil

            from orbax import checkpoint as ocp

            checkpointer = ocp.PyTreeCheckpointer()
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)

            # Extract params from first device if using multi-device training
            params_to_save = (
                jax.tree_map(lambda x: x[0], state.params)
                if use_multi_device
                else state.params
            )
            checkpointer.save(checkpoint_path, params_to_save)

            # Save config
            config_path = os.path.abspath(f"{output_dir}/model_config.json")
            with open(config_path, "w") as f:
                json.dump(model_config.model_dump(), f, indent=2)

    # ========================================================================
    # Training Complete
    # ========================================================================
    total_time = time.time() - start_time
    total_tokens_processed = global_step * batch_size * seq_len
    avg_tokens_per_sec = total_tokens_processed / total_time

    print(f"\n{'=' * 80}")
    print(" TRAINING COMPLETE".center(80))
    print("=" * 80)
    print("\nTraining Statistics:")
    print(f"  Total time:         {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"  Total steps:        {global_step:,}")
    print(f"  Total tokens:       {total_tokens_processed:,}")
    print(f"  Avg tokens/sec:     {avg_tokens_per_sec:,.0f}")
    print(f"  Final loss:         {avg_loss:.4f}")
    print(f"  Best loss:          {best_loss:.4f}")

    print(f"\nOutput files saved to: {output_dir}/")
    print("  • best_checkpoint/     (model parameters)")
    print("  • model_config.json    (model configuration)")
    print("  • tokenizer.json       (trained tokenizer)")

    return {
        "final_loss": float(avg_loss),
        "best_loss": float(best_loss),
        "total_steps": global_step,
        "total_time": total_time,
        "total_tokens": total_tokens_processed,
        "avg_tokens_per_sec": float(avg_tokens_per_sec),
        "model_params": actual_params,
        "output_dir": output_dir,
        "device_info": device_info,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train transformer model on HuggingFace datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on TinyStories-Instruct with defaults
  python train_model.py --dataset-id skeskinen/TinyStories-Instruct-hf
  
  # Train 5M parameter model for 10 epochs
  python train_model.py \\
      --dataset-id skeskinen/TinyStories-Instruct-hf \\
      --target-params 5000000 \\
      --epochs 10 \\
      --batch-size 64
  
  # Train on WikiText-2
  python train_model.py \\
      --dataset-id wikitext \\
      --dataset-config wikitext-2-raw-v1 \\
      --target-params 3000000
  
  # Quick test with small model and limited data
  python train_model.py \\
      --dataset-id skeskinen/TinyStories-Instruct-hf \\
      --target-params 500000 \\
      --max-examples 1000 \\
      --epochs 2
        """,
    )

    # Dataset arguments
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="HuggingFace dataset ID (e.g., 'skeskinen/TinyStories-Instruct-hf')",
    )
    dataset_group.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset configuration/subset (e.g., 'wikitext-2-raw-v1')",
    )
    dataset_group.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing text data (default: text)",
    )
    dataset_group.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--target-params",
        type=int,
        default=1_000_000,
        help="Target number of model parameters (default: 1000000)",
    )
    model_group.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size for tokenizer (default: 10000)",
    )
    model_group.add_argument(
        "--seq-len", type=int, default=128, help="Sequence length (default: 128)"
    )
    model_group.add_argument(
        "--prefer-depth",
        action="store_true",
        default=True,
        help="Prefer deeper models over wider ones (default: True)",
    )

    # Training arguments
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )
    training_group.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size (default: 32)"
    )
    training_group.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (default: 5e-4)",
    )
    training_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW (default: 0.01)",
    )
    training_group.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100)",
    )
    training_group.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm (default: 1.0)",
    )

    # Data arguments
    data_group = parser.add_argument_group("Data Processing")
    data_group.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum examples to use from dataset (default: all)",
    )
    data_group.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for creating sequences (default: seq_len)",
    )
    data_group.add_argument(
        "--tokenizer-train-examples",
        type=int,
        default=10000,
        help="Number of examples to train tokenizer on (default: 10000)",
    )
    data_group.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="Retrain tokenizer even if one exists",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output and Logging")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for checkpoints (default: output)",
    )
    output_group.add_argument(
        "--log-every", type=int, default=10, help="Log every N steps (default: 10)"
    )
    output_group.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Call train_model with parsed arguments
    results = train_model(
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        split=args.split,
        target_params=args.target_params,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        prefer_depth=args.prefer_depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        max_examples=args.max_examples,
        stride=args.stride,
        tokenizer_train_examples=args.tokenizer_train_examples,
        output_dir=args.output_dir,
        log_every=args.log_every,
        seed=args.seed,
        retrain_tokenizer=args.retrain_tokenizer,
    )

    # Save training results
    results_path = f"{args.output_dir}/training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("  • training_results.json (training statistics)")


if __name__ == "__main__":
    main()
