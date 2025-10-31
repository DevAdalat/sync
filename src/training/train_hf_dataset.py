"""
Train a transformer model on HuggingFace datasets with parameter-based sizing.

This script allows you to:
1. Specify target parameter count for the model
2. Load any HuggingFace dataset by ID
3. Select specific text columns from the dataset
4. Train with automatic tokenization
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from ...tests.model_sizing import create_model_from_params
from ..data.hf_dataset_loader import HFDatasetLoader, list_dataset_columns
from ..models.model import ProductionTransformer


def count_params(params):
    """Count total parameters in model"""
    total = 0
    for key, value in params.items():
        if isinstance(value, dict):
            total += count_params(value)
        else:
            total += value.size
    return total


def train_model(args):
    """Train model with specified configuration"""

    print("\n" + "=" * 70)
    print("MODEL AND DATASET CONFIGURATION")
    print("=" * 70)

    # Step 1: Load dataset
    print(f"\n1. Loading dataset: {args.dataset}")
    if args.dataset_config:
        print(f"   Config: {args.dataset_config}")
    print(f"   Split: {args.split}")
    print(f"   Text column: {args.text_column}")

    dataset_loader = HFDatasetLoader(
        dataset_id=args.dataset,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        split=args.split,
    )

    # Step 2: Train or load tokenizer
    print(f"\n2. Setting up tokenizer (vocab_size={args.vocab_size})")

    tokenizer_path = f"{args.output_dir}/tokenizer.json"
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(tokenizer_path) and not args.retrain_tokenizer:
        print(f"   Loading existing tokenizer from: {tokenizer_path}")
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
    else:
        print("   Training new tokenizer...")
        tokenizer = dataset_loader.train_tokenizer(
            vocab_size=args.vocab_size,
            save_path=tokenizer_path,
            max_examples=args.tokenizer_train_examples,
        )
        vocab_size = tokenizer.get_vocab_size()

    print(f"   Tokenizer ready! Vocab size: {vocab_size}")

    # Step 3: Create model configuration based on target parameters
    print(f"\n3. Creating model with ~{args.target_params:,} parameters")

    model_config = create_model_from_params(
        target_params=args.target_params,
        vocab_size=vocab_size,
        max_len=args.seq_len,
        prefer_depth=args.prefer_depth,
    )

    # Create model
    model = ProductionTransformer(config=model_config)

    # Initialize model
    rng = jax.random.PRNGKey(args.seed)
    dummy_input = jnp.ones((1, args.seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input, deterministic=True)

    actual_params = count_params(params)
    print(f"\n   Model initialized with {actual_params:,} parameters")

    # Step 4: Prepare training data
    print("\n4. Preparing training sequences")
    print(f"   Sequence length: {args.seq_len}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Max examples: {args.max_examples if args.max_examples else 'all'}")

    inputs, targets = dataset_loader.prepare_sequences(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        stride=args.stride,
        max_examples=args.max_examples,
    )

    # Convert to JAX arrays
    inputs = jnp.array(inputs, dtype=jnp.int32)
    targets = jnp.array(targets, dtype=jnp.int32)

    print(f"   Total sequences: {len(inputs)}")
    print(f"   Total batches: {len(inputs) // args.batch_size}")

    # Step 5: Setup training
    print("\n5. Setting up training")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Weight decay: {args.weight_decay}")

    # Create optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.max_steps
        if args.max_steps
        else len(inputs) // args.batch_size * args.epochs,
        end_value=args.learning_rate * 0.1,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
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
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            batch["labels"][:, :-1].reshape(-1),
        )
        return jnp.mean(loss)

    @jax.jit
    def train_step(state, batch, dropout_rng):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch, dropout_rng)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Step 6: Train
    print("\n6. Starting training")
    print("=" * 70)

    start_time = time.time()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0
        num_batches = 0

        # Shuffle data each epoch
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, len(inputs))
        inputs_shuffled = inputs[perm]
        targets_shuffled = targets[perm]

        # Training loop
        for i in range(0, len(inputs) - args.batch_size, args.batch_size):
            batch = {
                "input_ids": inputs_shuffled[i : i + args.batch_size],
                "labels": targets_shuffled[i : i + args.batch_size],
            }

            # Generate new dropout key for each step
            rng, dropout_rng = jax.random.split(rng)
            state, loss = train_step(state, batch, dropout_rng)
            epoch_loss += loss
            num_batches += 1
            global_step += 1

            # Log progress
            if global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | Step {global_step} | "
                    f"Loss: {loss:.4f} | Time: {elapsed:.1f}s"
                )

            # Check max steps
            if args.max_steps and global_step >= args.max_steps:
                break

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Throughput: {num_batches / epoch_time:.1f} batches/sec")

        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.abspath(f"{args.output_dir}/best_checkpoint")
            print(f"  New best loss! Saving checkpoint to {checkpoint_path}")

            # Save model params and config
            from orbax import checkpoint as ocp

            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(checkpoint_path, state.params)

            # Save config
            config_path = os.path.abspath(f"{args.output_dir}/model_config.json")
            with open(config_path, "w") as f:
                json.dump(model_config.dict(), f, indent=2)

        print("-" * 70)

        if args.max_steps and global_step >= args.max_steps:
            break

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Total steps: {global_step}")
    print(f"\nModel saved to: {args.output_dir}")
    print(f"  - Checkpoint: {args.output_dir}/best_checkpoint")
    print(f"  - Config: {args.output_dir}/model_config.json")
    print(f"  - Tokenizer: {args.output_dir}/tokenizer.json")


def main():
    parser = argparse.ArgumentParser(
        description="Train transformer model on HuggingFace datasets with parameter-based sizing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   # Train 1M parameter model on WikiText-2
   python train_hf_dataset.py --dataset wikitext --dataset-config wikitext-2-raw-v1 \\
       --text-column text --target-params 1000000 --epochs 3

   # Train 5M parameter model on TinyStories
   python train_hf_dataset.py --dataset roneneldan/TinyStories \\
       --text-column text --target-params 5000000 --epochs 5

   # Train on custom dataset
   python train_hf_dataset.py --dataset iohadrubin/wikitext-103-raw-v1 \\
       --text-column text --target-params 1000000 --epochs 3

   # List available columns in a dataset
   python train_hf_dataset.py --list-columns --dataset wikitext --dataset-config wikitext-2-raw-v1
        """,
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset ID (e.g., 'wikitext', 'roneneldan/TinyStories', 'iohadrubin/wikitext-103-raw-v1')",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset configuration/subset (e.g., 'wikitext-2-raw-v1')",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing text data (default: 'text')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: 'train')",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="List available columns in the dataset and exit",
    )

    # Model arguments
    parser.add_argument(
        "--target-params",
        type=int,
        default=1_000_000,
        help="Target number of model parameters (default: 1000000)",
    )
    parser.add_argument(
        "--prefer-depth",
        action="store_true",
        default=True,
        help="Prefer deeper models over wider ones",
    )

    # Tokenizer arguments
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size for tokenizer (default: 10000)",
    )
    parser.add_argument(
        "--tokenizer-train-examples",
        type=int,
        default=10000,
        help="Number of examples to train tokenizer on (default: 10000)",
    )
    parser.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="Retrain tokenizer even if one exists",
    )

    # Training arguments
    parser.add_argument(
        "--seq-len", type=int, default=128, help="Sequence length (default: 128)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for creating sequences (default: seq_len)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs (default: 3)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (default: None)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum examples to use from dataset (default: all)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=100, help="Warmup steps (default: 100)"
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm (default: 1.0)",
    )
    parser.add_argument(
        "--log-every", type=int, default=10, help="Log every N steps (default: 10)"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for checkpoints (default: 'output')",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # List columns if requested
    if args.list_columns:
        list_dataset_columns(args.dataset, args.dataset_config, args.split)
        return

    # Train model
    train_model(args)


if __name__ == "__main__":
    main()
