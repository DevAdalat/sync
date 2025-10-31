"""
HuggingFace Dataset Loader Utility

Supports loading datasets from HuggingFace Hub with:
- Dataset ID specification
- Column/field selection
- Automatic tokenization
- Sequence creation for language modeling
"""

import logging
import multiprocessing
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

logger = logging.getLogger(__name__)


class HFDatasetLoader:
    """
    Load and preprocess datasets from HuggingFace Hub.

    Example usage:
        loader = HFDatasetLoader(
            dataset_id="iohadrubin/wikitext-103-raw-v1",
            text_column="text",
            split="train"
        )

        # Train tokenizer
        tokenizer = loader.train_tokenizer(vocab_size=10000)

        # Prepare sequences
        inputs, targets = loader.prepare_sequences(tokenizer, seq_len=128)
    """

    def __init__(
        self,
        dataset_id: str,
        text_column: str = "text",
        dataset_config: Optional[str] = None,
        split: str = "train",
        streaming: bool = True,  # Default to streaming for memory efficiency
        trust_remote_code: bool = False,
    ):
        """
        Initialize dataset loader.

        Args:
            dataset_id: HuggingFace dataset ID (e.g., "wikitext", "openwebtext")
            text_column: Column name containing text data
            dataset_config: Dataset configuration/subset (e.g., "wikitext-2-raw-v1")
            split: Dataset split ("train", "validation", "test")
            streaming: Whether to stream the dataset (recommended for memory efficiency)
            trust_remote_code: Trust remote code when loading dataset
        """
        self.dataset_id = dataset_id
        self.text_column = text_column
        self.dataset_config = dataset_config
        self.split = split
        self.streaming = streaming
        self.trust_remote_code = trust_remote_code

        logger.info(f"Loading dataset: {dataset_id}")
        if dataset_config:
            logger.info(f"  Config: {dataset_config}")
        logger.info(f"  Split: {split}")
        logger.info(f"  Text column: {text_column}")
        logger.info(
            f"  Streaming mode: {streaming} ({'Memory efficient!' if streaming else 'Loads all data'})"
        )

        # Load dataset
        self.dataset = load_dataset(
            dataset_id,
            dataset_config,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
        )

        # Verify text column exists
        if not streaming:
            if text_column not in self.dataset.column_names:
                raise ValueError(
                    f"Column '{text_column}' not found in dataset. "
                    f"Available columns: {self.dataset.column_names}"
                )

        logger.info("Dataset loaded successfully!")
        if not streaming:
            logger.info(f"  Total examples: {len(self.dataset)}")

    def get_text_data(self, max_examples: Optional[int] = None) -> List[str]:
        """
        Extract text data from the dataset.

        Args:
            max_examples: Maximum number of examples to extract (None for all)

        Returns:
            List of text strings
        """
        texts = []

        if self.streaming:
            # For streaming datasets
            for i, example in enumerate(self.dataset):
                if max_examples and i >= max_examples:
                    break
                text = example[self.text_column]
                if text and len(text.strip()) > 0:
                    texts.append(text)
        else:
            # For non-streaming datasets
            dataset_slice = (
                self.dataset
                if max_examples is None
                else self.dataset.select(range(min(max_examples, len(self.dataset))))
            )
            for example in dataset_slice:
                text = example[self.text_column]
                if text and len(text.strip()) > 0:
                    texts.append(text)

        logger.info(f"Extracted {len(texts)} text examples")
        return texts

    def train_tokenizer(
        self,
        vocab_size: int = 10000,
        min_frequency: int = 2,
        special_tokens: List[str] = None,
        save_path: Optional[str] = None,
        max_examples: Optional[int] = 10000,
    ) -> Tokenizer:
        """
        Train a BPE tokenizer on the dataset.

        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum token frequency
            special_tokens: List of special tokens
            save_path: Path to save tokenizer (optional)
            max_examples: Maximum examples to use for training (None for all)

        Returns:
            Trained tokenizer
        """
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

        logger.info(f"Training tokenizer with vocab_size={vocab_size}")

        # Get text data
        texts = self.get_text_data(max_examples=max_examples)

        # Initialize tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Train tokenizer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)

        logger.info(f"Tokenizer trained! Vocab size: {tokenizer.get_vocab_size()}")

        # Save if path provided
        if save_path:
            tokenizer.save(save_path)
            logger.info(f"Tokenizer saved to: {save_path}")

        return tokenizer

    def prepare_sequences(
        self,
        tokenizer: Union[Tokenizer, str],
        seq_len: int = 128,
        stride: Optional[int] = None,
        max_examples: Optional[int] = None,
        num_workers: Optional[int] = None,
        use_gpu: bool = True,
        memory_efficient: bool = True,
    ) -> tuple:
        """
        Prepare input and target sequences for language modeling.
        Optimized with multithreading (CPU), GPU acceleration, and memory efficiency.

        Args:
            tokenizer: Tokenizer object or path to tokenizer file
            seq_len: Sequence length
            stride: Stride for overlapping sequences (default: seq_len, no overlap)
            max_examples: Maximum number of examples to process (None for all)
            num_workers: Number of threads for tokenization (default: CPU count)
            use_gpu: Whether to use GPU for sequence creation (default: True)
            memory_efficient: Use memory-efficient streaming approach (default: True)

        Returns:
            Tuple of (inputs, targets) as JAX arrays or lists
        """
        # Load tokenizer if path provided
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer.from_file(tokenizer)

        if stride is None:
            stride = seq_len

        logger.info(f"Preparing sequences (seq_len={seq_len}, stride={stride})")
        logger.info(f"Memory efficient: {memory_efficient}, GPU: {use_gpu}")

        if memory_efficient:
            return self._prepare_sequences_memory_efficient(
                tokenizer, seq_len, stride, max_examples, use_gpu
            )
        else:
            return self._prepare_sequences_original(
                tokenizer, seq_len, stride, max_examples, num_workers, use_gpu
            )

    def _prepare_sequences_memory_efficient(
        self,
        tokenizer: Tokenizer,
        seq_len: int,
        stride: int,
        max_examples: Optional[int],
        use_gpu: bool,
    ) -> tuple:
        """
        Memory-efficient sequence preparation using streaming approach.
        Avoids the 34x memory blowup by never loading all data at once.
        """
        logger.info("Using memory-efficient streaming approach...")

        # First pass: count total tokens (streaming)
        logger.info("Counting total tokens...")
        token_stream = self._stream_tokens(tokenizer, max_examples)
        total_tokens = sum(1 for _ in token_stream)
        logger.info(f"Total tokens: {total_tokens}")

        # Calculate number of sequences
        num_sequences = (total_tokens - seq_len) // stride + 1
        logger.info(f"Will create {num_sequences} sequences")

        # Pre-allocate arrays (this is the only memory we need!)
        if use_gpu:
            inputs = jnp.zeros((num_sequences, seq_len), dtype=jnp.int32)
            targets = jnp.zeros((num_sequences, seq_len), dtype=jnp.int32)
        else:
            inputs = np.zeros((num_sequences, seq_len), dtype=np.int32)
            targets = np.zeros((num_sequences, seq_len), dtype=np.int32)

        # Second pass: fill arrays with sequences (streaming)
        logger.info("Creating sequences (streaming)...")
        token_stream = self._stream_tokens(tokenizer, max_examples)

        # Buffer for sliding window
        token_buffer = []
        seq_idx = 0

        for token in token_stream:
            token_buffer.append(token)

            # Keep buffer size manageable
            if len(token_buffer) > seq_len + stride:
                token_buffer = token_buffer[-(seq_len + stride) :]

            # Create sequences when we have enough tokens
            while len(token_buffer) >= seq_len + 1:
                # Extract input and target sequences
                input_seq = token_buffer[:seq_len]
                target_seq = token_buffer[1 : seq_len + 1]

                # Store in arrays
                if use_gpu:
                    inputs = inputs.at[seq_idx].set(
                        jnp.array(input_seq, dtype=jnp.int32)
                    )
                    targets = targets.at[seq_idx].set(
                        jnp.array(target_seq, dtype=jnp.int32)
                    )
                else:
                    inputs[seq_idx] = input_seq
                    targets[seq_idx] = target_seq

                seq_idx += 1

                # Move window forward
                token_buffer = token_buffer[stride:]

                # Progress reporting and cleanup
                if seq_idx % 1000 == 0 or seq_idx == num_sequences - 1:
                    percent_done = (seq_idx / num_sequences) * 100
                    logger.info(
                        f"Progress: {percent_done:.1f}% ({seq_idx}/{num_sequences} sequences)"
                    )
                    sys.stdout.flush()  # Force real-time output
                    import gc

                    gc.collect()

        logger.info(f"Successfully created {seq_idx} sequences")

        # Trim arrays if we created fewer than expected
        if seq_idx < num_sequences:
            inputs = inputs[:seq_idx]
            targets = targets[:seq_idx]

        return inputs, targets

    def _stream_tokens(self, tokenizer: Tokenizer, max_examples: Optional[int]):
        """Stream tokens one by one (memory efficient)."""
        count = 0
        for example in self.dataset:
            if max_examples and count >= max_examples:
                break

            text = example[self.text_column]
            if text and len(text.strip()) > 0:
                # Tokenize and yield tokens one by one
                tokens = tokenizer.encode(text).ids
                for token in tokens:
                    yield token

            count += 1
            if count % 100 == 0:
                logger.info(f"Tokenizing: {count} examples processed...")
                sys.stdout.flush()  # Force real-time output

    def _prepare_sequences_original(
        self,
        tokenizer: Tokenizer,
        seq_len: int,
        stride: int,
        max_examples: Optional[int],
        num_workers: Optional[int],
        use_gpu: bool,
    ) -> tuple:
        """
        Original sequence preparation method (memory intensive).
        Kept for backward compatibility and comparison.
        """
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)

        logger.info(f"Using original method with {num_workers} threads, GPU: {use_gpu}")

        # Get text data (loads all texts into memory)
        texts = self.get_text_data(max_examples=max_examples)

        # Parallel tokenization using multithreading
        def tokenize_batch(text_batch):
            """Tokenize a batch of texts"""
            tokens = []
            for text in text_batch:
                tokens.extend(tokenizer.encode(text).ids)
            return tokens

        # Split texts into chunks for parallel processing
        chunk_size = max(1, len(texts) // num_workers)
        text_chunks = [
            texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)
        ]

        logger.info(f"Tokenizing {len(texts)} texts in {len(text_chunks)} chunks...")

        all_tokens = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tokenization tasks
            future_to_chunk = {
                executor.submit(tokenize_batch, chunk): i
                for i, chunk in enumerate(text_chunks)
            }

            # Collect results in order
            results = [None] * len(text_chunks)
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                results[chunk_idx] = future.result()

            # Flatten results
            for result in results:
                all_tokens.extend(result)

        logger.info(f"Total tokens: {len(all_tokens)}")

        # Create sequences using GPU if available and beneficial
        if use_gpu and len(all_tokens) > 10000:  # Use GPU for large datasets
            logger.info("Using GPU for sequence creation...")
            inputs, targets = self._create_sequences_gpu(all_tokens, seq_len, stride)
        else:
            logger.info("Using CPU for sequence creation...")
            inputs, targets = self._create_sequences_cpu(all_tokens, seq_len, stride)

        logger.info(f"Created {len(inputs)} sequences")

        return inputs, targets

    def _create_sequences_cpu(
        self, all_tokens: List[int], seq_len: int, stride: int
    ) -> tuple:
        """Create sequences using CPU (memory efficient - no duplication)"""
        # Calculate number of sequences
        num_sequences = (len(all_tokens) - seq_len) // stride + 1

        # Pre-allocate arrays (NO duplication!)
        inputs = np.zeros((num_sequences, seq_len), dtype=np.int32)
        targets = np.zeros((num_sequences, seq_len), dtype=np.int32)

        # Fill arrays directly (no intermediate lists)
        for i in range(num_sequences):
            start_idx = i * stride
            inputs[i] = all_tokens[start_idx : start_idx + seq_len]
            targets[i] = all_tokens[start_idx + 1 : start_idx + seq_len + 1]

        return inputs, targets

    def _create_sequences_gpu(
        self, all_tokens: List[int], seq_len: int, stride: int
    ) -> tuple:
        """Create sequences using GPU with JAX (memory efficient - returns JAX arrays)"""
        # Convert to JAX array
        tokens_array = jnp.array(all_tokens, dtype=jnp.int32)

        # Calculate valid starting indices
        max_start = len(all_tokens) - seq_len
        num_sequences = (max_start // stride) + (1 if max_start % stride == 0 else 0)

        # Create index array on GPU
        start_indices = jnp.arange(0, max_start, stride, dtype=jnp.int32)

        # Vectorized sequence creation using vmap
        def extract_input_seq(start_idx):
            return jax.lax.dynamic_slice(tokens_array, (start_idx,), (seq_len,))

        def extract_target_seq(start_idx):
            return jax.lax.dynamic_slice(tokens_array, (start_idx + 1,), (seq_len,))

        # Use vmap for parallel extraction on GPU
        inputs = jax.vmap(extract_input_seq)(start_indices)
        targets = jax.vmap(extract_target_seq)(start_indices)

        # Return JAX arrays directly (no memory-wasting list conversion)
        return inputs, targets

    def prepare_sequences_fast(
        self,
        tokenizer: Union[Tokenizer, str],
        seq_len: int = 128,
        stride: Optional[int] = None,
        max_examples: Optional[int] = None,
        num_workers: Optional[int] = None,
        return_jax: bool = True,
    ):
        """
        Optimized version that returns JAX arrays directly (no list conversion).
        Best for training workflows that expect JAX arrays.

        Args:
            tokenizer: Tokenizer object or path to tokenizer file
            seq_len: Sequence length
            stride: Stride for overlapping sequences (default: seq_len, no overlap)
            max_examples: Maximum number of examples to process (None for all)
            num_workers: Number of threads for tokenization (default: CPU count)
            return_jax: Return JAX arrays (True) or numpy arrays (False)

        Returns:
            Tuple of (inputs, targets) as JAX or numpy arrays
        """
        # Load tokenizer if path provided
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer.from_file(tokenizer)

        if stride is None:
            stride = seq_len

        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)

        logger.info(f"Preparing sequences (seq_len={seq_len}, stride={stride})")
        logger.info(f"Using {num_workers} threads for tokenization")

        # Get text data
        texts = self.get_text_data(max_examples=max_examples)

        # Parallel tokenization
        def tokenize_batch(text_batch):
            tokens = []
            for text in text_batch:
                tokens.extend(tokenizer.encode(text).ids)
            return tokens

        chunk_size = max(1, len(texts) // num_workers)
        text_chunks = [
            texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)
        ]

        logger.info(f"Tokenizing {len(texts)} texts in {len(text_chunks)} chunks...")

        all_tokens = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk = {
                executor.submit(tokenize_batch, chunk): i
                for i, chunk in enumerate(text_chunks)
            }
            results = [None] * len(text_chunks)
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                results[chunk_idx] = future.result()
            for result in results:
                all_tokens.extend(result)

        logger.info(f"Total tokens: {len(all_tokens)}")
        logger.info("Using GPU for sequence creation...")

        # GPU-based sequence creation (no list conversion)
        tokens_array = jnp.array(all_tokens, dtype=jnp.int32)
        max_start = len(all_tokens) - seq_len
        start_indices = jnp.arange(0, max_start, stride, dtype=jnp.int32)

        def extract_input_seq(start_idx):
            return jax.lax.dynamic_slice(tokens_array, (start_idx,), (seq_len,))

        def extract_target_seq(start_idx):
            return jax.lax.dynamic_slice(tokens_array, (start_idx + 1,), (seq_len,))

        inputs = jax.vmap(extract_input_seq)(start_indices)
        targets = jax.vmap(extract_target_seq)(start_indices)

        logger.info(f"Created {len(inputs)} sequences")

        if not return_jax:
            inputs = np.array(inputs)
            targets = np.array(targets)

        return inputs, targets

    def create_batch_iterator(
        self,
        tokenizer: Union[Tokenizer, str],
        batch_size: int,
        seq_len: int = 128,
        stride: Optional[int] = None,
        max_examples: Optional[int] = None,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
    ):
        """
        Create a batch iterator for training (now optimized).

        Args:
            tokenizer: Tokenizer object or path to tokenizer file
            batch_size: Batch size
            seq_len: Sequence length
            stride: Stride for overlapping sequences
            max_examples: Maximum number of examples to process
            shuffle: Whether to shuffle the data
            num_workers: Number of threads for tokenization (default: CPU count)

        Yields:
            Batches of (inputs, targets) as JAX arrays
        """
        # Use optimized method that returns JAX arrays directly
        inputs, targets = self.prepare_sequences_fast(
            tokenizer, seq_len, stride, max_examples, num_workers, return_jax=True
        )

        # Shuffle if requested
        if shuffle:
            rng = jax.random.PRNGKey(42)
            perm = jax.random.permutation(rng, len(inputs))
            inputs = inputs[perm]
            targets = targets[perm]

        # Yield batches
        num_batches = len(inputs) // batch_size
        logger.info(
            f"Creating batch iterator: {num_batches} batches of size {batch_size}"
        )

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            yield {
                "input_ids": inputs[start_idx:end_idx],
                "labels": targets[start_idx:end_idx],
            }


def list_dataset_columns(
    dataset_id: str, dataset_config: Optional[str] = None, split: str = "train"
) -> List[str]:
    """
    List available columns in a dataset.

    Args:
        dataset_id: HuggingFace dataset ID
        dataset_config: Dataset configuration
        split: Dataset split

    Returns:
        List of column names
    """
    dataset = load_dataset(dataset_id, dataset_config, split=split, streaming=True)

    # Get first example to see columns
    first_example = next(iter(dataset))
    columns = list(first_example.keys())

    print(f"\nDataset: {dataset_id}")
    if dataset_config:
        print(f"Config: {dataset_config}")
    print(f"Split: {split}")
    print("\nAvailable columns:")
    for col in columns:
        print(f"  - {col}")

    return columns


if __name__ == "__main__":
    # Example 1: Load wikitext dataset
    print("=" * 70)
    print("Example 1: WikiText-2 Dataset")
    print("=" * 70)

    loader = HFDatasetLoader(
        dataset_id="iohadrubin/wikitext-103-raw-v1", text_column="text", split="train"
    )

    # Train tokenizer
    tokenizer = loader.train_tokenizer(vocab_size=5000, max_examples=1000)

    # Prepare sequences
    inputs, targets = loader.prepare_sequences(tokenizer, seq_len=64, max_examples=100)
    print(f"\nPrepared {len(inputs)} sequences")
    print(f"Input shape: {len(inputs[0])} tokens")
    print(f"Target shape: {len(targets[0])} tokens")

    # Example 2: List available columns
    print("\n" + "=" * 70)
    print("Example 2: List Dataset Columns")
    print("=" * 70)

    list_dataset_columns("iohadrubin/wikitext-103-raw-v1", split="train")
