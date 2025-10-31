"""
Streaming Data Loader - Memory Efficient Training

This module provides a memory-efficient streaming data loader that:
1. Only loads data needed for the current batch
2. Tokenizes on-the-fly to avoid storing tokenized data in memory
3. Supports HuggingFace datasets with streaming mode
4. Works with all backends (JAX, PyTorch, TensorFlow)
"""

import logging
from typing import Dict, Iterator, List, Optional, Union

import jax.numpy as jnp
from datasets import load_dataset
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class StreamingDataLoader:
    """
    Memory-efficient streaming data loader that loads only batch data at a time.

    Features:
    - Streaming mode: loads data on-demand, not all at once
    - On-the-fly tokenization: tokenizes during iteration, not upfront
    - Minimal memory footprint: only keeps current batch in memory
    - Automatic data cycling: handles epoch boundaries automatically
    """

    def __init__(
        self,
        dataset_id: str,
        tokenizer: Union[Tokenizer, str],
        seq_len: int = 128,
        batch_size: int = 32,
        text_column: str = "text",
        dataset_config: Optional[str] = None,
        split: str = "train",
        max_examples: Optional[int] = None,
        stride: Optional[int] = None,
        streaming: bool = True,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ):
        """
        Initialize streaming data loader.

        Args:
            dataset_id: HuggingFace dataset ID
            tokenizer: Tokenizer object or path to tokenizer file
            seq_len: Sequence length for training
            batch_size: Batch size
            text_column: Column name containing text
            dataset_config: Dataset configuration/subset
            split: Dataset split to use
            max_examples: Maximum examples to use (None for all)
            stride: Stride for sequence creation (None = seq_len)
            streaming: Use streaming mode (recommended for large datasets)
            shuffle_buffer_size: Buffer size for shuffling (higher = more random but more memory)
            seed: Random seed for shuffling
        """
        self.dataset_id = dataset_id
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.text_column = text_column
        self.stride = stride if stride else seq_len
        self.streaming = streaming
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.max_examples = max_examples

        # Load tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = Tokenizer.from_file(tokenizer)
        else:
            self.tokenizer = tokenizer

        logger.info("Initializing StreamingDataLoader")
        logger.info(f"  Dataset: {dataset_id}")
        logger.info(f"  Streaming mode: {streaming}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Sequence length: {seq_len}")
        logger.info(f"  Stride: {self.stride}")

        # Load dataset in streaming mode
        self.dataset = load_dataset(
            dataset_id,
            dataset_config,
            split=split,
            streaming=streaming,
            trust_remote_code=False,
        )

        # Apply max_examples limit if specified
        if max_examples is not None and not streaming:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )
            logger.info(f"  Limited to {len(self.dataset)} examples")

        # Shuffle dataset
        if streaming:
            self.dataset = self.dataset.shuffle(
                seed=seed, buffer_size=shuffle_buffer_size
            )
        else:
            self.dataset = self.dataset.shuffle(seed=seed)

        self._current_tokens = []  # Buffer for incomplete sequences
        self._examples_processed = 0

    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize a single text string."""
        if not text or len(text.strip()) == 0:
            return []
        return self.tokenizer.encode(text).ids

    def _get_sequences_from_tokens(self, tokens: List[int]) -> Iterator[tuple]:
        """
        Generate (input, target) sequences from token list.

        Yields:
            Tuple of (input_seq, target_seq)
        """
        for i in range(0, len(tokens) - self.seq_len, self.stride):
            input_seq = tokens[i : i + self.seq_len]
            target_seq = tokens[i + 1 : i + self.seq_len + 1]

            if len(input_seq) == self.seq_len and len(target_seq) == self.seq_len:
                yield input_seq, target_seq

    def _text_iterator(self) -> Iterator[str]:
        """Iterate over text examples from dataset."""
        for i, example in enumerate(self.dataset):
            if self.max_examples and i >= self.max_examples:
                break

            text = example[self.text_column]
            if text and len(text.strip()) > 0:
                yield text
                self._examples_processed += 1

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Iterate over batches of data.

        Yields:
            Dictionary with 'input_ids' and 'labels' as JAX arrays
        """
        batch_inputs = []
        batch_targets = []

        # Process texts one at a time
        for text in self._text_iterator():
            # Tokenize on-the-fly
            tokens = self._tokenize_text(text)

            # Add to buffer
            self._current_tokens.extend(tokens)

            # Generate sequences from buffer
            while len(self._current_tokens) >= self.seq_len + 1:
                # Extract one sequence
                input_seq = self._current_tokens[: self.seq_len]
                target_seq = self._current_tokens[1 : self.seq_len + 1]

                batch_inputs.append(input_seq)
                batch_targets.append(target_seq)

                # Remove processed tokens (with stride)
                self._current_tokens = self._current_tokens[self.stride :]

                # Yield batch when full
                if len(batch_inputs) >= self.batch_size:
                    yield {
                        "input_ids": jnp.array(batch_inputs, dtype=jnp.int32),
                        "labels": jnp.array(batch_targets, dtype=jnp.int32),
                    }
                    batch_inputs = []
                    batch_targets = []

        # Yield final partial batch if any
        if len(batch_inputs) > 0:
            yield {
                "input_ids": jnp.array(batch_inputs, dtype=jnp.int32),
                "labels": jnp.array(batch_targets, dtype=jnp.int32),
            }

    def get_epoch_iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Get an iterator for one epoch of training.

        Returns:
            Iterator over batches
        """
        self._current_tokens = []
        self._examples_processed = 0
        return iter(self)


class MemoryEfficientDataLoader:
    """
    Alternative data loader that pre-computes sequences but loads them in chunks.

    Use this when:
    - Dataset fits in memory but you want to avoid loading all at once
    - You need deterministic ordering
    - You want faster iteration than full streaming
    """

    def __init__(
        self,
        dataset_id: str,
        tokenizer: Union[Tokenizer, str],
        seq_len: int = 128,
        batch_size: int = 32,
        text_column: str = "text",
        dataset_config: Optional[str] = None,
        split: str = "train",
        max_examples: Optional[int] = None,
        stride: Optional[int] = None,
        chunk_size: int = 10000,  # Load this many sequences at a time
        seed: int = 42,
    ):
        """
        Initialize memory-efficient data loader.

        Args:
            chunk_size: Number of sequences to keep in memory at once
        """
        self.dataset_id = dataset_id
        self.tokenizer = (
            tokenizer
            if isinstance(tokenizer, Tokenizer)
            else Tokenizer.from_file(tokenizer)
        )
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.text_column = text_column
        self.stride = stride if stride else seq_len
        self.chunk_size = chunk_size
        self.seed = seed

        logger.info("Initializing MemoryEfficientDataLoader")
        logger.info(f"  Chunk size: {chunk_size} sequences")
        logger.info(
            f"  Memory usage: ~{chunk_size * seq_len * 4 / 1024 / 1024:.1f} MB per chunk"
        )

        # Load dataset (not streaming, but we'll process in chunks)
        self.dataset = load_dataset(
            dataset_id, dataset_config, split=split, streaming=False
        )

        if max_examples:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )

        self.dataset = self.dataset.shuffle(seed=seed)

        # Pre-compute total number of sequences (without loading all data)
        self._estimate_sequence_count()

    def _estimate_sequence_count(self):
        """Estimate total number of sequences."""
        # Sample first few examples to estimate
        sample_size = min(100, len(self.dataset))
        total_tokens = 0

        for i in range(sample_size):
            text = self.dataset[i][self.text_column]
            tokens = self.tokenizer.encode(text).ids
            total_tokens += len(tokens)

        avg_tokens_per_example = total_tokens / sample_size
        estimated_total_tokens = avg_tokens_per_example * len(self.dataset)
        self.estimated_sequences = int(estimated_total_tokens / self.stride)

        logger.info(f"  Estimated sequences: ~{self.estimated_sequences:,}")

    def _process_chunk(self, start_idx: int, end_idx: int) -> tuple:
        """Process a chunk of data and return sequences."""
        chunk_inputs = []
        chunk_targets = []

        tokens_buffer = []
        for i in range(start_idx, min(end_idx, len(self.dataset))):
            text = self.dataset[i][self.text_column]
            tokens = self.tokenizer.encode(text).ids
            tokens_buffer.extend(tokens)

            # Generate sequences from buffer
            while len(tokens_buffer) >= self.seq_len + 1:
                input_seq = tokens_buffer[: self.seq_len]
                target_seq = tokens_buffer[1 : self.seq_len + 1]

                chunk_inputs.append(input_seq)
                chunk_targets.append(target_seq)

                tokens_buffer = tokens_buffer[self.stride :]

                # Stop if we have enough sequences for this chunk
                if len(chunk_inputs) >= self.chunk_size:
                    return chunk_inputs, chunk_targets, tokens_buffer

        return chunk_inputs, chunk_targets, tokens_buffer

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Iterate over batches, loading data in chunks."""
        example_idx = 0
        examples_per_chunk = self.chunk_size // (self.seq_len // self.stride) + 1
        tokens_buffer = []

        while example_idx < len(self.dataset):
            # Process one chunk
            chunk_inputs, chunk_targets, tokens_buffer = self._process_chunk(
                example_idx, example_idx + examples_per_chunk
            )

            example_idx += examples_per_chunk

            # Yield batches from chunk
            for i in range(0, len(chunk_inputs), self.batch_size):
                batch_inputs = chunk_inputs[i : i + self.batch_size]
                batch_targets = chunk_targets[i : i + self.batch_size]

                if len(batch_inputs) > 0:
                    yield {
                        "input_ids": jnp.array(batch_inputs, dtype=jnp.int32),
                        "labels": jnp.array(batch_targets, dtype=jnp.int32),
                    }


def estimate_memory_usage(
    num_sequences: int, seq_len: int, batch_size: int, dtype: str = "int32"
) -> Dict[str, float]:
    """
    Estimate memory usage for different loading strategies.

    Returns:
        Dictionary with memory estimates in MB
    """
    bytes_per_element = 4 if dtype == "int32" else 2

    # Full load: all sequences in memory
    full_load_mb = (num_sequences * seq_len * 2 * bytes_per_element) / (1024**2)

    # Streaming: only one batch in memory
    streaming_mb = (batch_size * seq_len * 2 * bytes_per_element) / (1024**2)

    # Chunk load: configurable chunk size
    chunk_sizes = [1000, 5000, 10000, 50000]
    chunk_mbs = {
        f"chunk_{size}": (size * seq_len * 2 * bytes_per_element) / (1024**2)
        for size in chunk_sizes
    }

    result = {
        "full_load_mb": full_load_mb,
        "streaming_mb": streaming_mb,
        "memory_savings": f"{(1 - streaming_mb / full_load_mb) * 100:.1f}%",
        **chunk_mbs,
    }

    return result


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Memory-Efficient Data Loading Example")
    print("=" * 70)

    # Estimate memory savings
    print("\nMemory Usage Estimates:")
    print("  100K sequences, seq_len=128, batch_size=32")
    estimates = estimate_memory_usage(100000, 128, 32)
    print(f"  Full load:    {estimates['full_load_mb']:.1f} MB")
    print(f"  Streaming:    {estimates['streaming_mb']:.1f} MB")
    print(f"  Savings:      {estimates['memory_savings']}")
    print(f"  Chunk (10K):  {estimates['chunk_10000']:.1f} MB")
