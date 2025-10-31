"""
Memory-efficient version of prepare_sequences that avoids the 34x memory blowup.

Key optimizations:
1. Streaming tokenization (never load all texts at once)
2. Generator-based sequence creation (no duplicate storage)
3. Direct JAX array creation (no intermediate lists)
4. Memory cleanup at each step
"""

import jax
import jax.numpy as jnp
from datasets import load_dataset
from typing import Optional, List, Union, Iterator, Generator
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import numpy as np
import gc

logger = logging.getLogger(__name__)


class MemoryEfficientDatasetLoader:
    """
    Memory-efficient dataset loader that avoids the massive memory blowup.
    
    Instead of loading everything into memory, this uses:
    - Streaming tokenization
    - Generator-based sequence creation
    - Direct JAX array construction
    - Aggressive memory cleanup
    """
    
    def __init__(
        self,
        dataset_id: str,
        text_column: str = "text",
        dataset_config: Optional[str] = None,
        split: str = "train",
        streaming: bool = True,
        trust_remote_code: bool = False
    ):
        self.dataset_id = dataset_id
        self.text_column = text_column
        self.dataset_config = dataset_config
        self.split = split
        self.streaming = streaming
        self.trust_remote_code = trust_remote_code
        
        logger.info(f"Loading dataset: {dataset_id}")
        self.dataset = load_dataset(
            dataset_id,
            dataset_config,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code
        )
        logger.info(f"Dataset loaded successfully!")
    
    def stream_tokens(
        self,
        tokenizer: Union[Tokenizer, str],
        max_examples: Optional[int] = None
    ) -> Generator[int, None, None]:
        """
        Stream tokens one by one (memory efficient).
        
        Args:
            tokenizer: Tokenizer object or path to tokenizer file
            max_examples: Maximum number of examples to process
        
        Yields:
            Individual tokens (integers)
        """
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer.from_file(tokenizer)
        
        logger.info(f"Streaming tokens (max_examples={max_examples})")
        
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
                gc.collect()  # Force garbage collection
    
    def prepare_sequences_memory_efficient(
        self,
        tokenizer: Union[Tokenizer, str],
        seq_len: int = 128,
        stride: Optional[int] = None,
        max_examples: Optional[int] = None,
        chunk_size: int = 10000,
        return_jax: bool = True
    ):
        """
        Memory-efficient sequence preparation that avoids the 34x memory blowup.
        
        Args:
            tokenizer: Tokenizer object or path to tokenizer file
            seq_len: Sequence length
            stride: Stride for overlapping sequences (default: seq_len)
            max_examples: Maximum number of examples to process
            chunk_size: Number of tokens to process at once (memory vs speed tradeoff)
            return_jax: Return JAX arrays (True) or numpy arrays (False)
        
        Returns:
            Tuple of (inputs, targets) as JAX or numpy arrays
        """
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer.from_file(tokenizer)
        
        if stride is None:
            stride = seq_len
        
        logger.info(f"Memory-efficient sequence preparation (seq_len={seq_len}, stride={stride})")
        logger.info(f"Chunk size: {chunk_size} tokens")
        
        # First pass: count total tokens (streaming)
        logger.info("Counting total tokens...")
        token_stream = self.stream_tokens(tokenizer, max_examples)
        total_tokens = sum(1 for _ in token_stream)
        logger.info(f"Total tokens: {total_tokens}")
        
        # Calculate number of sequences
        num_sequences = (total_tokens - seq_len) // stride + 1
        logger.info(f"Will create {num_sequences} sequences")
        
        # Pre-allocate arrays (this is the only memory we need!)
        if return_jax:
            inputs = jnp.zeros((num_sequences, seq_len), dtype=jnp.int32)
            targets = jnp.zeros((num_sequences, seq_len), dtype=jnp.int32)
        else:
            inputs = np.zeros((num_sequences, seq_len), dtype=np.int32)
            targets = np.zeros((num_sequences, seq_len), dtype=np.int32)
        
        # Second pass: fill arrays with sequences (streaming)
        logger.info("Creating sequences (streaming)...")
        token_stream = self.stream_tokens(tokenizer, max_examples)
        
        # Buffer for sliding window
        token_buffer = []
        seq_idx = 0
        
        for token in token_stream:
            token_buffer.append(token)
            
            # Keep buffer size manageable
            if len(token_buffer) > seq_len + stride:
                token_buffer = token_buffer[-(seq_len + stride):]
            
            # Create sequences when we have enough tokens
            while len(token_buffer) >= seq_len + 1:
                # Extract input and target sequences
                input_seq = token_buffer[:seq_len]
                target_seq = token_buffer[1:seq_len + 1]
                
                # Store in arrays
                if return_jax:
                    inputs = inputs.at[seq_idx].set(jnp.array(input_seq, dtype=jnp.int32))
                    targets = targets.at[seq_idx].set(jnp.array(target_seq, dtype=jnp.int32))
                else:
                    inputs[seq_idx] = input_seq
                    targets[seq_idx] = target_seq
                
                seq_idx += 1
                
                # Move window forward
                token_buffer = token_buffer[stride:]
                
                # Progress reporting
                if seq_idx % 1000 == 0 or seq_idx == num_sequences - 1:
                    percent_done = (seq_idx / num_sequences) * 100
                    logger.info(f"Progress: {percent_done:.1f}% ({seq_idx}/{num_sequences} sequences)")
                    sys.stdout.flush()  # Force real-time output
                    gc.collect()
        
        logger.info(f"Successfully created {seq_idx} sequences")
        
        # Trim arrays if we created fewer than expected
        if seq_idx < num_sequences:
            if return_jax:
                inputs = inputs[:seq_idx]
                targets = targets[:seq_idx]
            else:
                inputs = inputs[:seq_idx]
                targets = targets[:seq_idx]
        
        return inputs, targets
    
    def create_memory_efficient_batch_iterator(
        self,
        tokenizer: Union[Tokenizer, str],
        batch_size: int,
        seq_len: int = 128,
        stride: Optional[int] = None,
        max_examples: Optional[int] = None,
        shuffle: bool = True
    ):
        """
        Create a memory-efficient batch iterator that never loads everything at once.
        
        Args:
            tokenizer: Tokenizer object or path to tokenizer file
            batch_size: Batch size
            seq_len: Sequence length
            stride: Stride for overlapping sequences
            max_examples: Maximum number of examples to process
            shuffle: Whether to shuffle the data
        
        Yields:
            Batches of (inputs, targets) as JAX arrays
        """
        # Create sequences using memory-efficient method
        inputs, targets = self.prepare_sequences_memory_efficient(
            tokenizer, seq_len, stride, max_examples, return_jax=True
        )
        
        # Shuffle if requested
        if shuffle:
            rng = jax.random.PRNGKey(42)
            perm = jax.random.permutation(rng, len(inputs))
            inputs = inputs[perm]
            targets = targets[perm]
        
        # Yield batches
        num_batches = len(inputs) // batch_size
        logger.info(f"Creating batch iterator: {num_batches} batches of size {batch_size}")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            yield {
                "input_ids": inputs[start_idx:end_idx],
                "labels": targets[start_idx:end_idx]
            }


def compare_memory_usage():
    """
    Compare memory usage between original and memory-efficient methods.
    """
    import psutil
    import os
    
    def get_memory():
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
    
    print("="*70)
    print("MEMORY COMPARISON TEST")
    print("="*70)
    
    # Test parameters
    seq_len = 128
    stride = 64
    max_examples = 5000  # Smaller for testing
    
    # Original method (will use lots of memory)
    print("\n1. Testing original method...")
    from hf_dataset_loader import HFDatasetLoader
    
    original_loader = HFDatasetLoader(
        dataset_id="iohadrubin/wikitext-103-raw-v1",
        text_column="text",
        split="train",
        streaming=True
    )
    
    tokenizer = original_loader.train_tokenizer(vocab_size=5000, max_examples=1000)
    
    mem_before = get_memory()
    inputs_orig, targets_orig = original_loader.prepare_sequences(
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        max_examples=max_examples,
        use_gpu=False  # Disable GPU for fair comparison
    )
    mem_after_orig = get_memory()
    
    print(f"   Original method memory: {mem_after_orig:.2f} GB (+{mem_after_orig - mem_before:.2f} GB)")
    print(f"   Sequences created: {len(inputs_orig)}")
    
    # Clean up
    del inputs_orig, targets_orig
    import gc
    gc.collect()
    
    # Memory-efficient method
    print("\n2. Testing memory-efficient method...")
    efficient_loader = MemoryEfficientDatasetLoader(
        dataset_id="iohadrubin/wikitext-103-raw-v1",
        text_column="text",
        split="train",
        streaming=True
    )
    
    mem_before = get_memory()
    inputs_eff, targets_eff = efficient_loader.prepare_sequences_memory_efficient(
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        max_examples=max_examples,
        return_jax=True
    )
    mem_after_eff = get_memory()
    
    print(f"   Efficient method memory: {mem_after_eff:.2f} GB (+{mem_after_eff - mem_before:.2f} GB)")
    print(f"   Sequences created: {len(inputs_eff)}")
    
    # Comparison
    print(f"\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"Original method:     {mem_after_orig - mem_before:.2f} GB")
    print(f"Efficient method:     {mem_after_eff - mem_before:.2f} GB")
    print(f"Memory reduction:    {((mem_after_orig - mem_before) / (mem_after_eff - mem_before)):.1f}x")
    print(f"Same sequences:      {len(inputs_orig) == len(inputs_eff)}")
    
    return {
        'original_memory': mem_after_orig - mem_before,
        'efficient_memory': mem_after_eff - mem_before,
        'reduction_factor': (mem_after_orig - mem_before) / (mem_after_eff - mem_before)
    }


if __name__ == "__main__":
    compare_memory_usage()