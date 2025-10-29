"""
HuggingFace Dataset Loader Utility

Supports loading datasets from HuggingFace Hub with:
- Dataset ID specification
- Column/field selection
- Automatic tokenization
- Sequence creation for language modeling
"""

import jax.numpy as jnp
from datasets import load_dataset
from typing import Optional, List, Union
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import logging

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
        streaming: bool = False,
        trust_remote_code: bool = False
    ):
        """
        Initialize dataset loader.
        
        Args:
            dataset_id: HuggingFace dataset ID (e.g., "wikitext", "openwebtext")
            text_column: Column name containing text data
            dataset_config: Dataset configuration/subset (e.g., "wikitext-2-raw-v1")
            split: Dataset split ("train", "validation", "test")
            streaming: Whether to stream the dataset
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
        
        # Load dataset
        self.dataset = load_dataset(
            dataset_id,
            dataset_config,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code
        )
        
        # Verify text column exists
        if not streaming:
            if text_column not in self.dataset.column_names:
                raise ValueError(
                    f"Column '{text_column}' not found in dataset. "
                    f"Available columns: {self.dataset.column_names}"
                )
        
        logger.info(f"Dataset loaded successfully!")
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
            dataset_slice = self.dataset if max_examples is None else self.dataset.select(range(min(max_examples, len(self.dataset))))
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
        max_examples: Optional[int] = 10000
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
            special_tokens=special_tokens
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
        max_examples: Optional[int] = None
    ) -> tuple:
        """
        Prepare input and target sequences for language modeling.
        
        Args:
            tokenizer: Tokenizer object or path to tokenizer file
            seq_len: Sequence length
            stride: Stride for overlapping sequences (default: seq_len, no overlap)
            max_examples: Maximum number of examples to process (None for all)
        
        Returns:
            Tuple of (inputs, targets) as lists
        """
        # Load tokenizer if path provided
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer.from_file(tokenizer)
        
        if stride is None:
            stride = seq_len
        
        logger.info(f"Preparing sequences (seq_len={seq_len}, stride={stride})")
        
        # Get text data
        texts = self.get_text_data(max_examples=max_examples)
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text).ids
            all_tokens.extend(tokens)
        
        logger.info(f"Total tokens: {len(all_tokens)}")
        
        # Create sequences
        inputs = []
        targets = []
        
        for i in range(0, len(all_tokens) - seq_len, stride):
            input_seq = all_tokens[i:i + seq_len]
            target_seq = all_tokens[i + 1:i + seq_len + 1]
            
            if len(input_seq) == seq_len and len(target_seq) == seq_len:
                inputs.append(input_seq)
                targets.append(target_seq)
        
        logger.info(f"Created {len(inputs)} sequences")
        
        return inputs, targets
    
    def create_batch_iterator(
        self,
        tokenizer: Union[Tokenizer, str],
        batch_size: int,
        seq_len: int = 128,
        stride: Optional[int] = None,
        max_examples: Optional[int] = None,
        shuffle: bool = True
    ):
        """
        Create a batch iterator for training.
        
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
        # Prepare sequences
        inputs, targets = self.prepare_sequences(
            tokenizer, seq_len, stride, max_examples
        )
        
        # Convert to JAX arrays
        inputs = jnp.array(inputs, dtype=jnp.int32)
        targets = jnp.array(targets, dtype=jnp.int32)
        
        # Shuffle if requested
        if shuffle:
            import jax
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


def list_dataset_columns(dataset_id: str, dataset_config: Optional[str] = None, split: str = "train") -> List[str]:
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
    print(f"\nAvailable columns:")
    for col in columns:
        print(f"  - {col}")
    
    return columns


if __name__ == "__main__":
    # Example 1: Load wikitext dataset
    print("="*70)
    print("Example 1: WikiText-2 Dataset")
    print("="*70)
    
    loader = HFDatasetLoader(
        dataset_id="iohadrubin/wikitext-103-raw-v1",
        text_column="text",
        split="train"
    )
    
    # Train tokenizer
    tokenizer = loader.train_tokenizer(vocab_size=5000, max_examples=1000)
    
    # Prepare sequences
    inputs, targets = loader.prepare_sequences(tokenizer, seq_len=64, max_examples=100)
    print(f"\nPrepared {len(inputs)} sequences")
    print(f"Input shape: {len(inputs[0])} tokens")
    print(f"Target shape: {len(targets[0])} tokens")
    
    # Example 2: List available columns
    print("\n" + "="*70)
    print("Example 2: List Dataset Columns")
    print("="*70)
    
    list_dataset_columns("iohadrubin/wikitext-103-raw-v1", split="train")
