import jax.numpy as jnp
from typing import List, Dict, Any, Optional
import logging
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level.upper()))

def load_tokenizer(path: Optional[str] = None, pretrained: Optional[str] = None) -> Tokenizer:
    """Load tokenizer from file or pretrained."""
    if path:
        return Tokenizer.from_file(path)
    elif pretrained:
        return Tokenizer.from_pretrained(pretrained)
    else:
        # Default to BERT base uncased
        return Tokenizer.from_pretrained("bert-base-uncased")

def tokenize_text(text: str, tokenizer: Tokenizer, max_len: int) -> List[int]:
    """Tokenize text using the tokenizer."""
    encoded = tokenizer.encode(text)
    return encoded.ids[:max_len]

def detokenize_text(tokens: List[int], tokenizer: Tokenizer) -> str:
    """Detokenize tokens to text."""
    return tokenizer.decode(tokens)

def get_vocab(tokenizer: Tokenizer) -> Dict[str, int]:
    """Get vocabulary from tokenizer."""
    return tokenizer.get_vocab(with_added_tokens=True)

def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return float(jnp.exp(loss))

def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> float:
    """Compute accuracy."""
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == labels))

# Deprecated, use get_vocab
def load_vocab(path: str) -> Dict[str, int]:
    """Load vocabulary (deprecated, use tokenizer)."""
    tokenizer = load_tokenizer(path)
    return get_vocab(tokenizer)

def save_vocab(vocab: Dict[str, int], path: str):
    """Save vocabulary (deprecated)."""
    pass  # Not needed with tokenizer