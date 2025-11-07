"""
Flash Attention implementation using Kvax for JAX.
Provides efficient attention with automatic fallback for CPU/GPU/TPU compatibility.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

# Try to import kvax, fallback to standard attention if not available
try:
    from kvax.ops import create_attention_mask, flash_attention
    from kvax.utils import (
        PADDING_SEGMENT_ID,
        FlashAttentionParamsConfig,
        attention_specs,
        get_default_flash_attention_params,
    )

    KVAX_AVAILABLE = True
except ImportError:
    KVAX_AVAILABLE = False
    print("Warning: kvax not available. Using standard attention implementation.")


def detect_device_type() -> str:
    """
    Detect the type of device JAX is running on.

    Returns:
        str: 'cpu', 'gpu', or 'tpu'
    """
    try:
        devices = jax.devices()
        if len(devices) == 0:
            return "cpu"

        device = devices[0]
        platform = device.platform.lower()

        if "tpu" in platform:
            return "tpu"
        elif "gpu" in platform or "cuda" in platform:
            return "gpu"
        else:
            return "cpu"
    except Exception:
        return "cpu"


def is_flash_attention_supported() -> bool:
    """
    Check if flash attention is supported on the current device.

    Returns:
        bool: True if Kvax is available and device supports it
    """
    if not KVAX_AVAILABLE:
        return False

    device_type = detect_device_type()
    # Kvax flash attention is optimized for GPU/TPU, but can work on CPU
    # For production, we'll enable it for all devices
    return True


def create_segment_ids(
    batch_size: int, seq_len: int, padding_mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Create segment IDs for attention mask.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        padding_mask: Optional padding mask [batch_size, seq_len] where 0 = padding

    Returns:
        Segment IDs array [batch_size, seq_len]
    """
    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    if padding_mask is not None:
        # Mark padding tokens with PADDING_SEGMENT_ID (-1)
        if KVAX_AVAILABLE:
            segment_ids = jnp.where(padding_mask, segment_ids, PADDING_SEGMENT_ID)
        else:
            segment_ids = jnp.where(padding_mask, segment_ids, -1)

    return segment_ids


def create_positions(batch_size: int, seq_len: int) -> jnp.ndarray:
    """
    Create position indices for sequences.

    Args:
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Position indices [batch_size, seq_len]
    """
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    return positions


def kvax_flash_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    scale: float,
    causal: bool = True,
    padding_mask: Optional[jnp.ndarray] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
    query_specs: Optional[Tuple] = None,
    kv_specs: Optional[Tuple] = None,
) -> jnp.ndarray:
    """
    Perform flash attention using Kvax.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len, num_heads, head_dim]
        value: Value tensor [batch, seq_len, num_heads, head_dim]
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        causal: Whether to use causal masking
        padding_mask: Optional padding mask [batch, seq_len]
        mesh: Optional JAX mesh for distributed execution
        query_specs: Optional sharding specs for query
        kv_specs: Optional sharding specs for key/value

    Returns:
        Attention output [batch, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len = query.shape[0], query.shape[1]

    # Create positions and segment IDs
    positions = create_positions(batch_size, seq_len)
    segment_ids = create_segment_ids(batch_size, seq_len, padding_mask)

    # Get default flash attention params
    fwd_params = get_default_flash_attention_params(backward=False)
    bwd_params = get_default_flash_attention_params(backward=True)

    # Setup mesh and sharding specs
    if mesh is None:
        devices = jax.devices()
        mesh = jax.sharding.Mesh(devices, ("data",))

    if query_specs is None:
        query_specs = ("data", None, None, None)
    if kv_specs is None:
        kv_specs = ("data", None, None, None)

    # Create attention mask and compute attention
    with mesh, attention_specs(query_specs=query_specs, kv_specs=kv_specs):
        # Create attention mask
        attn_mask = create_attention_mask(
            query_positions=positions,
            query_segment_ids=segment_ids,
            kv_positions=positions,
            kv_segment_ids=segment_ids,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            calc_bwd_mask=True,
            skip_pad_tokens=True,
        )

        # Perform flash attention
        output = flash_attention(
            query=query,
            key=key,
            value=value,
            query_positions=positions,
            query_segment_ids=segment_ids,
            kv_positions=positions,
            kv_segment_ids=segment_ids,
            mask=attn_mask,
            scale=scale,
            assume_sequential_positions=True,
        )

    return output


def standard_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    scale: float,
    causal: bool = True,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Standard scaled dot-product attention (fallback implementation).

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_heads, seq_len, head_dim]
        value: Value tensor [batch, num_heads, seq_len, head_dim]
        scale: Attention scale factor
        causal: Whether to use causal masking
        mask: Optional attention mask

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]
    """
    # Compute attention scores
    scores = jnp.matmul(query, key.transpose(0, 1, 3, 2)) * scale

    # Apply causal mask
    if causal:
        seq_len = query.shape[-2]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(causal_mask, scores, -1e10)

    # Apply optional mask
    if mask is not None:
        scores = jnp.where(mask, scores, -1e10)

    # Apply softmax
    attn_weights = nn.softmax(scores, axis=-1)

    # Apply attention to values
    output = jnp.matmul(attn_weights, value)

    return output


def adaptive_flash_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    scale: float,
    causal: bool = True,
    padding_mask: Optional[jnp.ndarray] = None,
    use_flash_attention: bool = True,
    mesh: Optional[jax.sharding.Mesh] = None,
    query_specs: Optional[Tuple] = None,
    kv_specs: Optional[Tuple] = None,
) -> jnp.ndarray:
    """
    Adaptive attention that uses Kvax flash attention when available,
    otherwise falls back to standard attention.

    This function automatically handles the shape conversion and device compatibility.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
               or [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, num_heads, seq_len, head_dim]
             or [batch, seq_len, num_heads, head_dim]
        value: Value tensor [batch, num_heads, seq_len, head_dim]
               or [batch, seq_len, num_heads, head_dim]
        scale: Attention scale factor
        causal: Whether to use causal masking
        padding_mask: Optional padding mask [batch, seq_len]
        use_flash_attention: Whether to try using flash attention
        mesh: Optional JAX mesh for distributed execution
        query_specs: Optional sharding specs for query
        kv_specs: Optional sharding specs for key/value

    Returns:
        Attention output in same format as input
    """
    # Detect input shape format
    # If shape is [batch, num_heads, seq_len, head_dim], transpose to kvax format
    needs_transpose = query.shape[1] < query.shape[2]  # Heuristic: heads < seq_len

    if needs_transpose:
        # Convert from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, num_heads, head_dim]
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

    # Try flash attention if available and enabled
    if use_flash_attention and is_flash_attention_supported():
        try:
            output = kvax_flash_attention(
                query=query,
                key=key,
                value=value,
                scale=scale,
                causal=causal,
                padding_mask=padding_mask,
                mesh=mesh,
                query_specs=query_specs,
                kv_specs=kv_specs,
            )
        except Exception as e:
            # Fallback to standard attention on error
            print(
                f"Warning: Flash attention failed ({e}), falling back to standard attention"
            )
            if needs_transpose:
                # Convert back for standard attention
                query = query.transpose(0, 2, 1, 3)
                key = key.transpose(0, 2, 1, 3)
                value = value.transpose(0, 2, 1, 3)
            output = standard_attention(query, key, value, scale, causal)
            return output
    else:
        # Use standard attention
        if needs_transpose:
            # Convert back for standard attention
            query = query.transpose(0, 2, 1, 3)
            key = key.transpose(0, 2, 1, 3)
            value = value.transpose(0, 2, 1, 3)
        output = standard_attention(query, key, value, scale, causal)
        return output

    # Convert back to original format if needed
    if needs_transpose:
        output = output.transpose(0, 2, 1, 3)

    return output


def get_flash_attention_config() -> dict:
    """
    Get information about flash attention availability and configuration.

    Returns:
        Dictionary with flash attention configuration
    """
    device_type = detect_device_type()
    supported = is_flash_attention_supported()

    config = {
        "kvax_available": KVAX_AVAILABLE,
        "device_type": device_type,
        "flash_attention_supported": supported,
        "device_info": str(jax.devices()[0]) if jax.devices() else "No devices",
    }

    if KVAX_AVAILABLE and supported:
        try:
            fwd_params = get_default_flash_attention_params(backward=False)
            bwd_params = get_default_flash_attention_params(backward=True)
            config["forward_params"] = {
                "query_block_size": fwd_params.query_block_size,
                "kv_block_size": fwd_params.kv_block_size,
                "num_warps": fwd_params.num_warps,
                "num_stages": fwd_params.num_stages,
            }
            config["backward_params"] = {
                "query_block_size": bwd_params.query_block_size,
                "kv_block_size": bwd_params.kv_block_size,
                "num_warps": bwd_params.num_warps,
                "num_stages": bwd_params.num_stages,
            }
        except Exception as e:
            config["params_error"] = str(e)

    return config
