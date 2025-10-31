from typing import Callable

import jax.numpy as jnp
from flax import linen as nn

from ..config.config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - more efficient than LayerNorm"""

    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        # RMS normalization
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.epsilon)
        return (x / rms) * scale


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) - better than learned positional embeddings"""

    dim: int
    max_seq_len: int = 2048
    base: int = 10000

    def setup(self):
        # Precompute frequencies
        inv_freq = 1.0 / (
            self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim)
        )
        self.inv_freq = inv_freq

    def __call__(self, seq_len):
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, self.inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return jnp.cos(emb), jnp.sin(emb)


def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary positional embeddings to query and key tensors"""
    # x shape: [batch, heads, seq_len, dim]
    # cos, sin shape: [seq_len, dim]
    cos = cos[None, None, :, :]  # [1, 1, seq_len, dim]
    sin = sin[None, None, :, :]  # [1, 1, seq_len, dim]

    # Split into even and odd dimensions
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Apply rotation
    rotated_x1 = x1 * cos[..., 0::2] - x2 * sin[..., 1::2]
    rotated_x2 = x1 * sin[..., 0::2] + x2 * cos[..., 1::2]

    # Interleave back
    rotated = jnp.stack([rotated_x1, rotated_x2], axis=-1)
    return rotated.reshape(x.shape)


class LoRALinear(nn.Module):
    features: int
    rank: int = 8
    use_lora: bool = True

    @nn.compact
    def __call__(self, x):
        w = self.param(
            "kernel", nn.initializers.xavier_uniform(), (x.shape[-1], self.features)
        )
        b = self.param("bias", nn.initializers.zeros, (self.features,))
        if self.use_lora:
            lora_a = self.param(
                "lora_a", nn.initializers.normal(0.01), (x.shape[-1], self.rank)
            )
            lora_b = self.param(
                "lora_b", nn.initializers.zeros, (self.rank, self.features)
            )
            effective_w = w + lora_a @ lora_b
        else:
            effective_w = w
        return jnp.dot(x, effective_w) + b


class SwiGLU(nn.Module):
    """SwiGLU activation function - more powerful than GELU"""

    d_ff: int

    @nn.compact
    def __call__(self, x):
        gate = LoRALinear(self.d_ff, use_lora=False)(x)
        value = LoRALinear(self.d_ff, use_lora=False)(x)
        return nn.silu(gate) * value


class MultiHeadAttention(nn.Module):
    num_heads: int
    d_model: int
    dropout_rate: float = 0.1
    causal: bool = True
    use_rope: bool = True

    @nn.compact
    def __call__(self, query, key, value, mask=None, deterministic=True):
        d_k = self.d_model // self.num_heads

        # Project Q, K, V
        q_proj = LoRALinear(self.d_model, use_lora=False)(query)
        k_proj = LoRALinear(self.d_model, use_lora=False)(key)
        v_proj = LoRALinear(self.d_model, use_lora=False)(value)

        # Reshape for multi-head attention: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        q = q_proj.reshape(q_proj.shape[:-1] + (self.num_heads, d_k)).transpose(
            0, 2, 1, 3
        )
        k = k_proj.reshape(k_proj.shape[:-1] + (self.num_heads, d_k)).transpose(
            0, 2, 1, 3
        )
        v = v_proj.reshape(v_proj.shape[:-1] + (self.num_heads, d_k)).transpose(
            0, 2, 1, 3
        )

        # Apply RoPE if enabled
        if self.use_rope:
            rope = RotaryEmbedding(dim=d_k)
            cos, sin = rope(query.shape[1])
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        # Scaled dot-product attention
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(d_k)

        # Apply causal mask
        if self.causal:
            seq_len = q.shape[-2]
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            scores = jnp.where(causal_mask, scores, -1e10)

        if mask is not None:
            scores = jnp.where(mask, scores, -1e10)

        attn_weights = nn.softmax(scores, axis=-1)
        attn_weights = nn.Dropout(self.dropout_rate)(
            attn_weights, deterministic=deterministic
        )

        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, v)

        # Reshape back: [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            attn_output.shape[0], -1, self.d_model
        )

        # Output projection
        output = LoRALinear(self.d_model, use_lora=False)(attn_output)
        return output


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    activation: Callable = nn.gelu
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    use_rope: bool = True

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # Pre-normalization architecture (more stable training)
        norm_layer = RMSNorm() if self.use_rmsnorm else nn.LayerNorm()

        # Attention block with residual connection
        residual = x
        x = norm_layer(x)
        attn_out = MultiHeadAttention(
            self.num_heads, self.d_model, self.dropout_rate, use_rope=self.use_rope
        )(x, x, x, mask, deterministic)
        x = residual + attn_out

        # Feed-forward block with residual connection
        residual = x
        x = norm_layer(x)

        if self.use_swiglu:
            # SwiGLU activation
            ff_out = SwiGLU(self.d_ff)(x)
            ff_out = nn.Dropout(self.dropout_rate)(ff_out, deterministic=deterministic)
            ff_out = LoRALinear(self.d_model, use_lora=False)(ff_out)
        else:
            # Standard feed-forward
            ff_out = LoRALinear(self.d_ff, use_lora=False)(x)
            ff_out = self.activation(ff_out)
            ff_out = nn.Dropout(self.dropout_rate)(ff_out, deterministic=deterministic)
            ff_out = LoRALinear(self.d_model, use_lora=False)(ff_out)

        x = residual + ff_out

        return x


class ProductionTransformer(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # Token embeddings
        embed = nn.Embed(self.config.vocab_size, self.config.d_model)(x)

        # Add positional embeddings only if not using RoPE
        use_rope = getattr(self.config, "use_rope", True)
        if not use_rope:
            pos = jnp.arange(x.shape[1])
            pos_embed = nn.Embed(self.config.max_len, self.config.d_model)(pos)
            x = embed + pos_embed
        else:
            x = embed

        x = nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)

        # Transformer blocks
        use_rmsnorm = getattr(self.config, "use_rmsnorm", True)
        use_swiglu = getattr(self.config, "use_swiglu", True)

        for _ in range(self.config.num_layers):
            x = TransformerBlock(
                self.config.d_model,
                self.config.num_heads,
                self.config.d_ff,
                self.config.dropout_rate,
                getattr(nn, self.config.activation),
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                use_rope=use_rope,
            )(x, mask, deterministic)

        # Final normalization
        if use_rmsnorm:
            x = RMSNorm()(x)
        else:
            x = nn.LayerNorm()(x)

        # Output projection
        x = LoRALinear(self.config.vocab_size, use_lora=self.config.use_lora)(x)
        return x

    def encode(self, x, mask=None, deterministic=True):
        # Get embeddings before final linear
        embed = nn.Embed(self.config.vocab_size, self.config.d_model)(x)

        use_rope = getattr(self.config, "use_rope", True)
        if not use_rope:
            pos = jnp.arange(x.shape[1])
            pos_embed = nn.Embed(self.config.max_len, self.config.d_model)(pos)
            x = embed + pos_embed
        else:
            x = embed

        x = nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)

        use_rmsnorm = getattr(self.config, "use_rmsnorm", True)
        use_swiglu = getattr(self.config, "use_swiglu", True)

        for _ in range(self.config.num_layers):
            x = TransformerBlock(
                self.config.d_model,
                self.config.num_heads,
                self.config.d_ff,
                self.config.dropout_rate,
                getattr(nn, self.config.activation),
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                use_rope=use_rope,
            )(x, mask, deterministic)

        return x  # embeddings
