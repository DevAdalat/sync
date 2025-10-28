import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from typing import Optional, Callable
from config import ModelConfig

class LoRALinear(nn.Module):
    features: int
    rank: int = 8
    use_lora: bool = True

    @nn.compact
    def __call__(self, x):
        w = self.param('kernel', nn.initializers.xavier_uniform(), (x.shape[-1], self.features))
        b = self.param('bias', nn.initializers.zeros, (self.features,))
        if self.use_lora:
            lora_a = self.param('lora_a', nn.initializers.normal(0.01), (x.shape[-1], self.rank))
            lora_b = self.param('lora_b', nn.initializers.zeros, (self.rank, self.features))
            effective_w = w + lora_a @ lora_b
        else:
            effective_w = w
        return jnp.dot(x, effective_w) + b

class MultiHeadAttention(nn.Module):
    num_heads: int
    d_model: int
    dropout_rate: float = 0.1
    causal: bool = False

    @nn.compact
    def __call__(self, query, key, value, mask=None, deterministic=True):
        d_k = self.d_model // self.num_heads
        q_proj = LoRALinear(self.d_model)(query)
        k_proj = LoRALinear(self.d_model)(key)
        v_proj = LoRALinear(self.d_model)(value)
        q = q_proj.reshape(q_proj.shape[:-1] + (self.num_heads, d_k)).transpose(0, 2, 1, 3)
        k = k_proj.reshape(k_proj.shape[:-1] + (self.num_heads, d_k)).transpose(0, 2, 1, 3)
        v = v_proj.reshape(v_proj.shape[:-1] + (self.num_heads, d_k)).transpose(0, 2, 1, 3)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(d_k)
        if self.causal:
            seq_len = q.shape[-2]
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            scores = jnp.where(causal_mask, scores, -jnp.inf)
        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)
        attn_weights = nn.softmax(scores, axis=-1)
        attn_weights = nn.Dropout(self.dropout_rate)(attn_weights, deterministic=deterministic)
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(attn_output.shape[0], -1, self.d_model)
        output = LoRALinear(self.d_model)(attn_output)
        return output

class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        attn_out = MultiHeadAttention(self.num_heads, self.d_model, self.dropout_rate)(x, x, x, mask, deterministic)
        x = x + attn_out
        x = nn.LayerNorm()(x)
        ff_out = LoRALinear(self.d_ff)(x)
        ff_out = self.activation(ff_out)
        ff_out = nn.Dropout(self.dropout_rate)(ff_out, deterministic=deterministic)
        ff_out = LoRALinear(self.d_model)(ff_out)
        x = x + ff_out
        x = nn.LayerNorm()(x)
        return x

class ProductionTransformer(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        embed = nn.Embed(self.config.vocab_size, self.config.d_model)(x)
        pos = jnp.arange(x.shape[1])
        pos_embed = nn.Embed(self.config.max_len, self.config.d_model)(pos)
        x = embed + pos_embed
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)
        for _ in range(self.config.num_layers):
            x = TransformerBlock(self.config.d_model, self.config.num_heads, self.config.d_ff, self.config.dropout_rate, getattr(nn, self.config.activation))(x, mask, deterministic)
        x = LoRALinear(self.config.vocab_size, use_lora=self.config.use_lora)(x)
        return x

    def encode(self, x, mask=None, deterministic=True):
        # Get embeddings before final linear
        embed = nn.Embed(self.config.vocab_size, self.config.d_model)(x)
        pos = jnp.arange(x.shape[1])
        pos_embed = nn.Embed(self.config.max_len, self.config.d_model)(pos)
        x = embed + pos_embed
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)
        for _ in range(self.config.num_layers):
            x = TransformerBlock(self.config.d_model, self.config.num_heads, self.config.d_ff, self.config.dropout_rate, getattr(nn, self.config.activation))(x, mask, deterministic)
        return x  # embeddings