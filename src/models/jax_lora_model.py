# Install dependencies: pip install jax flax optax
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

class LoRALinear(nn.Module):
    features: int
    rank: int = 8

    @nn.compact
    def __call__(self, x):
        w = self.param('kernel', nn.initializers.xavier_uniform(), (x.shape[-1], self.features))
        b = self.param('bias', nn.initializers.zeros, (self.features,))
        lora_a = self.param('lora_a', nn.initializers.normal(0.01), (x.shape[-1], self.rank))
        lora_b = self.param('lora_b', nn.initializers.zeros, (self.rank, self.features))
        effective_w = w + lora_a @ lora_b
        return jnp.dot(x, effective_w) + b

class MultiHeadAttention(nn.Module):
    num_heads: int
    d_model: int

    @nn.compact
    def __call__(self, query, key, value):
        d_k = self.d_model // self.num_heads
        q_proj = LoRALinear(self.d_model)(query)
        k_proj = LoRALinear(self.d_model)(key)
        v_proj = LoRALinear(self.d_model)(value)
        q = q_proj.reshape(q_proj.shape[:-1] + (self.num_heads, d_k)).transpose(0, 2, 1, 3)
        k = k_proj.reshape(k_proj.shape[:-1] + (self.num_heads, d_k)).transpose(0, 2, 1, 3)
        v = v_proj.reshape(v_proj.shape[:-1] + (self.num_heads, d_k)).transpose(0, 2, 1, 3)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(d_k)
        attn_weights = nn.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(attn_output.shape[0], -1, self.d_model)
        output = LoRALinear(self.d_model)(attn_output)
        return output

class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int

    @nn.compact
    def __call__(self, x):
        attn_out = MultiHeadAttention(self.num_heads, self.d_model)(x, x, x)
        x = x + attn_out
        x = nn.LayerNorm()(x)
        ff_out = LoRALinear(self.d_ff)(x)
        ff_out = nn.relu(ff_out)
        ff_out = LoRALinear(self.d_model)(ff_out)
        x = x + ff_out
        x = nn.LayerNorm()(x)
        return x

class SmallModel(nn.Module):
    vocab_size: int
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 128
    max_len: int = 100

    @nn.compact
    def __call__(self, x):
        embed = nn.Embed(self.vocab_size, self.d_model)(x)
        pos = jnp.arange(x.shape[1])
        pos_embed = nn.Embed(self.max_len, self.d_model)(pos)
        x = embed + pos_embed
        for _ in range(self.num_layers):
            x = TransformerBlock(self.d_model, self.num_heads, self.d_ff)(x)
        x = LoRALinear(self.vocab_size)(x)
        return x

def loss_fn(params, model, x, y):
    logits = model.apply(params, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(loss)

def train_model(model, vocab_size, seq_len=10, batch_size=32, num_steps=100):
    key = jax.random.PRNGKey(42)
    # Dummy data for training
    x = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    y = jax.random.randint(jax.random.split(key)[0], (batch_size, seq_len), 0, vocab_size)
    params = model.init(key, x)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(num_steps):
        params, opt_state, loss = step(params, opt_state, x, y)
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
    return params

def predict_next(model, params, input_seq):
    logits = model.apply(params, input_seq)
    next_token = jnp.argmax(logits[:, -1, :], axis=-1)
    return next_token