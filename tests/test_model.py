import pytest
import jax
import jax.numpy as jnp
from model import ProductionTransformer, LoRALinear
from config import ModelConfig

def test_lora_linear():
    model = LoRALinear(features=10, rank=5)
    x = jnp.ones((2, 8))
    params = model.init(jax.random.PRNGKey(0), x)
    output = model.apply(params, x)
    assert output.shape == (2, 10)

def test_production_transformer():
    config = ModelConfig(vocab_size=100, d_model=32, num_heads=2, num_layers=1, d_ff=64, max_len=50)
    model = ProductionTransformer(config)
    x = jnp.ones((1, 10), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), x)
    output = model.apply(params, x)
    assert output.shape == (1, 10, 100)