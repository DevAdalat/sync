import jax

from api import TransformerAPI
from src.config.config import ModelConfig
from src.models.model import ProductionTransformer
from src.utils.utils import load_vocab

# Simple test
model_config = ModelConfig(
    vocab_size=100, d_model=32, num_heads=2, num_layers=1, d_ff=64, max_len=10
)
model = ProductionTransformer(model_config)

# Test init
x = jax.numpy.ones((1, 10), dtype=jax.numpy.int32)
params = model.init(jax.random.PRNGKey(0), x)
print("Model initialized successfully")

# Test API
vocab = load_vocab("")  # dummy
api = TransformerAPI(model, params, vocab)
# Since vocab is dummy, prediction may not work, but init is fine
print("API initialized successfully")

print("System test passed")
