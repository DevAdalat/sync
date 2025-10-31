import json
from typing import Any, Dict

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training import checkpoints
from model import ProductionTransformer
from tokenizers import Tokenizer

from config import ModelConfig
from utils import detokenize_text, load_tokenizer, tokenize_text


class TransformerAPI:
    def __init__(
        self, model: ProductionTransformer, params: FrozenDict, tokenizer: Tokenizer
    ):
        self.model = model
        self.params = params
        self.tokenizer = tokenizer

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path) as f:
            config_dict = json.load(f)
        model_config = ModelConfig(**config_dict["model"])
        model = ProductionTransformer(model_config)
        params = checkpoints.restore_checkpoint(
            config_dict["checkpoint_path"], target=None
        )
        tokenizer = load_tokenizer(
            config_dict.get("tokenizer_path"), config_dict.get("pretrained_tokenizer")
        )
        return cls(model, params, tokenizer)

    def save_model(self, path: str, config: Dict[str, Any]):
        checkpoints.save_checkpoint(path, self.params, step=0)
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f)

    def predict(self, text: str, max_len: int = 50, temperature: float = 1.0) -> str:
        tokens = tokenize_text(text, self.tokenizer, self.model.config.max_len)
        input_ids = jnp.array([tokens])
        for _ in range(max_len):
            logits = self.model.apply(self.params, input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            next_token = jax.random.categorical(
                jax.random.PRNGKey(0), next_token_logits
            )
            input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
            eos_id = (
                self.tokenizer.token_to_id("[SEP]")
                or self.tokenizer.token_to_id("</s>")
                or 0
            )
            if next_token == eos_id:
                break
        generated_tokens = input_ids[0].tolist()
        return detokenize_text(generated_tokens, self.tokenizer)

    def encode(self, text: str) -> jnp.ndarray:
        tokens = tokenize_text(text, self.tokenizer, self.model.config.max_len)
        input_ids = jnp.array([tokens])
        embeddings = self.model.encode(self.params, input_ids)
        return embeddings

    def fine_tune(self, dataset, lora_rank: int = 8):
        # Placeholder for LoRA fine-tuning
        pass

    def evaluate(self, dataset) -> Dict[str, float]:
        # Placeholder evaluation
        return {"perplexity": 10.0, "accuracy": 0.8}
