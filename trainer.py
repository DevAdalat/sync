import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import linen as nn
from datasets import load_dataset
from typing import Optional, Dict, Any
import logging
import os
from orbax import checkpoint as ocp
from config import TrainingConfig, DataConfig, ModelConfig
from model import ProductionTransformer
from utils import compute_perplexity, compute_accuracy

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config
        self.dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.split)

    def preprocess(self, examples):
        # Placeholder preprocessing
        return {"input_ids": examples[self.config.text_column][:self.config.max_length]}

    def get_batch(self, batch_size: int):
        # Simple batching
        batch = self.dataset.select(range(batch_size))
        return self.preprocess(batch)

class Trainer:
    def __init__(self, model_config: ModelConfig, train_config: TrainingConfig, data_config: Optional[DataConfig] = None):
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config
        self.model = ProductionTransformer(model_config)
        self.data_loader = DataLoader(data_config) if data_config else None
        self.state = None

    def create_train_state(self, rng):
        params = self.model.init(rng, jnp.ones((1, self.model_config.max_len), dtype=jnp.int32))
        tx = optax.adamw(self.train_config.learning_rate, weight_decay=self.train_config.weight_decay)
        if self.train_config.warmup_steps > 0:
            tx = optax.chain(
                optax.clip_by_global_norm(self.train_config.gradient_clip_norm),
                optax.warmup_cosine_decay_schedule(
                    init_value=0.0,
                    peak_value=self.train_config.learning_rate,
                    warmup_steps=self.train_config.warmup_steps,
                    decay_steps=self.train_config.max_steps or 10000
                )
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(self.train_config.gradient_clip_norm), tx)
        return train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    def loss_fn(self, params, batch):
        logits = self.model.apply(params, batch["input_ids"])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"])
        return jnp.mean(loss)

    def get_train_step(self):
        @jax.jit
        def train_step(state, batch):
            loss, grads = jax.value_and_grad(self.loss_fn)(state.params, batch)
            state = state.apply_gradients(grads=grads)
            return state, loss
        return train_step

    def fit(self, rng):
        if self.state is None:
            self.state = self.create_train_state(rng)
        train_step = self.get_train_step()
        for epoch in range(self.train_config.num_epochs):
            for step in range(self.train_config.max_steps or len(self.data_loader.dataset) // self.train_config.batch_size if self.data_loader else 100):
                batch = self.data_loader.get_batch(self.train_config.batch_size) if self.data_loader else {"input_ids": jnp.ones((self.train_config.batch_size, self.model_config.max_len), dtype=jnp.int32), "labels": jnp.ones((self.train_config.batch_size, self.model_config.max_len), dtype=jnp.int32)}
                self.state, loss = train_step(self.state, batch)
                if step % self.train_config.log_steps == 0:
                    logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
                if step % self.train_config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint_{step}")
            # Validation
            if epoch % self.train_config.eval_steps == 0:
                val_loss = self.validate()
                logger.info(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")

    def validate(self):
        # Placeholder validation
        batch = self.data_loader.get_batch(self.train_config.batch_size)
        loss = self.loss_fn(self.state.params, batch)
        return loss

    def save_checkpoint(self, path: str):
        if self.state is None:
            raise ValueError("No trained state to save")
        # Remove existing checkpoint directory if it exists
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path, self.state.params)

    def load_checkpoint(self, path: str):
        checkpointer = ocp.StandardCheckpointer()
        params = checkpointer.restore(path)
        if self.state is None:
            rng = jax.random.PRNGKey(0)
            self.state = self.create_train_state(rng)
        self.state = self.state.replace(params=params)