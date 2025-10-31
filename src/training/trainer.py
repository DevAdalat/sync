import logging
import os
import shutil
import time
from datetime import datetime
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset
from flax.training import train_state
from orbax import checkpoint as ocp

from ..config.config import DataConfig, ModelConfig, TrainingConfig
from ..models.model import ProductionTransformer
from ..utils.utils import download_checkpoint_from_cloud, upload_checkpoint_to_cloud

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config
        self.dataset = load_dataset(
            config.dataset_name, config.dataset_config, split=config.split
        )

    def preprocess(self, examples):
        # Placeholder preprocessing
        return {
            "input_ids": examples[self.config.text_column][: self.config.max_length]
        }

    def get_batch(self, batch_size: int):
        # Simple batching
        batch = self.dataset.select(range(batch_size))
        return self.preprocess(batch)


class Trainer:
    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        data_config: Optional[DataConfig] = None,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config
        self.model = ProductionTransformer(model_config)
        self.data_loader = DataLoader(data_config) if data_config else None
        self.state = None

        # Detect multi-device setup
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        self.use_multi_device = self.num_devices > 1

        if self.use_multi_device:
            logger.info(
                f"Multi-device training enabled: {self.num_devices} devices detected"
            )
        else:
            logger.info(f"Single-device training: {self.devices[0]}")

    def create_train_state(self, rng):
        params = self.model.init(
            rng, jnp.ones((1, self.model_config.max_len), dtype=jnp.int32)
        )
        tx = optax.adamw(
            self.train_config.learning_rate, weight_decay=self.train_config.weight_decay
        )
        if self.train_config.warmup_steps > 0:
            tx = optax.chain(
                optax.clip_by_global_norm(self.train_config.gradient_clip_norm),
                optax.warmup_cosine_decay_schedule(
                    init_value=0.0,
                    peak_value=self.train_config.learning_rate,
                    warmup_steps=self.train_config.warmup_steps,
                    decay_steps=self.train_config.max_steps or 10000,
                ),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self.train_config.gradient_clip_norm), tx
            )
        return train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )

    def loss_fn(self, params, batch):
        logits = self.model.apply(params, batch["input_ids"])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"])
        return jnp.mean(loss)

    def get_train_step(self):
        if self.use_multi_device:
            # Multi-device training step using pmap
            @jax.pmap
            def train_step(state, batch):
                loss, grads = jax.value_and_grad(self.loss_fn)(state.params, batch)
                state = state.apply_gradients(grads=grads)
                return state, loss

            logger.info(f"Using pmap train step for {self.num_devices} devices")
        else:
            # Single-device training step
            @jax.jit
            def train_step(state, batch):
                loss, grads = jax.value_and_grad(self.loss_fn)(state.params, batch)
                state = state.apply_gradients(grads=grads)
                return state, loss

        return train_step

    def fit(self, rng):
        if self.state is None:
            self.state = self.create_train_state(rng)

        # Replicate state across devices if using multi-device training
        if self.use_multi_device:
            logger.info(f"Replicating model state across {self.num_devices} devices...")
            self.state = jax.device_put_replicated(self.state, self.devices)

        train_step = self.get_train_step()

        # Track training start time for timeout functionality
        training_start_time = time.time()
        global_step = 0

        # Calculate effective batch size
        effective_batch_size = (
            self.train_config.batch_size * self.num_devices
            if self.use_multi_device
            else self.train_config.batch_size
        )

        for epoch in range(self.train_config.num_epochs):
            for step in range(
                self.train_config.max_steps
                or len(self.data_loader.dataset) // effective_batch_size
                if self.data_loader
                else 100
            ):
                # Check timeout before each step
                if self.train_config.timeout_seconds:
                    elapsed_time = time.time() - training_start_time
                    if elapsed_time >= self.train_config.timeout_seconds:
                        logger.info(
                            f"Training timeout reached after {elapsed_time:.2f} seconds. Saving final checkpoint..."
                        )
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        checkpoint_name = f"timeout_checkpoint_{timestamp}"
                        cloud_path = self.save_checkpoint(checkpoint_name)
                        logger.info(f"Training stopped. Checkpoint saved: {cloud_path}")
                        return cloud_path  # Return cloud path for resuming

                # Prepare batch for single or multi-device training
                if self.data_loader:
                    batch = self.data_loader.get_batch(effective_batch_size)
                    if self.use_multi_device:
                        # Reshape batch for pmap: (num_devices, per_device_batch_size, ...)
                        batch = {
                            k: v.reshape(
                                self.num_devices, self.train_config.batch_size, -1
                            )
                            for k, v in batch.items()
                        }
                else:
                    # Dummy batch
                    if self.use_multi_device:
                        batch = {
                            "input_ids": jnp.ones(
                                (
                                    self.num_devices,
                                    self.train_config.batch_size,
                                    self.model_config.max_len,
                                ),
                                dtype=jnp.int32,
                            ),
                            "labels": jnp.ones(
                                (
                                    self.num_devices,
                                    self.train_config.batch_size,
                                    self.model_config.max_len,
                                ),
                                dtype=jnp.int32,
                            ),
                        }
                    else:
                        batch = {
                            "input_ids": jnp.ones(
                                (
                                    self.train_config.batch_size,
                                    self.model_config.max_len,
                                ),
                                dtype=jnp.int32,
                            ),
                            "labels": jnp.ones(
                                (
                                    self.train_config.batch_size,
                                    self.model_config.max_len,
                                ),
                                dtype=jnp.int32,
                            ),
                        }

                self.state, loss = train_step(self.state, batch)
                global_step += 1

                # Average loss across devices for logging
                current_loss = jnp.mean(loss) if self.use_multi_device else loss

                if step % self.train_config.log_steps == 0:
                    logger.info(f"Epoch {epoch}, Step {step}, Loss: {current_loss:.4f}")

                if step % self.train_config.save_steps == 0:
                    checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}"
                    self.save_checkpoint(checkpoint_name)

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
        # Convert to absolute path as required by orbax
        abs_path = os.path.abspath(path)
        # Remove existing checkpoint directory if it exists
        if os.path.exists(abs_path):
            shutil.rmtree(abs_path)
        checkpointer = ocp.PyTreeCheckpointer()

        # Extract params from first device if using multi-device training
        params_to_save = (
            jax.tree_map(lambda x: x[0], self.state.params)
            if self.use_multi_device
            else self.state.params
        )
        checkpointer.save(abs_path, params_to_save)

        # Upload to cloud if configured
        if self.train_config.cloud_config:
            checkpoint_name = os.path.basename(path)
            cloud_path = upload_checkpoint_to_cloud(
                abs_path, self.train_config.cloud_config, checkpoint_name
            )
            logger.info(f"Checkpoint uploaded to cloud: {cloud_path}")
            return cloud_path
        return abs_path

    def load_checkpoint(self, path: str):
        # Check if path is a cloud URL
        if self.train_config.cloud_config and (
            path.startswith("s3://")
            or path.startswith("gs://")
            or "blob.core.windows.net" in path
        ):
            # Download from cloud first
            local_temp_dir = f"/tmp/checkpoint_{int(time.time())}"
            os.makedirs(local_temp_dir, exist_ok=True)
            local_path = download_checkpoint_from_cloud(
                path, self.train_config.cloud_config, local_temp_dir
            )
            abs_path = os.path.abspath(local_path)
        else:
            abs_path = os.path.abspath(path)

        checkpointer = ocp.PyTreeCheckpointer()
        params = checkpointer.restore(abs_path)
        if self.state is None:
            rng = jax.random.PRNGKey(0)
            self.state = self.create_train_state(rng)
        self.state = self.state.replace(params=params)

        # Clean up temp directory if downloaded from cloud
        if self.train_config.cloud_config and (
            path.startswith("s3://")
            or path.startswith("gs://")
            or "blob.core.windows.net" in path
        ):
            shutil.rmtree(os.path.dirname(abs_path))
