"""
Production JAX Transformer Model

A scalable, production-ready transformer model built with JAX and Flax, supporting LoRA for efficient fine-tuning.
"""

__version__ = "1.0.0"

from .models.model import ProductionTransformer, LoRALinear
from .config.config import ModelConfig, TrainingConfig, DataConfig, CloudConfig
from .training.trainer import Trainer
from .utils.utils import *

__all__ = [
    "ProductionTransformer",
    "LoRALinear", 
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "CloudConfig",
    "Trainer"
]