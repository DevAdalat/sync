from typing import Literal, Optional

from pydantic import BaseModel, Field

# Predefined model size configurations
MODEL_PRESETS = {
    "nano": {
        "d_model": 128,
        "num_layers": 6,
        "num_heads": 4,
        "d_ff": 512,
        "params": "~1M",
        "description": "Tiny model for testing and ultra-fast inference",
    },
    "tiny": {
        "d_model": 256,
        "num_layers": 8,
        "num_heads": 8,
        "d_ff": 1024,
        "params": "~5M",
        "description": "Small model for resource-constrained environments",
    },
    "small": {
        "d_model": 512,
        "num_layers": 12,
        "num_heads": 8,
        "d_ff": 2048,
        "params": "~50M",
        "description": "Balanced model for general tasks",
    },
    "medium": {
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
        "params": "~125M",
        "description": "Similar to GPT-2 Small / BERT Base",
    },
    "large": {
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
        "params": "~350M",
        "description": "Large model for high-quality generation",
    },
    "xlarge": {
        "d_model": 1280,
        "num_layers": 32,
        "num_heads": 20,
        "d_ff": 5120,
        "params": "~700M",
        "description": "Extra large model for production use",
    },
}


class ModelConfig(BaseModel):
    vocab_size: int = Field(..., description="Vocabulary size")
    d_model: int = Field(64, description="Model dimension")
    num_heads: int = Field(4, description="Number of attention heads")
    num_layers: int = Field(2, description="Number of transformer layers")
    d_ff: int = Field(128, description="Feed-forward dimension")
    max_len: int = Field(100, description="Maximum sequence length")
    dropout_rate: float = Field(0.1, description="Dropout rate")
    activation: Literal["relu", "gelu", "silu"] = Field(
        "gelu", description="Activation function"
    )
    use_lora: bool = Field(False, description="Enable LoRA")
    lora_rank: int = Field(8, description="LoRA rank")
    use_rmsnorm: bool = Field(True, description="Use RMSNorm instead of LayerNorm")
    use_swiglu: bool = Field(True, description="Use SwiGLU activation in FFN")
    use_rope: bool = Field(True, description="Use Rotary Position Embeddings")

    @classmethod
    def from_preset(
        cls,
        preset: Literal["nano", "tiny", "small", "medium", "large", "xlarge"],
        vocab_size: int,
        max_len: int = 128,
        **kwargs,
    ):
        """
        Create a ModelConfig from a predefined preset.

        Args:
            preset: Model size preset (nano, tiny, small, medium, large, xlarge)
            vocab_size: Vocabulary size for the tokenizer
            max_len: Maximum sequence length (default: 128)
            **kwargs: Additional config overrides (dropout_rate, activation, etc.)

        Returns:
            ModelConfig with preset architecture

        Example:
            >>> config = ModelConfig.from_preset("tiny", vocab_size=8000, max_len=128)
            >>> config = ModelConfig.from_preset("medium", vocab_size=50000, dropout_rate=0.2)
        """
        if preset not in MODEL_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available presets: {list(MODEL_PRESETS.keys())}"
            )

        preset_config = MODEL_PRESETS[preset].copy()
        # Remove metadata fields
        preset_config.pop("params", None)
        preset_config.pop("description", None)

        # Merge preset with required args and overrides
        config_dict = {
            "vocab_size": vocab_size,
            "max_len": max_len,
            **preset_config,
            **kwargs,  # User overrides take precedence
        }

        return cls(**config_dict)

    @staticmethod
    def list_presets():
        """Print available model presets with their specifications."""
        print("\n" + "=" * 80)
        print("Available Model Presets".center(80))
        print("=" * 80)
        for name, config in MODEL_PRESETS.items():
            print(f"\n{name.upper()}")
            print(f"  Size:        {config['params']}")
            print(f"  Dimensions:  {config['d_model']}")
            print(f"  Layers:      {config['num_layers']}")
            print(f"  Heads:       {config['num_heads']}")
            print(f"  FFN Size:    {config['d_ff']}")
            print(f"  Description: {config['description']}")
        print("\n" + "=" * 80)
        print("Usage: ModelConfig.from_preset('tiny', vocab_size=8000)")
        print("=" * 80 + "\n")


class CloudConfig(BaseModel):
    provider: Literal["s3", "gcs", "azure"] = Field(
        "s3", description="Cloud storage provider"
    )
    bucket_name: str = Field(..., description="Cloud bucket/container name")
    region: Optional[str] = Field(None, description="AWS region or Azure region")
    prefix: str = Field("checkpoints", description="Prefix for checkpoint files")
    sas_token: Optional[str] = Field(
        None, description="Azure SAS token (for SAS authentication)"
    )
    account_name: Optional[str] = Field(None, description="Azure storage account name")


class TrainingConfig(BaseModel):
    batch_size: int = Field(32, description="Batch size")
    learning_rate: float = Field(1e-3, description="Learning rate")
    num_epochs: int = Field(10, description="Number of epochs")
    max_steps: Optional[int] = Field(None, description="Maximum training steps")
    warmup_steps: int = Field(0, description="Warmup steps")
    weight_decay: float = Field(0.01, description="Weight decay")
    gradient_clip_norm: float = Field(1.0, description="Gradient clipping norm")
    save_steps: int = Field(1000, description="Save checkpoint every N steps")
    eval_steps: int = Field(500, description="Evaluate every N steps")
    log_steps: int = Field(10, description="Log every N steps")
    timeout_seconds: Optional[int] = Field(
        None, description="Training timeout in seconds"
    )
    cloud_config: Optional[CloudConfig] = Field(
        None, description="Cloud storage configuration for checkpoints"
    )


class DataConfig(BaseModel):
    dataset_name: str = Field(..., description="Dataset name (e.g., 'wikitext')")
    dataset_config: Optional[str] = Field(None, description="Dataset config")
    split: str = Field("train", description="Dataset split")
    text_column: str = Field("text", description="Text column name")
    max_length: int = Field(512, description="Maximum sequence length for tokenization")
    stride: int = Field(128, description="Stride for overlapping sequences")


class ExportConfig(BaseModel):
    format: Literal["onnx", "saved_model"] = Field("onnx", description="Export format")
    output_path: str = Field(..., description="Output path for exported model")
    input_shape: tuple = Field((1, 100), description="Input shape for export")
