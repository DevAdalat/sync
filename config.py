from pydantic import BaseModel, Field
from typing import Optional, Literal

class ModelConfig(BaseModel):
    vocab_size: int = Field(..., description="Vocabulary size")
    d_model: int = Field(64, description="Model dimension")
    num_heads: int = Field(4, description="Number of attention heads")
    num_layers: int = Field(2, description="Number of transformer layers")
    d_ff: int = Field(128, description="Feed-forward dimension")
    max_len: int = Field(100, description="Maximum sequence length")
    dropout_rate: float = Field(0.1, description="Dropout rate")
    activation: Literal["relu", "gelu"] = Field("gelu", description="Activation function")
    use_lora: bool = Field(False, description="Enable LoRA")
    lora_rank: int = Field(8, description="LoRA rank")

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