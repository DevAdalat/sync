# Production JAX Transformer Model

A scalable, production-ready transformer model built with JAX and Flax, supporting LoRA for efficient fine-tuning.

## Features

- Variable parameter sizes for flexibility
- LoRA support for parameter-efficient fine-tuning
- Training API for large datasets
- Powerful interfacing API for inference and generation
- Integration with Hugging Face tokenizers for robust text processing
- Export to ONNX and other formats (experimental)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```python
from config import ModelConfig, TrainingConfig, DataConfig
from trainer import Trainer

model_config = ModelConfig(vocab_size=1000, d_model=512, num_layers=12)
train_config = TrainingConfig(batch_size=32, num_epochs=10)
data_config = DataConfig(dataset_name="wikitext", dataset_config="wikitext-2-raw-v1")

trainer = Trainer(model_config, train_config, data_config)
trainer.fit(jax.random.PRNGKey(0))
```

### Inference

```python
from api import TransformerAPI

api = TransformerAPI.from_config("config.json")
output = api.predict("Hello world", max_len=50)
print(output)
```

### Export

```python
from export import export_to_onnx

export_to_onnx(model, params, export_config, model_config)
```

## Configuration

Use Pydantic models in `config.py` for configuration.

## Testing

Run tests with:

```bash
pytest tests/
```