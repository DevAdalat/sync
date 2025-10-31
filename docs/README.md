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

### Training with Timeout and Cloud Checkpointing

The trainer supports automatic checkpointing with timeout and cloud storage integration.

#### Basic Timeout Training

```python
from config import ModelConfig, TrainingConfig, CloudConfig

model_config = ModelConfig(vocab_size=1000, d_model=512, num_layers=12)

# Configure cloud storage
cloud_config = CloudConfig(
    provider="s3",  # or "gcs", "azure"
    bucket_name="my-training-bucket",
    region="us-west-2"
)

train_config = TrainingConfig(
    batch_size=32,
    num_epochs=10,
    timeout_seconds=3600,  # 1 hour timeout
    cloud_config=cloud_config
)

trainer = Trainer(model_config, train_config, data_config)
result = trainer.fit(jax.random.PRNGKey(0))

if result:
    print(f"Training stopped due to timeout. Checkpoint: {result}")
```

#### Command Line Training with Cloud Storage

```bash
# Train with 1-hour timeout and S3 storage
python train_tiny_shakespeare.py --timeout 3600 --cloud-bucket my-bucket --cloud-provider s3 --cloud-region us-west-2

# Train with Azure SAS token
python train_tiny_shakespeare.py --timeout 3600 --cloud-bucket my-container --cloud-provider azure --azure-account-name mystorage --azure-sas-token "?sv=...&st=...&se=...&sr=...&sp=...&sig=..."

# Resume training from cloud checkpoint
python train_tiny_shakespeare.py --resume-from "s3://my-bucket/checkpoints/timeout_checkpoint_20241201_120000.zip" --cloud-bucket my-bucket --cloud-provider s3
```

#### Cloud Storage Providers

- **AWS S3**: Set `provider="s3"` and provide `bucket_name` and `region`
- **Google Cloud Storage**: Set `provider="gcs"` and provide `bucket_name`
- **Azure Blob Storage**: Set `provider="azure"`, provide `bucket_name` (container), `account_name`, and `sas_token`

For Azure SAS authentication, the SAS token should include the necessary permissions for blob upload/download operations.

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