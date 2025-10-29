import jax.numpy as jnp
from typing import List, Dict, Any, Optional
import logging
from tokenizers import Tokenizer
import os
import tempfile
import shutil
from pathlib import Path
from config import CloudConfig

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level.upper()))

def load_tokenizer(path: Optional[str] = None, pretrained: Optional[str] = None) -> Tokenizer:
    """Load tokenizer from file or pretrained."""
    if path:
        return Tokenizer.from_file(path)
    elif pretrained:
        return Tokenizer.from_pretrained(pretrained)
    else:
        # Default to BERT base uncased
        return Tokenizer.from_pretrained("bert-base-uncased")

def tokenize_text(text: str, tokenizer: Tokenizer, max_len: int) -> List[int]:
    """Tokenize text using the tokenizer."""
    encoded = tokenizer.encode(text)
    return encoded.ids[:max_len]

def detokenize_text(tokens: List[int], tokenizer: Tokenizer) -> str:
    """Detokenize tokens to text."""
    return tokenizer.decode(tokens)

def get_vocab(tokenizer: Tokenizer) -> Dict[str, int]:
    """Get vocabulary from tokenizer."""
    return tokenizer.get_vocab(with_added_tokens=True)

def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return float(jnp.exp(loss))

def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> float:
    """Compute accuracy."""
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == labels))

# Deprecated, use get_vocab
def load_vocab(path: str) -> Dict[str, int]:
    """Load vocabulary (deprecated, use tokenizer)."""
    tokenizer = load_tokenizer(path)
    return get_vocab(tokenizer)

def save_vocab(vocab: Dict[str, int], path: str):
    """Save vocabulary (deprecated)."""
    pass  # Not needed with tokenizer

# Cloud storage utilities
def upload_checkpoint_to_cloud(local_path: str, cloud_config: CloudConfig, checkpoint_name: str) -> str:
    """Upload checkpoint to cloud storage and return the cloud path/key."""
    if cloud_config.provider == "s3":
        return _upload_to_s3(local_path, cloud_config, checkpoint_name)
    elif cloud_config.provider == "gcs":
        return _upload_to_gcs(local_path, cloud_config, checkpoint_name)
    elif cloud_config.provider == "azure":
        return _upload_to_azure(local_path, cloud_config, checkpoint_name)
    else:
        raise ValueError(f"Unsupported cloud provider: {cloud_config.provider}")

def download_checkpoint_from_cloud(cloud_path: str, cloud_config: CloudConfig, local_path: str) -> str:
    """Download checkpoint from cloud storage to local path."""
    if cloud_config.provider == "s3":
        return _download_from_s3(cloud_path, cloud_config, local_path)
    elif cloud_config.provider == "gcs":
        return _download_from_gcs(cloud_path, cloud_config, local_path)
    elif cloud_config.provider == "azure":
        return _download_from_azure(cloud_path, cloud_config, local_path)
    else:
        raise ValueError(f"Unsupported cloud provider: {cloud_config.provider}")

def _upload_to_s3(local_path: str, cloud_config: CloudConfig, checkpoint_name: str) -> str:
    """Upload to AWS S3."""
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for S3 uploads. Install with: pip install boto3")

    s3_client = boto3.client('s3', region_name=cloud_config.region)
    key = f"{cloud_config.prefix}/{checkpoint_name}"

    # Upload directory as zip
    import zipfile
    zip_path = f"{local_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(local_path):
            for file in files:
                zipf.write(os.path.join(root, file),
                          os.path.relpath(os.path.join(root, file), local_path))

    s3_client.upload_file(zip_path, cloud_config.bucket_name, f"{key}.zip")
    os.remove(zip_path)

    logger.info(f"Uploaded checkpoint to s3://{cloud_config.bucket_name}/{key}.zip")
    return f"s3://{cloud_config.bucket_name}/{key}.zip"

def _download_from_s3(cloud_path: str, cloud_config: CloudConfig, local_path: str) -> str:
    """Download from AWS S3."""
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for S3 downloads. Install with: pip install boto3")

    s3_client = boto3.client('s3', region_name=cloud_config.region)

    # Extract key from cloud_path
    key = cloud_path.replace(f"s3://{cloud_config.bucket_name}/", "").replace(".zip", "")
    zip_key = f"{key}.zip"

    # Download zip file
    zip_path = f"{local_path}.zip"
    s3_client.download_file(cloud_config.bucket_name, zip_key, zip_path)

    # Extract zip
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(local_path)

    os.remove(zip_path)
    logger.info(f"Downloaded checkpoint from {cloud_path}")
    return local_path

def _upload_to_gcs(local_path: str, cloud_config: CloudConfig, checkpoint_name: str) -> str:
    """Upload to Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage is required for GCS uploads. Install with: pip install google-cloud-storage")

    client = storage.Client()
    bucket = client.bucket(cloud_config.bucket_name)
    key = f"{cloud_config.prefix}/{checkpoint_name}"

    # Upload directory as zip
    import zipfile
    zip_path = f"{local_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(local_path):
            for file in files:
                zipf.write(os.path.join(root, file),
                          os.path.relpath(os.path.join(root, file), local_path))

    blob = bucket.blob(f"{key}.zip")
    blob.upload_from_filename(zip_path)
    os.remove(zip_path)

    logger.info(f"Uploaded checkpoint to gs://{cloud_config.bucket_name}/{key}.zip")
    return f"gs://{cloud_config.bucket_name}/{key}.zip"

def _download_from_gcs(cloud_path: str, cloud_config: CloudConfig, local_path: str) -> str:
    """Download from Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage is required for GCS downloads. Install with: pip install google-cloud-storage")

    client = storage.Client()
    bucket = client.bucket(cloud_config.bucket_name)

    # Extract key from cloud_path
    key = cloud_path.replace(f"gs://{cloud_config.bucket_name}/", "").replace(".zip", "")
    zip_key = f"{key}.zip"

    blob = bucket.blob(zip_key)
    zip_path = f"{local_path}.zip"
    blob.download_to_filename(zip_path)

    # Extract zip
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(local_path)

    os.remove(zip_path)
    logger.info(f"Downloaded checkpoint from {cloud_path}")
    return local_path

def _upload_to_azure(local_path: str, cloud_config: CloudConfig, checkpoint_name: str) -> str:
    """Upload to Azure Blob Storage using SAS token."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise ImportError("azure-storage-blob is required for Azure uploads. Install with: pip install azure-storage-blob")

    if not cloud_config.sas_token or not cloud_config.account_name:
        raise ValueError("Azure SAS token and account name are required for Azure uploads")

    # Construct SAS URL
    account_url = f"https://{cloud_config.account_name}.blob.core.windows.net{cloud_config.sas_token}"
    blob_service_client = BlobServiceClient(account_url=account_url)

    container_client = blob_service_client.get_container_client(cloud_config.bucket_name)
    key = f"{cloud_config.prefix}/{checkpoint_name}"

    # Upload directory as zip
    import zipfile
    zip_path = f"{local_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(local_path):
            for file in files:
                zipf.write(os.path.join(root, file),
                          os.path.relpath(os.path.join(root, file), local_path))

    blob_client = container_client.get_blob_client(f"{key}.zip")
    with open(zip_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    os.remove(zip_path)
    blob_url = f"https://{cloud_config.account_name}.blob.core.windows.net/{cloud_config.bucket_name}/{key}.zip{cloud_config.sas_token}"

    logger.info(f"Uploaded checkpoint to Azure: {blob_url}")
    return blob_url

def _download_from_azure(cloud_path: str, cloud_config: CloudConfig, local_path: str) -> str:
    """Download from Azure Blob Storage using SAS token."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise ImportError("azure-storage-blob is required for Azure downloads. Install with: pip install azure-storage-blob")

    if not cloud_config.sas_token or not cloud_config.account_name:
        raise ValueError("Azure SAS token and account name are required for Azure downloads")

    # Construct SAS URL
    account_url = f"https://{cloud_config.account_name}.blob.core.windows.net{cloud_config.sas_token}"
    blob_service_client = BlobServiceClient(account_url=account_url)

    container_client = blob_service_client.get_container_client(cloud_config.bucket_name)

    # Extract blob name from cloud_path (remove SAS token if present)
    if "?" in cloud_path:
        blob_path = cloud_path.split("?")[0].split(f"/{cloud_config.bucket_name}/")[1]
    else:
        blob_path = cloud_path.replace(f"https://{cloud_config.account_name}.blob.core.windows.net/{cloud_config.bucket_name}/", "")

    blob_client = container_client.get_blob_client(blob_path)
    zip_path = f"{local_path}.zip"

    with open(zip_path, "wb") as download_file:
        download_stream = blob_client.download_blob()
        download_file.write(download_stream.readall())

    # Extract zip
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(local_path)

    os.remove(zip_path)
    logger.info(f"Downloaded checkpoint from Azure: {cloud_path}")
    return local_path