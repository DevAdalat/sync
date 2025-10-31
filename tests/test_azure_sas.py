#!/usr/bin/env python3
"""
Test script for Azure SAS bucket functionality.
This demonstrates how to use the Azure SAS token authentication.
"""

import os
import tempfile

from src.config.config import CloudConfig
from src.utils.utils import download_checkpoint_from_cloud, upload_checkpoint_to_cloud


def test_azure_sas():
    """Test Azure SAS upload/download functionality."""

    # Example Azure SAS configuration
    cloud_config = CloudConfig(
        provider="azure",
        bucket_name="my-container",  # Azure container name
        region="eastus",  # Azure region
        account_name="mystorageaccount",
        sas_token="?sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-12-31T23:59:59Z&st=2024-01-01T00:00:00Z&spr=https&sig=abcd1234...",  # Example SAS token
        prefix="test-checkpoints",
    )

    # Create a dummy checkpoint directory
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint")
        os.makedirs(checkpoint_path)

        # Create a dummy file
        with open(os.path.join(checkpoint_path, "params.pkl"), "w") as f:
            f.write("dummy checkpoint data")

        print("Testing Azure SAS upload...")
        try:
            cloud_path = upload_checkpoint_to_cloud(
                checkpoint_path, cloud_config, "test_checkpoint"
            )
            print(f"Upload successful! Cloud path: {cloud_path}")

            # Test download
            download_dir = os.path.join(temp_dir, "downloaded")
            os.makedirs(download_dir)

            print("Testing Azure SAS download...")
            local_path = download_checkpoint_from_cloud(
                cloud_path, cloud_config, download_dir
            )
            print(f"Download successful! Local path: {local_path}")

        except Exception as e:
            print(f"Test failed (expected if Azure credentials not configured): {e}")
            print("This is normal - the test demonstrates the API structure")


if __name__ == "__main__":
    test_azure_sas()
