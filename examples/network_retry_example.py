#!/usr/bin/env python3
"""
Example of using the SpikeZoo network utilities with retry mechanisms.
"""

import sys
import os
import tempfile
import torch
import torch.nn as nn

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.utils.network_utils import (
    load_network_with_retry,
    download_file_with_retry,
    verify_file_checksum,
    get_file_size
)
from spikezoo.models.base_model import BaseModelConfig
from spikezoo.models import BaseModel


class SimpleModel(BaseModel):
    """Simple model for testing."""
    
    def __init__(self, cfg: BaseModelConfig):
        super().__init__(cfg)
        self.net = nn.Linear(10, 1)
    
    def spk2img(self, spike):
        """Simple spike to image conversion."""
        return self.net(spike)


def example_load_network_with_retry():
    """Example of loading network with retry mechanism."""
    print("=== Network Loading with Retry Example ===\n")
    
    # Create a temporary model and save it
    model = nn.Linear(10, 1)
    temp_dir = tempfile.mkdtemp()
    ckpt_path = os.path.join(temp_dir, "test_model.pth")
    
    # Save model state dict
    torch.save(model.state_dict(), ckpt_path)
    print(f"1. Saved test model to {ckpt_path}")
    
    # Create a new model to load weights into
    new_model = nn.Linear(10, 1)
    
    # Load network with retry
    try:
        loaded_model = load_network_with_retry(ckpt_path, new_model, max_retries=2)
        print(f"2. Successfully loaded model with retry mechanism")
        print(f"   Original weight: {model.weight.data.flatten()[0]:.6f}")
        print(f"   Loaded weight: {loaded_model.weight.data.flatten()[0]:.6f}")
    except Exception as e:
        print(f"2. Failed to load model: {e}")
    
    # Clean up
    os.remove(ckpt_path)
    os.rmdir(temp_dir)
    print()


def example_download_with_retry():
    """Example of downloading file with retry mechanism."""
    print("=== File Download with Retry Example ===\n")
    
    # Create a temporary file to serve as a mock download
    temp_dir = tempfile.mkdtemp()
    source_file = os.path.join(temp_dir, "source.txt")
    dest_file = os.path.join(temp_dir, "downloaded.txt")
    
    # Create source file
    with open(source_file, 'w') as f:
        f.write("This is a test file for download retry example.\n" * 100)
    
    # For this example, we'll simulate download by copying the file
    # In a real scenario, you would use an actual URL
    import shutil
    shutil.copy(source_file, dest_file)
    
    print(f"1. Created source file: {source_file}")
    print(f"2. Simulated download to: {dest_file}")
    
    # Verify file was "downloaded"
    if os.path.exists(dest_file):
        file_size = get_file_size(dest_file)
        print(f"3. Downloaded file size: {file_size} bytes")
        
        # Read a portion of the file to verify content
        with open(dest_file, 'r') as f:
            content = f.read(50)
            print(f"4. File content preview: {content}...")
    else:
        print("3. Download failed")
    
    # Clean up
    os.remove(source_file)
    os.remove(dest_file)
    os.rmdir(temp_dir)
    print()


def example_checksum_verification():
    """Example of file checksum verification."""
    print("=== Checksum Verification Example ===\n")
    
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test_file.txt")
    
    # Create test file
    content = "This is a test file for checksum verification.\n" * 50
    with open(test_file, 'w') as f:
        f.write(content)
    
    print(f"1. Created test file: {test_file}")
    
    # Calculate expected checksum (in a real scenario, this would be provided)
    import hashlib
    with open(test_file, 'rb') as f:
        expected_checksum = hashlib.sha256(f.read()).hexdigest()
    
    print(f"2. Expected SHA256 checksum: {expected_checksum[:32]}...")
    
    # Verify checksum
    is_valid = verify_file_checksum(test_file, expected_checksum, 'sha256')
    print(f"3. Checksum verification: {'PASSED' if is_valid else 'FAILED'}")
    
    # Test with incorrect checksum
    is_valid = verify_file_checksum(test_file, "incorrect_checksum", 'sha256')
    print(f"4. Incorrect checksum verification: {'PASSED' if is_valid else 'FAILED'}")
    
    # Clean up
    os.remove(test_file)
    os.rmdir(temp_dir)
    print()


def example_model_config_retry_settings():
    """Example of model configuration with retry settings."""
    print("=== Model Configuration Retry Settings Example ===\n")
    
    # Create model config with custom retry settings
    config = BaseModelConfig(
        model_name="test_model",
        load_state=True,
        max_retry_attempts=5,
        retry_delay=2.0
    )
    
    print("1. Model configuration with retry settings:")
    print(f"   Model name: {config.model_name}")
    print(f"   Max retry attempts: {config.max_retry_attempts}")
    print(f"   Retry delay: {config.retry_delay} seconds")
    print(f"   Load state: {config.load_state}")
    print()
    
    # Create model with config
    model = SimpleModel(config)
    print("2. Created model with custom retry configuration")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Config max retries: {model.cfg.max_retry_attempts}")
    print(f"   Config retry delay: {model.cfg.retry_delay}")
    print()


def example_error_handling():
    """Example of error handling with retry mechanisms."""
    print("=== Error Handling Example ===\n")
    
    # Test loading non-existent file
    print("1. Testing load of non-existent file:")
    try:
        dummy_model = nn.Linear(10, 1)
        loaded_model = load_network_with_retry("non_existent.pth", dummy_model, max_retries=1)
    except RuntimeError as e:
        print(f"   Caught expected error: {e}")
    except Exception as e:
        print(f"   Caught unexpected error: {e}")
    
    # Test download of non-existent URL
    print("\n2. Testing download of non-existent URL:")
    try:
        success = download_file_with_retry(
            "http://non.existent.url/file.txt",
            "/tmp/non_existent_file.txt",
            max_retries=1
        )
        print(f"   Download success: {success}")
    except Exception as e:
        print(f"   Caught error during download: {e}")
    
    print()


if __name__ == "__main__":
    example_load_network_with_retry()
    example_download_with_retry()
    example_checksum_verification()
    example_model_config_retry_settings()
    example_error_handling()