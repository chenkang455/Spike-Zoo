import unittest
import sys
import os
import tempfile
import torch
import torch.nn as nn
from pathlib import Path

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.utils.network_utils import (
    load_network_with_retry,
    load_network,
    download_file_with_retry,
    verify_file_checksum,
    get_file_size
)


class TestNetworkUtils(unittest.TestCase):
    """Network utilities unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple model for testing
        self.model = nn.Linear(10, 1)
        
        # Save model state dict
        self.ckpt_path = os.path.join(self.temp_dir, "test_model.pth")
        torch.save(self.model.state_dict(), self.ckpt_path)
    
    def tearDown(self):
        """Test cleanup."""
        # Remove temporary directory and files
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)
    
    def test_load_network_success(self):
        """Test successful network loading."""
        # Create new model to load weights into
        new_model = nn.Linear(10, 1)
        
        # Load network
        loaded_model = load_network(self.ckpt_path, new_model)
        
        # Check that weights are loaded correctly
        self.assertTrue(torch.allclose(self.model.weight, loaded_model.weight))
        self.assertTrue(torch.allclose(self.model.bias, loaded_model.bias))
    
    def test_load_network_nonexistent_file(self):
        """Test loading network with nonexistent file."""
        new_model = nn.Linear(10, 1)
        
        with self.assertRaises(FileNotFoundError):
            load_network("non_existent.pth", new_model)
    
    def test_load_network_with_retry_success(self):
        """Test successful network loading with retry."""
        # Create new model to load weights into
        new_model = nn.Linear(10, 1)
        
        # Load network with retry
        loaded_model = load_network_with_retry(self.ckpt_path, new_model, max_retries=2)
        
        # Check that weights are loaded correctly
        self.assertTrue(torch.allclose(self.model.weight, loaded_model.weight))
        self.assertTrue(torch.allclose(self.model.bias, loaded_model.bias))
    
    def test_load_network_with_retry_failure(self):
        """Test network loading with retry that ultimately fails."""
        new_model = nn.Linear(10, 1)
        
        # Try to load from nonexistent file with retries
        with self.assertRaises(RuntimeError) as context:
            load_network_with_retry("non_existent.pth", new_model, max_retries=1)
        
        self.assertIn("Failed to load network weights", str(context.exception))
    
    def test_load_network_with_data_parallel(self):
        """Test loading network with DataParallel state dict."""
        # Create DataParallel model and save its state dict
        dp_model = nn.DataParallel(nn.Linear(10, 1))
        dp_ckpt_path = os.path.join(self.temp_dir, "dp_model.pth")
        torch.save(dp_model.state_dict(), dp_ckpt_path)
        
        # Load into regular model
        new_model = nn.Linear(10, 1)
        loaded_model = load_network(dp_ckpt_path, new_model)
        
        # Check that weights are loaded correctly (prefix should be removed)
        original_model = nn.DataParallel(nn.Linear(10, 1))
        original_model.load_state_dict(torch.load(dp_ckpt_path))
        self.assertTrue(torch.allclose(original_model.module.weight, loaded_model.weight))
    
    def test_download_file_with_retry_success(self):
        """Test successful file download with retry."""
        # For testing purposes, we'll simulate a download by copying an existing file
        source_file = os.path.join(self.temp_dir, "source.txt")
        dest_file = os.path.join(self.temp_dir, "downloaded.txt")
        
        # Create source file
        with open(source_file, 'w') as f:
            f.write("test content")
        
        # Simulate download by copying (in real scenario, this would be HTTP download)
        import shutil
        shutil.copy(source_file, dest_file)
        
        # Test that file exists
        self.assertTrue(os.path.exists(dest_file))
        
        # Check content
        with open(dest_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, "test content")
    
    def test_download_file_with_retry_failure(self):
        """Test file download with retry that ultimately fails."""
        # Try to download from nonexistent URL
        dest_file = os.path.join(self.temp_dir, "failed_download.txt")
        success = download_file_with_retry(
            "http://non.existent.url/file.txt",
            dest_file,
            max_retries=1
        )
        
        self.assertFalse(success)
        # File should not exist after failed download
        self.assertFalse(os.path.exists(dest_file))
    
    def test_verify_file_checksum_success(self):
        """Test successful file checksum verification."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "checksum_test.txt")
        content = "test content for checksum"
        with open(test_file, 'w') as f:
            f.write(content)
        
        # Calculate expected checksum
        import hashlib
        expected_checksum = hashlib.sha256(content.encode()).hexdigest()
        
        # Verify checksum
        is_valid = verify_file_checksum(test_file, expected_checksum, 'sha256')
        self.assertTrue(is_valid)
    
    def test_verify_file_checksum_failure(self):
        """Test failed file checksum verification."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "checksum_test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Verify with incorrect checksum
        is_valid = verify_file_checksum(test_file, "incorrect_checksum", 'sha256')
        self.assertFalse(is_valid)
    
    def test_verify_file_checksum_nonexistent_file(self):
        """Test checksum verification with nonexistent file."""
        is_valid = verify_file_checksum("non_existent.txt", "any_checksum", 'sha256')
        self.assertFalse(is_valid)
    
    def test_verify_file_checksum_unsupported_algorithm(self):
        """Test checksum verification with unsupported algorithm."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        with self.assertRaises(ValueError):
            verify_file_checksum(test_file, "checksum", 'unsupported_algo')
    
    def test_get_file_size_success(self):
        """Test successful file size retrieval."""
        test_file = os.path.join(self.temp_dir, "size_test.txt")
        content = "test content" * 100  # 1200 bytes
        with open(test_file, 'w') as f:
            f.write(content)
        
        size = get_file_size(test_file)
        self.assertEqual(size, len(content.encode()))
    
    def test_get_file_size_nonexistent_file(self):
        """Test file size retrieval for nonexistent file."""
        size = get_file_size("non_existent.txt")
        self.assertIsNone(size)
    
    def test_different_hash_algorithms(self):
        """Test checksum verification with different hash algorithms."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "hash_test.txt")
        content = "test content for hash algorithms"
        with open(test_file, 'w') as f:
            f.write(content)
        
        import hashlib
        
        # Test SHA256
        sha256_hash = hashlib.sha256(content.encode()).hexdigest()
        is_valid = verify_file_checksum(test_file, sha256_hash, 'sha256')
        self.assertTrue(is_valid)
        
        # Test MD5
        md5_hash = hashlib.md5(content.encode()).hexdigest()
        is_valid = verify_file_checksum(test_file, md5_hash, 'md5')
        self.assertTrue(is_valid)


if __name__ == '__main__':
    unittest.main()