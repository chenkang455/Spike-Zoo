import unittest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from spikezoo.utils.validation_utils import (
    ParameterValidator,
    validate_infer_from_dataset_params,
    validate_infer_from_file_params,
    validate_infer_from_spk_params
)


class TestValidationUtils(unittest.TestCase):
    """Validation utilities unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy files for testing
        self.test_file = Path(self.temp_dir) / "test.dat"
        self.test_file.touch()
        
        self.test_img = Path(self.temp_dir) / "test.png"
        self.test_img.touch()
    
    def tearDown(self):
        """Test cleanup."""
        # Remove temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)
    
    def test_validate_dataset_index_valid(self):
        """Test validating valid dataset index."""
        # Should not raise exception
        ParameterValidator.validate_dataset_index(0, 10)
        ParameterValidator.validate_dataset_index(5, 10)
        ParameterValidator.validate_dataset_index(9, 10)
    
    def test_validate_dataset_index_invalid_type(self):
        """Test validating dataset index with invalid type."""
        with self.assertRaises(TypeError):
            ParameterValidator.validate_dataset_index("0", 10)
        
        with self.assertRaises(TypeError):
            ParameterValidator.validate_dataset_index(0.5, 10)
    
    def test_validate_dataset_index_negative(self):
        """Test validating negative dataset index."""
        with self.assertRaises(ValueError):
            ParameterValidator.validate_dataset_index(-1, 10)
    
    def test_validate_dataset_index_out_of_range(self):
        """Test validating out of range dataset index."""
        with self.assertRaises(ValueError):
            ParameterValidator.validate_dataset_index(10, 10)
    
    def test_validate_file_path_valid(self):
        """Test validating valid file path."""
        # Should not raise exception
        ParameterValidator.validate_file_path(self.test_file)
    
    def test_validate_file_path_invalid_type(self):
        """Test validating file path with invalid type."""
        with self.assertRaises(TypeError):
            ParameterValidator.validate_file_path(123)
        
        with self.assertRaises(TypeError):
            ParameterValidator.validate_file_path(None)
    
    def test_validate_file_path_nonexistent(self):
        """Test validating nonexistent file path."""
        with self.assertRaises(FileNotFoundError):
            ParameterValidator.validate_file_path("nonexistent.txt")
    
    def test_validate_file_path_directory(self):
        """Test validating directory path as file path."""
        with self.assertRaises(ValueError):
            ParameterValidator.validate_file_path(self.temp_dir)
    
    def test_validate_dimensions_valid(self):
        """Test validating valid dimensions."""
        # Should not raise exception
        ParameterValidator.validate_dimensions(100, 200)
        ParameterValidator.validate_dimensions(1, 1)
    
    def test_validate_dimensions_invalid_type(self):
        """Test validating dimensions with invalid type."""
        with self.assertRaises(TypeError):
            ParameterValidator.validate_dimensions("100", 200)
        
        with self.assertRaises(TypeError):
            ParameterValidator.validate_dimensions(100, "200")
    
    def test_validate_dimensions_negative(self):
        """Test validating negative dimensions."""
        with self.assertRaises(ValueError):
            ParameterValidator.validate_dimensions(-1, 200)
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_dimensions(100, -1)
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_dimensions(0, 200)
    
    def test_validate_rate_valid(self):
        """Test validating valid rate."""
        # Should not raise exception
        ParameterValidator.validate_rate(1.0)
        ParameterValidator.validate_rate(0.5)
        ParameterValidator.validate_rate(100)
    
    def test_validate_rate_invalid_type(self):
        """Test validating rate with invalid type."""
        with self.assertRaises(TypeError):
            ParameterValidator.validate_rate("1.0")
        
        with self.assertRaises(TypeError):
            ParameterValidator.validate_rate(None)
    
    def test_validate_rate_negative(self):
        """Test validating negative rate."""
        with self.assertRaises(ValueError):
            ParameterValidator.validate_rate(-1.0)
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_rate(0)
    
    def test_validate_image_path_valid(self):
        """Test validating valid image path."""
        # Should not raise exception
        ParameterValidator.validate_image_path(self.test_img)
        ParameterValidator.validate_image_path(None)
    
    def test_validate_image_path_invalid_type(self):
        """Test validating image path with invalid type."""
        with self.assertRaises(TypeError):
            ParameterValidator.validate_image_path(123)
    
    def test_validate_image_path_nonexistent(self):
        """Test validating nonexistent image path."""
        with self.assertRaises(FileNotFoundError):
            ParameterValidator.validate_image_path("nonexistent.png")
    
    def test_validate_spike_data_valid(self):
        """Test validating valid spike data."""
        # Should not raise exception
        spike_np = np.random.randn(200, 250, 400)
        ParameterValidator.validate_spike_data(spike_np)
        
        spike_torch = torch.randn(1, 200, 250, 400)
        ParameterValidator.validate_spike_data(spike_torch)
    
    def test_validate_spike_data_invalid_type(self):
        """Test validating spike data with invalid type."""
        with self.assertRaises(TypeError):
            ParameterValidator.validate_spike_data("invalid")
        
        with self.assertRaises(TypeError):
            ParameterValidator.validate_spike_data(None)
        
        with self.assertRaises(TypeError):
            ParameterValidator.validate_spike_data([1, 2, 3])
    
    def test_validate_spike_data_empty(self):
        """Test validating empty spike data."""
        with self.assertRaises(ValueError):
            ParameterValidator.validate_spike_data(np.array([]))
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_spike_data(torch.tensor([]))
    
    def test_validate_spike_data_wrong_dimensions(self):
        """Test validating spike data with wrong dimensions."""
        with self.assertRaises(ValueError):
            ParameterValidator.validate_spike_data(np.random.randn(100))
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_spike_data(np.random.randn(100, 200, 300, 400, 500))
    
    def test_validate_image_data_valid(self):
        """Test validating valid image data."""
        # Should not raise exception
        img_np = np.random.randint(0, 255, (250, 400), dtype=np.uint8)
        ParameterValidator.validate_image_data(img_np)
        
        ParameterValidator.validate_image_data(None)
    
    def test_validate_image_data_invalid_type(self):
        """Test validating image data with invalid type."""
        with self.assertRaises(TypeError):
            ParameterValidator.validate_image_data("invalid")
        
        with self.assertRaises(TypeError):
            ParameterValidator.validate_image_data(torch.tensor([1, 2, 3]))
    
    def test_validate_image_data_empty(self):
        """Test validating empty image data."""
        with self.assertRaises(ValueError):
            ParameterValidator.validate_image_data(np.array([]))
    
    def test_validate_image_data_wrong_dimensions(self):
        """Test validating image data with wrong dimensions."""
        with self.assertRaises(ValueError):
            ParameterValidator.validate_image_data(np.random.randn(100, 200, 300, 400))
    
    def test_validate_remove_head_valid(self):
        """Test validating valid remove_head parameter."""
        # Should not raise exception
        ParameterValidator.validate_remove_head(True)
        ParameterValidator.validate_remove_head(False)
    
    def test_validate_remove_head_invalid_type(self):
        """Test validating remove_head with invalid type."""
        with self.assertRaises(TypeError):
            ParameterValidator.validate_remove_head("true")
        
        with self.assertRaises(TypeError):
            ParameterValidator.validate_remove_head(1)
        
        with self.assertRaises(TypeError):
            ParameterValidator.validate_remove_head(None)
    
    def test_validate_infer_from_dataset_params_valid(self):
        """Test validating valid infer_from_dataset parameters."""
        # Should not raise exception
        validate_infer_from_dataset_params(0, 10)
        validate_infer_from_dataset_params(5, 10)
    
    def test_validate_infer_from_dataset_params_invalid(self):
        """Test validating invalid infer_from_dataset parameters."""
        with self.assertRaises(ValueError):
            validate_infer_from_dataset_params(10, 10)
        
        with self.assertRaises(TypeError):
            validate_infer_from_dataset_params("0", 10)
    
    def test_validate_infer_from_file_params_valid(self):
        """Test validating valid infer_from_file parameters."""
        # Should not raise exception
        validate_infer_from_file_params(
            self.test_file, 250, 400, 1.0, self.test_img, False
        )
    
    def test_validate_infer_from_file_params_invalid(self):
        """Test validating invalid infer_from_file parameters."""
        with self.assertRaises(FileNotFoundError):
            validate_infer_from_file_params(
                "nonexistent.dat", 250, 400, 1.0, self.test_img, False
            )
        
        with self.assertRaises(TypeError):
            validate_infer_from_file_params(
                self.test_file, "250", 400, 1.0, self.test_img, False
            )
        
        with self.assertRaises(ValueError):
            validate_infer_from_file_params(
                self.test_file, -250, 400, 1.0, self.test_img, False
            )
    
    def test_validate_infer_from_spk_params_valid(self):
        """Test validating valid infer_from_spk parameters."""
        # Should not raise exception
        spike_np = np.random.randn(200, 250, 400)
        img_np = np.random.randint(0, 255, (250, 400), dtype=np.uint8)
        validate_infer_from_spk_params(spike_np, 1.0, img_np)
        
        # Test with None image
        validate_infer_from_spk_params(spike_np, 1.0, None)
    
    def test_validate_infer_from_spk_params_invalid(self):
        """Test validating invalid infer_from_spk parameters."""
        spike_np = np.random.randn(200, 250, 400)
        
        with self.assertRaises(TypeError):
            validate_infer_from_spk_params("invalid", 1.0, None)
        
        with self.assertRaises(ValueError):
            validate_infer_from_spk_params(np.array([]), 1.0, None)
        
        with self.assertRaises(TypeError):
            validate_infer_from_spk_params(spike_np, "1.0", None)


if __name__ == '__main__':
    unittest.main()