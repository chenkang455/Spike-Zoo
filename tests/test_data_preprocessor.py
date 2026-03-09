import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os
from spikezoo.pipeline.data_preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """DataPreprocessor unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.device = "cpu"
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Test cleanup."""
        # Remove temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)
    
    def test_preprocess_dataset_item(self):
        """Test preprocessing dataset item."""
        # Mock batch data
        batch = {
            "spike": torch.randn(200, 250, 400),
            "gt_img": torch.randn(250, 400),
            "rate": 1.0,
            "with_img": True
        }
        
        spike, img, rate = DataPreprocessor.preprocess_dataset_item(batch, self.device)
        
        # Check shapes and types
        self.assertEqual(spike.shape, (1, 200, 250, 400))
        self.assertEqual(img.shape, (1, 1, 250, 400))
        self.assertEqual(rate, 1.0)
        self.assertEqual(spike.device.type, self.device)
        self.assertEqual(img.device.type, self.device)
    
    def test_preprocess_dataset_item_no_img(self):
        """Test preprocessing dataset item without image."""
        # Mock batch data without image
        batch = {
            "spike": torch.randn(200, 250, 400),
            "gt_img": None,
            "rate": 1.0,
            "with_img": False
        }
        
        spike, img, rate = DataPreprocessor.preprocess_dataset_item(batch, self.device)
        
        # Check shapes and types
        self.assertEqual(spike.shape, (1, 200, 250, 400))
        self.assertIsNone(img)
        self.assertEqual(rate, 1.0)
        self.assertEqual(spike.device.type, self.device)
    
    def test_preprocess_spike_input_numpy(self):
        """Test preprocessing spike input from numpy array."""
        # Create mock spike data
        spike_np = np.random.randn(200, 250, 400).astype(np.float32)
        
        spike, img = DataPreprocessor.preprocess_spike_input(spike_np, self.device)
        
        # Check shapes and types
        self.assertEqual(spike.shape, (1, 200, 250, 400))
        self.assertIsNone(img)
        self.assertEqual(spike.device.type, self.device)
    
    def test_preprocess_spike_input_torch(self):
        """Test preprocessing spike input from torch tensor."""
        # Create mock spike data
        spike_torch = torch.randn(200, 250, 400)
        
        spike, img = DataPreprocessor.preprocess_spike_input(spike_torch, self.device)
        
        # Check shapes and types
        self.assertEqual(spike.shape, (1, 200, 250, 400))
        self.assertIsNone(img)
        self.assertEqual(spike.device.type, self.device)
    
    def test_preprocess_spike_input_with_image(self):
        """Test preprocessing spike input with image."""
        # Create mock spike data and image
        spike_np = np.random.randn(200, 250, 400).astype(np.float32)
        img_np = np.random.randint(0, 255, (250, 400, 3), dtype=np.uint8)
        
        spike, img = DataPreprocessor.preprocess_spike_input(spike_np, self.device, img_np)
        
        # Check shapes and types
        self.assertEqual(spike.shape, (1, 200, 250, 400))
        self.assertEqual(img.shape, (1, 1, 250, 400))
        self.assertEqual(spike.device.type, self.device)
        self.assertEqual(img.device.type, self.device)
    
    def test_create_save_folder_dataset(self):
        """Test creating save folder for dataset method."""
        base_folder = Path(self.temp_dir)
        save_folder = DataPreprocessor.create_save_folder(
            base_folder, "dataset", 5, "test_dataset", "test_split"
        )
        
        expected_path = base_folder / "infer_from_dataset" / "test_dataset_dataset" / "test_split" / "000005"
        self.assertEqual(save_folder, expected_path)
        self.assertTrue(os.path.exists(save_folder))
    
    def test_create_save_folder_file(self):
        """Test creating save folder for file method."""
        base_folder = Path(self.temp_dir)
        save_folder = DataPreprocessor.create_save_folder(
            base_folder, "file", "test_file.dat"
        )
        
        expected_path = base_folder / "infer_from_file" / "test_file.dat"
        self.assertEqual(save_folder, expected_path)
        self.assertTrue(os.path.exists(save_folder))
    
    def test_create_save_folder_spk(self):
        """Test creating save folder for spk method."""
        base_folder = Path(self.temp_dir)
        save_folder = DataPreprocessor.create_save_folder(base_folder, "spk")
        
        expected_path = base_folder / "infer_from_spk"
        self.assertEqual(save_folder, expected_path)
        self.assertTrue(os.path.exists(save_folder))


if __name__ == '__main__':
    unittest.main()