import unittest
import tempfile
import os
import yaml
import json
from pathlib import Path
from spikezoo.config.config_manager import ConfigManager, load_config, save_config


class TestConfigManager(unittest.TestCase):
    """ConfigManager unit tests."""
    
    def setUp(self):
        """Test setup."""
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
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration."""
        # Create a test YAML config
        config_data = {
            "version": "local",
            "save_folder": "results",
            "exp_name": "test_exp",
            "save_metric": True,
            "metric_names": ["psnr", "ssim"]
        }
        
        config_path = Path(self.temp_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config
        loaded_config = ConfigManager.load_config(config_path)
        
        # Check if data matches
        self.assertEqual(loaded_config, config_data)
    
    def test_load_json_config(self):
        """Test loading JSON configuration."""
        # Create a test JSON config
        config_data = {
            "version": "local",
            "save_folder": "results",
            "exp_name": "test_exp",
            "save_metric": True,
            "metric_names": ["psnr", "ssim"]
        }
        
        config_path = Path(self.temp_dir) / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        # Load config
        loaded_config = ConfigManager.load_config(config_path)
        
        # Check if data matches
        self.assertEqual(loaded_config, config_data)
    
    def test_load_nonexistent_config(self):
        """Test loading nonexistent configuration."""
        config_path = Path(self.temp_dir) / "nonexistent.yaml"
        
        with self.assertRaises(FileNotFoundError):
            ConfigManager.load_config(config_path)
    
    def test_save_yaml_config(self):
        """Test saving YAML configuration."""
        config_data = {
            "version": "local",
            "save_folder": "results",
            "exp_name": "test_exp",
            "save_metric": True,
            "metric_names": ["psnr", "ssim"]
        }
        
        config_path = Path(self.temp_dir) / "saved_config.yaml"
        ConfigManager.save_config(config_data, config_path, format='yaml')
        
        # Check if file exists
        self.assertTrue(os.path.exists(config_path))
        
        # Load and verify
        with open(config_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        self.assertEqual(saved_data, config_data)
    
    def test_save_json_config(self):
        """Test saving JSON configuration."""
        config_data = {
            "version": "local",
            "save_folder": "results",
            "exp_name": "test_exp",
            "save_metric": True,
            "metric_names": ["psnr", "ssim"]
        }
        
        config_path = Path(self.temp_dir) / "saved_config.json"
        ConfigManager.save_config(config_data, config_path, format='json')
        
        # Check if file exists
        self.assertTrue(os.path.exists(config_path))
        
        # Load and verify
        with open(config_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, config_data)
    
    def test_save_config_with_nested_directories(self):
        """Test saving configuration to nested directories."""
        config_data = {
            "version": "local",
            "save_folder": "results",
            "exp_name": "test_exp"
        }
        
        config_path = Path(self.temp_dir) / "nested" / "directories" / "config.yaml"
        ConfigManager.save_config(config_data, config_path)
        
        # Check if file exists
        self.assertTrue(os.path.exists(config_path))
        
        # Load and verify
        loaded_config = ConfigManager.load_config(config_path)
        self.assertEqual(loaded_config, config_data)
    
    def test_convenience_functions(self):
        """Test convenience load_config and save_config functions."""
        config_data = {
            "version": "local",
            "save_folder": "results",
            "exp_name": "test_exp"
        }
        
        config_path = Path(self.temp_dir) / "convenience_test.yaml"
        
        # Save using convenience function
        save_config(config_data, config_path)
        
        # Load using convenience function
        loaded_config = load_config(config_path)
        
        # Verify data
        self.assertEqual(loaded_config, config_data)


if __name__ == '__main__':
    unittest.main()