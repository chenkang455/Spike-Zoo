import unittest
import sys
import os
import torch
import tempfile
import shutil
from pathlib import Path

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.models.base_model import BaseModel, BaseModelConfig
from spikezoo.models.model_registry import ModelRegistry
from spikezoo.models.modules import (
    network_loader,
    loss_functions,
    metric_utils
)


class TestLossFunctions(unittest.TestCase):
    """Test loss functions module."""
    
    def test_loss_registry_creation(self):
        """Test loss registry creation."""
        registry = loss_functions.LossFunctionRegistry()
        self.assertIsInstance(registry, loss_functions.LossFunctionRegistry)
    
    def test_default_losses(self):
        """Test default loss functions."""
        registry = loss_functions.LossFunctionRegistry()
        losses = registry.list_losses()
        
        self.assertIn("l1", losses)
        self.assertIn("l2", losses)
        self.assertIn("smooth_l1", losses)
    
    def test_register_loss(self):
        """Test registering custom loss function."""
        registry = loss_functions.LossFunctionRegistry()
        
        def custom_loss(x, y):
            return torch.mean((x - y) ** 2)
        
        registry.register_loss("custom", custom_loss)
        losses = registry.list_losses()
        self.assertIn("custom", losses)
        
        # Test retrieving the loss
        loss_func = registry.get_loss_function("custom")
        self.assertEqual(loss_func, custom_loss)
    
    def test_get_loss_function(self):
        """Test getting loss functions."""
        # Test existing loss
        l1_loss = loss_functions.get_loss_function("l1")
        self.assertIsNotNone(l1_loss)
        
        # Test non-existing loss (should return zero loss)
        zero_loss = loss_functions.get_loss_function("nonexistent")
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        result = zero_loss(x, y)
        self.assertEqual(result.item(), 0.0)
    
    def test_compute_loss_dict(self):
        """Test computing loss dictionary."""
        # Create dummy data
        outputs = {"recon_img": torch.randn(2, 3, 32, 32)}
        batch = {"gt_img": torch.randn(2, 3, 32, 32)}
        loss_weight_dict = {"l1": 1.0, "l2": 0.5}
        
        # Compute losses
        loss_dict, loss_values_dict = loss_functions.compute_loss_dict(
            outputs, batch, loss_weight_dict
        )
        
        # Check results
        self.assertIn("l1", loss_dict)
        self.assertIn("l2", loss_dict)
        self.assertIn("l1", loss_values_dict)
        self.assertIn("l2", loss_values_dict)
        self.assertIsInstance(loss_dict["l1"], torch.Tensor)
        self.assertIsInstance(loss_values_dict["l1"], float)


class TestMetricUtils(unittest.TestCase):
    """Test metric utilities module."""
    
    def test_get_paired_images(self):
        """Test getting paired images."""
        # Create dummy data
        batch = {"gt_img": torch.randn(2, 3, 32, 32)}
        outputs = {"recon_img": torch.randn(2, 3, 32, 32)}
        
        # Get paired images
        recon_img, gt_img = metric_utils.get_paired_images(batch, outputs)
        
        self.assertEqual(recon_img.shape, batch["gt_img"].shape)
        self.assertEqual(gt_img.shape, batch["gt_img"].shape)
    
    def test_prepare_visualization_dict(self):
        """Test preparing visualization dictionary."""
        # Create dummy data
        batch = {"gt_img": torch.randn(2, 3, 32, 32)}
        outputs = {"recon_img": torch.randn(2, 3, 32, 32)}
        
        # Prepare visualization dict
        visual_dict = metric_utils.prepare_visualization_dict(batch, outputs)
        
        self.assertIn("recon_img", visual_dict)
        self.assertIn("gt_img", visual_dict)
        self.assertEqual(visual_dict["recon_img"].shape, outputs["recon_img"].shape)
        self.assertEqual(visual_dict["gt_img"].shape, batch["gt_img"].shape)


class TestNetworkLoader(unittest.TestCase):
    """Test network loader module."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_model_weights(self):
        """Test saving model weights."""
        # Create dummy model
        model = torch.nn.Linear(10, 1)
        
        # Save weights
        save_path = os.path.join(self.temp_dir, "test_model.pth")
        network_loader.save_model_weights(model, save_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(save_path))
        
        # Load and verify
        loaded_state_dict = torch.load(save_path)
        self.assertIn("weight", loaded_state_dict)
        self.assertIn("bias", loaded_state_dict)


class TestModularArchitecture(unittest.TestCase):
    """Test the modular architecture implementation."""
    
    def test_base_model_structure(self):
        """Test base model structure."""
        config = BaseModelConfig(model_name="test_model")
        model = BaseModel(config)
        
        self.assertEqual(model.cfg, config)
        self.assertIsNone(model.net)
        self.assertIsNone(model.device)
    
    def test_model_registry_modularity(self):
        """Test model registry modularity."""
        registry = ModelRegistry()
        
        class TestModel(BaseModel):
            def __init__(self, cfg: BaseModelConfig):
                super().__init__(cfg)
        
        class TestConfig(BaseModelConfig):
            pass
        
        # Register model
        registry.register_model("test_model", TestModel, TestConfig)
        
        # Check registration
        self.assertTrue(registry.is_model_registered("test_model"))
        
        # Create model
        model = registry.create_model("test_model")
        self.assertIsInstance(model, TestModel)
        
        # List models
        models = registry.list_models()
        self.assertIn("test_model", models)
    
    def test_module_imports(self):
        """Test that all modules can be imported."""
        # These imports should work without errors
        from spikezoo.models import (
            BaseModel,
            BaseModelConfig,
            ModelRegistry
        )
        
        from spikezoo.models.modules import (
            network_loader,
            loss_functions,
            metric_utils
        )
        
        # Check that modules are properly imported
        self.assertIsNotNone(BaseModel)
        self.assertIsNotNone(network_loader)
        self.assertIsNotNone(loss_functions)
        self.assertIsNotNone(metric_utils)


if __name__ == '__main__':
    unittest.main()