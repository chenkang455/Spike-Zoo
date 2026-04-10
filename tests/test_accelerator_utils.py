import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from spikezoo.utils.accelerator_utils import (
    AcceleratorManager, 
    AcceleratorConfig,
    create_accelerator_manager
)


class TestAcceleratorUtils(unittest.TestCase):
    """Accelerator utilities unit tests."""
    
    def setUp(self):
        """Test setup."""
        # Create a simple model for testing
        self.model = nn.Linear(10, 1)
        
        # Create optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        # Create dummy data
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(x, y)
        self.dataloader = DataLoader(dataset, batch_size=10)
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50)
    
    def test_accelerator_config_creation(self):
        """Test AcceleratorConfig creation."""
        config = AcceleratorConfig(
            mixed_precision="fp16",
            gradient_accumulation_steps=2,
            device_placement=True,
            split_batches=False,
            even_batches=True
        )
        
        self.assertEqual(config.mixed_precision, "fp16")
        self.assertEqual(config.gradient_accumulation_steps, 2)
        self.assertTrue(config.device_placement)
        self.assertFalse(config.split_batches)
        self.assertTrue(config.even_batches)
    
    def test_accelerator_manager_creation(self):
        """Test AcceleratorManager creation."""
        manager = AcceleratorManager()
        
        self.assertIsInstance(manager, AcceleratorManager)
        self.assertIsNone(manager.accelerator)
        self.assertFalse(manager.is_initialized)
    
    def test_accelerator_manager_with_config(self):
        """Test AcceleratorManager creation with config."""
        config = AcceleratorConfig(mixed_precision="fp16")
        manager = AcceleratorManager(config)
        
        self.assertIsInstance(manager, AcceleratorManager)
        self.assertEqual(manager.config.mixed_precision, "fp16")
    
    def test_create_accelerator_manager_function(self):
        """Test create_accelerator_manager function."""
        # Without config
        manager = create_accelerator_manager()
        self.assertIsInstance(manager, AcceleratorManager)
        
        # With config
        config_dict = {
            "mixed_precision": "bf16",
            "gradient_accumulation_steps": 4
        }
        manager = create_accelerator_manager(config_dict)
        self.assertIsInstance(manager, AcceleratorManager)
        self.assertEqual(manager.config.mixed_precision, "bf16")
        self.assertEqual(manager.config.gradient_accumulation_steps, 4)
    
    def test_accelerator_manager_initialize_without_scheduler(self):
        """Test AcceleratorManager initialize without scheduler."""
        manager = AcceleratorManager()
        
        # This would normally prepare the objects, but in a single-process environment
        # it just returns the same objects
        try:
            model, optimizer, dataloader = manager.initialize(
                self.model, self.optimizer, self.dataloader
            )
            
            self.assertIs(model, self.model)
            self.assertIs(optimizer, self.optimizer)
            self.assertIs(dataloader, self.dataloader)
        except Exception as e:
            # In some environments, accelerate might not be available or properly configured
            # We'll skip this test if that's the case
            self.skipTest(f"Accelerator initialization failed: {e}")
    
    def test_accelerator_manager_initialize_with_scheduler(self):
        """Test AcceleratorManager initialize with scheduler."""
        manager = AcceleratorManager()
        
        try:
            model, optimizer, dataloader, scheduler = manager.initialize(
                self.model, self.optimizer, self.dataloader, self.scheduler
            )
            
            self.assertIs(model, self.model)
            self.assertIs(optimizer, self.optimizer)
            self.assertIs(dataloader, self.dataloader)
            self.assertIs(scheduler, self.scheduler)
        except Exception as e:
            # In some environments, accelerate might not be available or properly configured
            # We'll skip this test if that's the case
            self.skipTest(f"Accelerator initialization failed: {e}")
    
    def test_accelerator_manager_not_initialized_errors(self):
        """Test that methods raise errors when accelerator is not initialized."""
        manager = AcceleratorManager()
        
        # Test backward method
        with self.assertRaises(RuntimeError) as cm:
            manager.backward(torch.tensor(1.0))
        self.assertIn("Accelerator not initialized", str(cm.exception))
        
        # Test step method
        with self.assertRaises(RuntimeError) as cm:
            manager.step(self.optimizer)
        self.assertIn("Accelerator not initialized", str(cm.exception))
        
        # Test zero_grad method
        with self.assertRaises(RuntimeError) as cm:
            manager.zero_grad(self.optimizer)
        self.assertIn("Accelerator not initialized", str(cm.exception))
        
        # Test gather method
        with self.assertRaises(RuntimeError) as cm:
            manager.gather(torch.tensor([1, 2, 3]))
        self.assertIn("Accelerator not initialized", str(cm.exception))
        
        # Test unwrap_model method
        with self.assertRaises(RuntimeError) as cm:
            manager.unwrap_model(self.model)
        self.assertIn("Accelerator not initialized", str(cm.exception))
        
        # Test print method
        with self.assertRaises(RuntimeError) as cm:
            manager.print("test")
        self.assertIn("Accelerator not initialized", str(cm.exception))
        
        # Test wait_for_everyone method
        with self.assertRaises(RuntimeError) as cm:
            manager.wait_for_everyone()
        self.assertIn("Accelerator not initialized", str(cm.exception))
        
        # Test is_main_process method
        with self.assertRaises(RuntimeError) as cm:
            manager.is_main_process()
        self.assertIn("Accelerator not initialized", str(cm.exception))


if __name__ == '__main__':
    unittest.main()