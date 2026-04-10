import unittest
import sys
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.utils.visualization_utils import (
    VisualizationConfig,
    BaseVisualizer,
    TensorBoardVisualizer,
    WandBVisualizer,
    MatplotlibVisualizer,
    VisualizationManager,
    get_visualization_manager,
    setup_visualization,
    log_scalar,
    log_image,
    log_histogram,
    log_text,
    log_config,
    flush_visualization,
    close_visualization
)


class TestVisualizationConfig(unittest.TestCase):
    """VisualizationConfig unit tests."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VisualizationConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.experiment_name, "experiment")
        self.assertEqual(config.log_dir, "logs")
        self.assertEqual(config.save_frequency, 100)
        self.assertTrue(config.tensorboard_enabled)
        self.assertFalse(config.wandb_enabled)
        self.assertTrue(config.plot_enabled)
        self.assertEqual(config.plot_format, "png")
        self.assertEqual(config.plot_dpi, 100)
        self.assertIn("loss", config.metrics_to_track)
        self.assertIn("accuracy", config.metrics_to_track)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = VisualizationConfig(
            enabled=False,
            experiment_name="test_exp",
            log_dir="/tmp/logs",
            save_frequency=50,
            tensorboard_enabled=False,
            wandb_enabled=True,
            wandb_project="test_project",
            plot_enabled=False,
            metrics_to_track=["custom_metric"],
            images_to_track=["custom_image"]
        )
        
        self.assertFalse(config.enabled)
        self.assertEqual(config.experiment_name, "test_exp")
        self.assertEqual(config.log_dir, "/tmp/logs")
        self.assertEqual(config.save_frequency, 50)
        self.assertFalse(config.tensorboard_enabled)
        self.assertTrue(config.wandb_enabled)
        self.assertEqual(config.wandb_project, "test_project")
        self.assertFalse(config.plot_enabled)
        self.assertIn("custom_metric", config.metrics_to_track)
        self.assertIn("custom_image", config.images_to_track)


class TestBaseVisualizer(unittest.TestCase):
    """BaseVisualizer unit tests."""
    
    def test_base_visualizer_abstract_methods(self):
        """Test that BaseVisualizer raises NotImplementedError for abstract methods."""
        config = VisualizationConfig()
        visualizer = BaseVisualizer(config)
        
        with self.assertRaises(NotImplementedError):
            visualizer.initialize()
        
        with self.assertRaises(NotImplementedError):
            visualizer.log_scalar("test", 1.0, 1)
        
        with self.assertRaises(NotImplementedError):
            visualizer.log_image("test", np.array([1, 2, 3]), 1)
        
        with self.assertRaises(NotImplementedError):
            visualizer.log_histogram("test", np.array([1, 2, 3]), 1)
        
        with self.assertRaises(NotImplementedError):
            visualizer.log_text("test", "text", 1)
        
        with self.assertRaises(NotImplementedError):
            visualizer.log_config({"test": "config"})


class TestTensorBoardVisualizer(unittest.TestCase):
    """TensorBoardVisualizer unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_tensorboard_visualizer_initialization(self):
        """Test TensorBoardVisualizer initialization."""
        config = VisualizationConfig(
            log_dir=self.temp_dir,
            tensorboard_enabled=True
        )
        
        # This will fail gracefully if TensorBoard is not available
        visualizer = TensorBoardVisualizer(config)
        visualizer.initialize()
        
        # Check that it handled the initialization properly
        # If TensorBoard is available, it should be initialized
        # If not, it should have disabled itself
        self.assertIsInstance(visualizer.is_initialized, bool)
    
    def test_tensorboard_visualizer_log_methods(self):
        """Test TensorBoardVisualizer log methods."""
        config = VisualizationConfig(
            log_dir=self.temp_dir,
            tensorboard_enabled=True
        )
        
        visualizer = TensorBoardVisualizer(config)
        
        # These should not raise exceptions even if TensorBoard is not available
        visualizer.log_scalar("test_scalar", 1.0, 1)
        visualizer.log_image("test_image", np.random.rand(10, 10), 1)
        visualizer.log_histogram("test_histogram", np.random.rand(100), 1)
        visualizer.log_text("test_text", "test message", 1)
        visualizer.log_config({"test": "config"})
    
    def test_tensorboard_visualizer_flush_close(self):
        """Test TensorBoardVisualizer flush and close methods."""
        config = VisualizationConfig(
            log_dir=self.temp_dir,
            tensorboard_enabled=True
        )
        
        visualizer = TensorBoardVisualizer(config)
        
        # These should not raise exceptions
        visualizer.flush()
        visualizer.close()


class TestWandBVisualizer(unittest.TestCase):
    """WandBVisualizer unit tests."""
    
    def test_wandb_visualizer_initialization(self):
        """Test WandBVisualizer initialization."""
        config = VisualizationConfig(
            wandb_enabled=True,
            wandb_project="test_project"
        )
        
        # This will fail gracefully if WandB is not available
        visualizer = WandBVisualizer(config)
        visualizer.initialize()
        
        # Check that it handled the initialization properly
        # If WandB is available, it should be initialized
        # If not, it should have disabled itself
        self.assertIsInstance(visualizer.is_initialized, bool)
    
    def test_wandb_visualizer_log_methods(self):
        """Test WandBVisualizer log methods."""
        config = VisualizationConfig(
            wandb_enabled=True,
            wandb_project="test_project"
        )
        
        visualizer = WandBVisualizer(config)
        
        # These should not raise exceptions even if WandB is not available
        visualizer.log_scalar("test_scalar", 1.0, 1)
        visualizer.log_image("test_image", np.random.rand(10, 10), 1)
        visualizer.log_histogram("test_histogram", np.random.rand(100), 1)
        visualizer.log_text("test_text", "test message", 1)
        visualizer.log_config({"test": "config"})
    
    def test_wandb_visualizer_close(self):
        """Test WandBVisualizer close method."""
        config = VisualizationConfig(
            wandb_enabled=True,
            wandb_project="test_project"
        )
        
        visualizer = WandBVisualizer(config)
        
        # This should not raise exceptions
        visualizer.close()


class TestMatplotlibVisualizer(unittest.TestCase):
    """MatplotlibVisualizer unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_matplotlib_visualizer_initialization(self):
        """Test MatplotlibVisualizer initialization."""
        config = VisualizationConfig(
            log_dir=self.temp_dir,
            plot_enabled=True
        )
        
        # This will fail gracefully if matplotlib is not available
        visualizer = MatplotlibVisualizer(config)
        visualizer.initialize()
        
        # Check that it handled the initialization properly
        # If matplotlib is available, it should be initialized
        # If not, it should have disabled itself
        self.assertIsInstance(visualizer.is_initialized, bool)
    
    def test_matplotlib_visualizer_close(self):
        """Test MatplotlibVisualizer close method."""
        config = VisualizationConfig(
            log_dir=self.temp_dir,
            plot_enabled=True
        )
        
        visualizer = MatplotlibVisualizer(config)
        
        # This should not raise exceptions
        visualizer.close()


class TestVisualizationManager(unittest.TestCase):
    """VisualizationManager unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_visualization_manager_initialization(self):
        """Test VisualizationManager initialization."""
        config = VisualizationConfig(
            log_dir=self.temp_dir,
            tensorboard_enabled=True,
            wandb_enabled=False,  # Disable WandB to avoid authentication issues
            plot_enabled=True
        )
        
        manager = VisualizationManager(config)
        
        # Check that manager was created
        self.assertIsInstance(manager, VisualizationManager)
        self.assertEqual(manager.config, config)
    
    def test_visualization_manager_log_methods(self):
        """Test VisualizationManager log methods."""
        config = VisualizationConfig(
            log_dir=self.temp_dir,
            tensorboard_enabled=True,
            wandb_enabled=False,
            plot_enabled=True
        )
        
        manager = VisualizationManager(config)
        
        # These should not raise exceptions
        manager.log_scalar("test_scalar", 1.0, 1)
        manager.log_image("test_image", np.random.rand(10, 10), 1)
        manager.log_histogram("test_histogram", np.random.rand(100), 1)
        manager.log_text("test_text", "test message", 1)
        manager.log_config({"test": "config"})
        
        # Test with auto step counting
        manager.log_scalar("auto_step", 2.0)
        manager.log_image("auto_step", np.random.rand(5, 5))
        
        # Flush and close
        manager.flush()
        manager.close()
    
    def test_visualization_manager_disabled(self):
        """Test VisualizationManager with disabled visualization."""
        config = VisualizationConfig(
            enabled=False,
            log_dir=self.temp_dir,
            tensorboard_enabled=True,
            wandb_enabled=True,
            plot_enabled=True
        )
        
        manager = VisualizationManager(config)
        
        # These should not raise exceptions even with visualization disabled
        manager.log_scalar("test", 1.0, 1)
        manager.log_image("test", np.random.rand(10, 10), 1)
        manager.log_histogram("test", np.random.rand(100), 1)
        manager.log_text("test", "test message", 1)
        manager.log_config({"test": "config"})
        
        # Flush and close
        manager.flush()
        manager.close()


class TestGlobalFunctions(unittest.TestCase):
    """Test global visualization functions."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
        # Clear global manager
        global _visualization_manager
        _visualization_manager = None
    
    def tearDown(self):
        """Test cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clear global manager
        global _visualization_manager
        _visualization_manager = None
    
    def test_get_visualization_manager(self):
        """Test get_visualization_manager function."""
        # First call should create new manager
        config = VisualizationConfig(log_dir=self.temp_dir)
        manager1 = get_visualization_manager(config)
        self.assertIsInstance(manager1, VisualizationManager)
        
        # Second call should return same instance
        manager2 = get_visualization_manager()
        self.assertIs(manager1, manager2)
    
    def test_setup_visualization(self):
        """Test setup_visualization function."""
        config1 = VisualizationConfig(
            experiment_name="test1",
            log_dir=self.temp_dir
        )
        
        # Setup first manager
        setup_visualization(config1)
        manager1 = get_visualization_manager()
        
        # Setup second manager (should close first)
        config2 = VisualizationConfig(
            experiment_name="test2",
            log_dir=self.temp_dir
        )
        setup_visualization(config2)
        manager2 = get_visualization_manager()
        
        # Should be different instances
        self.assertIsNot(manager1, manager2)
    
    def test_convenience_functions(self):
        """Test convenience logging functions."""
        config = VisualizationConfig(
            log_dir=self.temp_dir,
            tensorboard_enabled=True,
            wandb_enabled=False,
            plot_enabled=True
        )
        
        # Setup manager
        setup_visualization(config)
        
        # These should not raise exceptions
        log_scalar("test_scalar", 1.0, 1)
        log_image("test_image", np.random.rand(10, 10), 1)
        log_histogram("test_histogram", np.random.rand(100), 1)
        log_text("test_text", "test message", 1)
        log_config({"test": "config"})
        flush_visualization()
        close_visualization()


class TestIntegration(unittest.TestCase):
    """Integration tests for visualization system."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_visualization_workflow(self):
        """Test full visualization workflow."""
        # Create configuration
        config = VisualizationConfig(
            experiment_name="integration_test",
            log_dir=self.temp_dir,
            tensorboard_enabled=True,
            wandb_enabled=False,  # Disable to avoid authentication
            plot_enabled=True,
            save_frequency=10
        )
        
        # Create manager
        manager = VisualizationManager(config)
        
        # Simulate training loop
        for epoch in range(20):
            # Log training metrics
            train_loss = 1.0 * np.exp(-epoch * 0.2) + np.random.normal(0, 0.1)
            manager.log_scalar("train/loss", train_loss, epoch)
            
            val_loss = 1.0 * np.exp(-epoch * 0.15) + np.random.normal(0, 0.1)
            manager.log_scalar("val/loss", val_loss, epoch)
            
            # Log images periodically
            if epoch % 5 == 0:
                sample_img = np.random.rand(32, 32)
                manager.log_image("samples/input", sample_img, epoch)
            
            # Log histograms
            if epoch % 10 == 0:
                weights = np.random.normal(0, 1.0 * np.exp(-epoch * 0.05), 1000)
                manager.log_histogram("weights/layer1", weights, epoch)
        
        # Log configuration
        manager.log_config({
            "model": "TestModel",
            "epochs": 20,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss
        })
        
        # Log text
        manager.log_text("summary", f"Training completed. Final val loss: {val_loss:.4f}", 20)
        
        # Flush and close
        manager.flush()
        manager.close()
        
        # Check that log directory was created
        log_path = Path(self.temp_dir) / "tensorboard" / "integration_test"
        # Note: Actual file creation depends on whether TensorBoard is available
    
    def test_multiple_managers_isolation(self):
        """Test that multiple managers are isolated."""
        # Create first manager
        config1 = VisualizationConfig(
            experiment_name="manager1",
            log_dir=self.temp_dir,
            tensorboard_enabled=True,
            wandb_enabled=False
        )
        manager1 = VisualizationManager(config1)
        
        # Create second manager
        config2 = VisualizationConfig(
            experiment_name="manager2",
            log_dir=self.temp_dir,
            tensorboard_enabled=True,
            wandb_enabled=False
        )
        manager2 = VisualizationManager(config2)
        
        # Log different data to each manager
        manager1.log_scalar("test", 1.0, 1)
        manager2.log_scalar("test", 2.0, 1)
        
        # Close managers
        manager1.close()
        manager2.close()
        
        # Should be able to close without interference
        self.assertTrue(True)  # If we get here without exceptions, test passes


if __name__ == '__main__':
    unittest.main()