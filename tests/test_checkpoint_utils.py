import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import os
from pathlib import Path
from spikezoo.utils.checkpoint_utils import CheckpointManager, create_checkpoint_manager


class TestCheckpointUtils(unittest.TestCase):
    """Checkpoint utilities unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple model for testing
        self.model = nn.Linear(10, 1)
        
        # Create optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50)
        
        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.temp_dir)
    
    def tearDown(self):
        """Test cleanup."""
        # Remove temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)
    
    def test_checkpoint_manager_creation(self):
        """Test CheckpointManager creation."""
        manager = CheckpointManager(self.temp_dir)
        
        self.assertIsInstance(manager, CheckpointManager)
        self.assertEqual(manager.save_dir, Path(self.temp_dir))
        self.assertTrue(manager.save_dir.exists())
    
    def test_create_checkpoint_manager_function(self):
        """Test create_checkpoint_manager function."""
        manager = create_checkpoint_manager(self.temp_dir)
        
        self.assertIsInstance(manager, CheckpointManager)
        self.assertEqual(manager.save_dir, Path(self.temp_dir))
    
    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        epoch = 5
        step = 100
        best_metric = 0.95
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, self.scheduler, epoch, step, best_metric
        )
        
        # Check if checkpoint file exists
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Check filename format
        filename = Path(checkpoint_path).name
        self.assertIn(f"checkpoint_epoch_{epoch:06d}_step_{step:06d}", filename)
    
    def test_save_checkpoint_without_scheduler(self):
        """Test saving checkpoint without scheduler."""
        epoch = 5
        step = 100
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, None, epoch, step
        )
        
        # Check if checkpoint file exists
        self.assertTrue(os.path.exists(checkpoint_path))
    
    def test_save_checkpoint_with_additional_state(self):
        """Test saving checkpoint with additional state."""
        epoch = 5
        step = 100
        additional_state = {
            'custom_value': 42,
            'custom_dict': {'key': 'value'}
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, self.scheduler, epoch, step,
            additional_state=additional_state
        )
        
        # Check if checkpoint file exists
        self.assertTrue(os.path.exists(checkpoint_path))
    
    def test_save_checkpoint_with_custom_filename(self):
        """Test saving checkpoint with custom filename."""
        epoch = 5
        step = 100
        filename = "custom_checkpoint.pth"
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, self.scheduler, epoch, step,
            filename=filename
        )
        
        # Check if checkpoint file exists with custom name
        self.assertTrue(os.path.exists(checkpoint_path))
        self.assertEqual(Path(checkpoint_path).name, filename)
    
    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        epoch = 5
        step = 100
        best_metric = 0.95
        
        # Save checkpoint first
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, self.scheduler, epoch, step, best_metric
        )
        
        # Create new model, optimizer, and scheduler for loading
        new_model = nn.Linear(10, 1)
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=50)
        
        # Load checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(
            new_model, new_optimizer, new_scheduler, checkpoint_path
        )
        
        # Check checkpoint contents
        self.assertEqual(checkpoint['epoch'], epoch)
        self.assertEqual(checkpoint['step'], step)
        self.assertEqual(checkpoint['best_metric'], best_metric)
        
        # Check that model, optimizer, and scheduler states are loaded
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
        self.assertIn('scheduler_state_dict', checkpoint)
    
    def test_load_checkpoint_without_scheduler(self):
        """Test loading checkpoint without scheduler."""
        epoch = 5
        step = 100
        
        # Save checkpoint first
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, None, epoch, step
        )
        
        # Create new model and optimizer for loading
        new_model = nn.Linear(10, 1)
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)
        
        # Load checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(
            new_model, new_optimizer, None, checkpoint_path
        )
        
        # Check checkpoint contents
        self.assertEqual(checkpoint['epoch'], epoch)
        self.assertEqual(checkpoint['step'], step)
        
        # Check that model and optimizer states are loaded
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
    
    def test_load_checkpoint_with_filename(self):
        """Test loading checkpoint with filename."""
        epoch = 5
        step = 100
        filename = "test_checkpoint.pth"
        
        # Save checkpoint first
        self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, self.scheduler, epoch, step,
            filename=filename
        )
        
        # Create new model, optimizer, and scheduler for loading
        new_model = nn.Linear(10, 1)
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=50)
        
        # Load checkpoint using filename
        checkpoint = self.checkpoint_manager.load_checkpoint(
            new_model, new_optimizer, new_scheduler, filename=filename
        )
        
        # Check checkpoint contents
        self.assertEqual(checkpoint['epoch'], epoch)
        self.assertEqual(checkpoint['step'], step)
    
    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint."""
        with self.assertRaises(FileNotFoundError):
            self.checkpoint_manager.load_checkpoint(
                self.model, self.optimizer, self.scheduler, "nonexistent.pth"
            )
    
    def test_save_and_load_best_model(self):
        """Test saving and loading best model."""
        metric_value = 0.98
        
        # Save best model
        save_path = self.checkpoint_manager.save_best_model(self.model, metric_value)
        
        # Check if file exists
        self.assertTrue(os.path.exists(save_path))
        
        # Create new model for loading
        new_model = nn.Linear(10, 1)
        
        # Load best model
        checkpoint = self.checkpoint_manager.load_best_model(new_model)
        
        # Check checkpoint contents
        self.assertEqual(checkpoint['metric_value'], metric_value)
        self.assertIn('model_state_dict', checkpoint)
    
    def test_load_nonexistent_best_model(self):
        """Test loading nonexistent best model."""
        new_model = nn.Linear(10, 1)
        
        with self.assertRaises(FileNotFoundError):
            self.checkpoint_manager.load_best_model(new_model, "nonexistent_best.pth")
    
    def test_get_latest_checkpoint(self):
        """Test getting latest checkpoint."""
        # Save multiple checkpoints
        self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, self.scheduler, 1, 10)
        self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, self.scheduler, 2, 20)
        self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, self.scheduler, 3, 30)
        
        # Get latest checkpoint
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        
        # Should return the last saved checkpoint
        self.assertIsNotNone(latest_checkpoint)
        
        # Load and check epoch
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        self.assertEqual(checkpoint['epoch'], 3)
        self.assertEqual(checkpoint['step'], 30)
    
    def test_get_latest_checkpoint_no_checkpoints(self):
        """Test getting latest checkpoint when no checkpoints exist."""
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        self.assertIsNone(latest_checkpoint)
    
    def test_get_checkpoints_sorted(self):
        """Test getting sorted checkpoints."""
        # Save multiple checkpoints in different order
        self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, self.scheduler, 3, 30)
        self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, self.scheduler, 1, 10)
        self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, self.scheduler, 2, 20)
        
        # Get sorted checkpoints
        sorted_checkpoints = self.checkpoint_manager.get_checkpoints_sorted()
        
        # Should have 3 checkpoints
        self.assertEqual(len(sorted_checkpoints), 3)
        
        # Load and check epochs are in order
        epochs = []
        for checkpoint_path in sorted_checkpoints:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            epochs.append(checkpoint['epoch'])
        
        # Should be sorted by epoch
        self.assertEqual(epochs, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()