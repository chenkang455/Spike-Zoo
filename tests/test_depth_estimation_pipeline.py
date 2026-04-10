"""
Unit tests for depth estimation pipelines in SpikeZoo.
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

# Add the spikezoo package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spikezoo.pipeline.depth_estimation_pipeline import (
    DepthEstimationPipeline, DepthEstimationTrainPipeline,
    DepthEstimationPipelineConfig, DepthEstimationTrainPipelineConfig
)
from spikezoo.models.depth_estimation_model import DepthEstimationModelConfig
from spikezoo.datasets.depth_estimation_dataset import DepthEstimationDatasetConfig
from spikezoo.utils.optimizer_utils import AdamOptimizerConfig
from spikezoo.utils.scheduler_utils import CosineAnnealingLRConfig


class TestDepthEstimationPipelineConfig(unittest.TestCase):
    """Test depth estimation pipeline configurations."""
    
    def test_inference_pipeline_config(self):
        """Test depth estimation inference pipeline configuration."""
        cfg = DepthEstimationPipelineConfig()
        
        # Check default values
        self.assertEqual(cfg._mode, "depth_estimation_mode")
        self.assertIn("abs_rel", cfg.metric_names)
        self.assertIn("sq_rel", cfg.metric_names)
        self.assertIn("rmse", cfg.metric_names)
        self.assertIn("accuracy", cfg.metric_names)
    
    def test_training_pipeline_config(self):
        """Test depth estimation training pipeline configuration."""
        cfg = DepthEstimationTrainPipelineConfig()
        
        # Check default values
        self.assertEqual(cfg._mode, "depth_estimation_train_mode")
        self.assertEqual(cfg.steps_per_log_metrics, 100)
        self.assertEqual(cfg.steps_per_save_depth_vis, 500)
        self.assertIn("l1_loss", cfg.loss_weight_dict)
        self.assertIn("l2_loss", cfg.loss_weight_dict)
        self.assertEqual(cfg.loss_weight_dict["l1_loss"], 1.0)
        self.assertEqual(cfg.loss_weight_dict["l2_loss"], 0.5)
        self.assertIn("abs_rel", cfg.metric_names)
        self.assertIn("accuracy", cfg.metric_names)
    
    def test_custom_configurations(self):
        """Test custom pipeline configurations."""
        # Test inference config with custom values
        inf_cfg = DepthEstimationPipelineConfig(
            version="v010",
            save_folder="./custom_results",
            save_metric=False,
            metric_names=["rmse"]
        )
        
        self.assertEqual(inf_cfg.version, "v010")
        self.assertEqual(inf_cfg.save_folder, "./custom_results")
        self.assertFalse(inf_cfg.save_metric)
        self.assertEqual(inf_cfg.metric_names, ["rmse"])
        
        # Test training config with custom values
        train_cfg = DepthEstimationTrainPipelineConfig(
            epochs=100,
            bs_train=16,
            steps_per_log_metrics=50,
            loss_weight_dict={"l1_loss": 2.0, "l2_loss": 1.0},
            metric_names=["rmse", "accuracy"]
        )
        
        self.assertEqual(train_cfg.epochs, 100)
        self.assertEqual(train_cfg.bs_train, 16)
        self.assertEqual(train_cfg.steps_per_log_metrics, 50)
        self.assertEqual(train_cfg.loss_weight_dict["l1_loss"], 2.0)
        self.assertEqual(train_cfg.loss_weight_dict["l2_loss"], 1.0)
        self.assertIn("accuracy", train_cfg.metric_names)


class TestDepthEstimationPipeline(unittest.TestCase):
    """Test depth estimation inference pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create basic configurations
        self.pipeline_cfg = DepthEstimationPipelineConfig(
            version="local",
            save_folder=self.temp_dir,
            save_metric=True,
            save_img=True,
            bs_test=1
        )
        
        self.model_cfg = DepthEstimationModelConfig(
            model_name="depth_estimation",
            network_type="basic",
            input_channels=3,
            output_channels=1,
            hidden_dim=16,
            num_layers=2,
            load_state=False
        )
        
        self.dataset_cfg = DepthEstimationDatasetConfig(
            dataset_name="depth_estimation",
            data_root=self.temp_dir,
            split="test",
            height=32,
            width=32,
            sequence_length=3,
            augment=False
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test depth estimation pipeline initialization."""
        # Mock the model and dataset classes to avoid complex setup
        with patch('spikezoo.models.depth_estimation_model.DepthEstimationModel') as mock_model, \
             patch('spikezoo.datasets.depth_estimation_dataset.DepthEstimationDataset') as mock_dataset:
            
            # Setup mocks
            mock_model_instance = Mock()
            mock_model_instance.net = Mock()
            mock_model.return_value = mock_model_instance
            
            mock_dataset_instance = Mock()
            mock_dataset_instance.net = Mock()
            mock_dataset.return_value = mock_dataset_instance
            
            # Create pipeline
            pipeline = DepthEstimationPipeline(
                self.pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Verify pipeline was created
            self.assertIsInstance(pipeline, DepthEstimationPipeline)
            self.assertEqual(pipeline.cfg, self.pipeline_cfg)
    
    def test_pipeline_setup_model_data(self):
        """Test pipeline model and data setup."""
        # Mock the required classes
        with patch('spikezoo.models.depth_estimation_model.DepthEstimationModel') as mock_model, \
             patch('spikezoo.datasets.depth_estimation_dataset.DepthEstimationDataset') as mock_dataset, \
             patch('spikezoo.datasets.build_dataloader') as mock_dataloader:
            
            # Setup mocks
            mock_model_instance = Mock()
            mock_model_instance.net = Mock()
            mock_model_instance.build_network = Mock()
            mock_model.return_value = mock_model_instance
            
            mock_dataset_instance = Mock()
            mock_dataset_instance.build_source = Mock()
            mock_dataset.return_value = mock_dataset_instance
            
            mock_dataloader.return_value = Mock()
            
            # Create pipeline
            pipeline = DepthEstimationPipeline(
                self.pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Verify setup methods were called
            mock_model.assert_called_once()
            mock_dataset.assert_called_once()
            mock_dataset_instance.build_source.assert_called_once_with(split="test")
            mock_dataloader.assert_called_once()
    
    def test_move_batch_to_device(self):
        """Test moving batch to device."""
        # Create pipeline instance
        with patch('spikezoo.models.depth_estimation_model.DepthEstimationModel'), \
             patch('spikezoo.datasets.depth_estimation_dataset.DepthEstimationDataset'), \
             patch('spikezoo.datasets.build_dataloader'):
            
            pipeline = DepthEstimationPipeline(
                self.pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Set device
            pipeline.device = "cpu"
            
            # Create test batch
            batch = {
                "events": torch.randn(2, 2, 32, 32),
                "depth_gt": torch.randn(2, 1, 32, 32),
                "image": torch.randn(2, 3, 32, 32),
                "metadata": "some_string"  # Non-tensor data
            }
            
            # Test moving to device
            moved_batch = pipeline._move_batch_to_device(batch)
            
            # Check tensor data was moved
            self.assertTrue(moved_batch["events"].device.type == "cpu")
            self.assertTrue(moved_batch["depth_gt"].device.type == "cpu")
            self.assertTrue(moved_batch["image"].device.type == "cpu")
            # Check non-tensor data unchanged
            self.assertEqual(moved_batch["metadata"], "some_string")
    
    def test_compute_batch_metrics(self):
        """Test computing batch metrics."""
        # Create pipeline instance
        with patch('spikezoo.models.depth_estimation_model.DepthEstimationModel'), \
             patch('spikezoo.datasets.depth_estimation_dataset.DepthEstimationDataset'), \
             patch('spikezoo.datasets.build_dataloader'):
            
            pipeline = DepthEstimationPipeline(
                self.pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Create test data
            outputs = {
                "depth_pred": torch.randn(2, 1, 32, 32)
            }
            
            batch = {
                "depth_gt": torch.randn(2, 1, 32, 32)
            }
            
            # Test metrics computation
            metrics = pipeline._compute_batch_metrics(outputs, batch)
            
            # Check metrics keys
            self.assertIn("abs_rel", metrics)
            self.assertIn("sq_rel", metrics)
            self.assertIn("rmse", metrics)
            self.assertIn("accuracy", metrics)
            
            # Check metric values are reasonable
            self.assertGreaterEqual(metrics["abs_rel"], 0)
            self.assertGreaterEqual(metrics["sq_rel"], 0)
            self.assertGreaterEqual(metrics["rmse"], 0)
            self.assertGreaterEqual(metrics["accuracy"], 0)
            self.assertLessEqual(metrics["accuracy"], 1)
    
    def test_save_metrics(self):
        """Test saving metrics."""
        # Create pipeline instance
        with patch('spikezoo.models.depth_estimation_model.DepthEstimationModel'), \
             patch('spikezoo.datasets.depth_estimation_dataset.DepthEstimationDataset'), \
             patch('spikezoo.datasets.build_dataloader'):
            
            pipeline = DepthEstimationPipeline(
                self.pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Create test metrics
            metrics = {
                "abs_rel": 0.123456,
                "sq_rel": 0.234567,
                "rmse": 0.345678,
                "accuracy": 0.876543
            }
            
            # Test saving metrics
            pipeline._save_metrics(metrics)
            
            # Check metrics file was created
            metrics_file = Path(pipeline.save_folder) / "metrics.txt"
            self.assertTrue(metrics_file.exists())
            
            # Check file contents
            with open(metrics_file, 'r') as f:
                content = f.read()
                self.assertIn("abs_rel: 0.123456", content)
                self.assertIn("sq_rel: 0.234567", content)
                self.assertIn("rmse: 0.345678", content)
                self.assertIn("accuracy: 0.876543", content)


class TestDepthEstimationTrainPipeline(unittest.TestCase):
    """Test depth estimation training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create basic configurations
        self.train_pipeline_cfg = DepthEstimationTrainPipelineConfig(
            version="local",
            save_folder=self.temp_dir,
            epochs=2,
            bs_train=1,
            bs_test=1,
            nw_train=0,
            nw_test=0,
            steps_per_save_ckpt=1,
            steps_per_cal_metrics=1,
            steps_per_log_metrics=1,
            steps_per_save_depth_vis=1,
            optimizer_cfg=AdamOptimizerConfig(lr=1e-4),
            scheduler_cfg=CosineAnnealingLRConfig(T_max=2, eta_min=1e-6),
            loss_weight_dict={"l1_loss": 1.0, "l2_loss": 0.5},
            enable_visualization=False,
            enable_checkpoint=True,
            seed=42
        )
        
        self.model_cfg = DepthEstimationModelConfig(
            model_name="depth_estimation",
            network_type="basic",
            input_channels=3,
            output_channels=1,
            hidden_dim=16,
            num_layers=2,
            load_state=False
        )
        
        self.dataset_cfg = DepthEstimationDatasetConfig(
            dataset_name="depth_estimation",
            data_root=self.temp_dir,
            split="train",
            height=32,
            width=32,
            sequence_length=3,
            augment=False
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_train_pipeline_initialization(self):
        """Test depth estimation training pipeline initialization."""
        # Mock the required classes
        with patch('spikezoo.models.depth_estimation_model.DepthEstimationModel') as mock_model, \
             patch('spikezoo.datasets.depth_estimation_dataset.DepthEstimationDataset') as mock_dataset, \
             patch('torch.utils.data.DataLoader') as mock_dataloader:
            
            # Setup mocks
            mock_model_instance = Mock()
            mock_model_instance.net = Mock()
            mock_model_instance.build_network = Mock()
            mock_model.return_value = mock_model_instance
            
            mock_dataset_instance = Mock()
            mock_dataset_instance.build_source = Mock()
            mock_dataset.return_value = mock_dataset_instance
            
            mock_dataloader_instance = Mock()
            mock_dataloader.return_value = mock_dataloader_instance
            
            # Create training pipeline
            train_pipeline = DepthEstimationTrainPipeline(
                self.train_pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Verify pipeline was created
            self.assertIsInstance(train_pipeline, DepthEstimationTrainPipeline)
            self.assertEqual(train_pipeline.cfg, self.train_pipeline_cfg)
    
    def test_train_pipeline_setup(self):
        """Test training pipeline setup."""
        # Mock the required classes
        with patch('spikezoo.models.depth_estimation_model.DepthEstimationModel') as mock_model, \
             patch('spikezoo.datasets.depth_estimation_dataset.DepthEstimationDataset') as mock_dataset, \
             patch('torch.utils.data.DataLoader') as mock_dataloader, \
             patch('spikezoo.utils.optimizer_utils.create_optimizer') as mock_optimizer, \
             patch('spikezoo.utils.scheduler_utils.create_scheduler') as mock_scheduler:
            
            # Setup mocks
            mock_model_instance = Mock()
            mock_model_instance.net = Mock()
            mock_model_instance.build_network = Mock()
            mock_model.return_value = mock_model_instance
            
            mock_dataset_instance = Mock()
            mock_dataset_instance.build_source = Mock()
            mock_dataset.return_value = mock_dataset_instance
            
            mock_dataloader_instance = Mock()
            mock_dataloader.return_value = mock_dataloader_instance
            
            mock_optimizer_instance = Mock()
            mock_optimizer.return_value = mock_optimizer_instance
            
            mock_scheduler_instance = Mock()
            mock_scheduler.return_value = mock_scheduler_instance
            
            # Create training pipeline
            train_pipeline = DepthEstimationTrainPipeline(
                self.train_pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Verify setup methods were called
            mock_model.assert_called_once()
            mock_dataset.assert_called()
            mock_optimizer.assert_called_once()
            mock_scheduler.assert_called_once()
    
    def test_create_optimizer(self):
        """Test optimizer creation."""
        # Mock the required classes
        with patch('spikezoo.models.depth_estimation_model.DepthEstimationModel') as mock_model, \
             patch('spikezoo.datasets.depth_estimation_dataset.DepthEstimationDataset') as mock_dataset, \
             patch('torch.utils.data.DataLoader') as mock_dataloader, \
             patch('spikezoo.utils.optimizer_utils.create_optimizer') as mock_create_optimizer:
            
            # Setup mocks
            mock_model_instance = Mock()
            mock_model_instance.net = Mock()
            mock_model_instance.net.parameters = Mock(return_value=[])
            mock_model_instance.build_network = Mock()
            mock_model.return_value = mock_model_instance
            
            mock_dataset_instance = Mock()
            mock_dataset_instance.build_source = Mock()
            mock_dataset.return_value = mock_dataset_instance
            
            mock_dataloader_instance = Mock()
            mock_dataloader.return_value = mock_dataloader_instance
            
            mock_optimizer_instance = Mock()
            mock_create_optimizer.return_value = mock_optimizer_instance
            
            # Create training pipeline
            train_pipeline = DepthEstimationTrainPipeline(
                self.train_pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Test optimizer creation
            optimizer = train_pipeline._create_optimizer()
            
            # Verify optimizer was created
            self.assertEqual(optimizer, mock_optimizer_instance)
            mock_create_optimizer.assert_called_once()
    
    def test_create_scheduler(self):
        """Test scheduler creation."""
        # Mock the required classes
        with patch('spikezoo.models.depth_estimation_model.DepthEstimationModel') as mock_model, \
             patch('spikezoo.datasets.depth_estimation_dataset.DepthEstimationDataset') as mock_dataset, \
             patch('torch.utils.data.DataLoader') as mock_dataloader, \
             patch('spikezoo.utils.scheduler_utils.create_scheduler') as mock_create_scheduler:
            
            # Setup mocks
            mock_model_instance = Mock()
            mock_model_instance.net = Mock()
            mock_model_instance.build_network = Mock()
            mock_model.return_value = mock_model_instance
            
            mock_dataset_instance = Mock()
            mock_dataset_instance.build_source = Mock()
            mock_dataset.return_value = mock_dataset_instance
            
            mock_dataloader_instance = Mock()
            mock_dataloader.return_value = mock_dataloader_instance
            
            mock_scheduler_instance = Mock()
            mock_create_scheduler.return_value = mock_scheduler_instance
            
            # Create training pipeline
            train_pipeline = DepthEstimationTrainPipeline(
                self.train_pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Test scheduler creation
            scheduler = train_pipeline._create_scheduler()
            
            # Verify scheduler was created
            self.assertEqual(scheduler, mock_scheduler_instance)
            mock_create_scheduler.assert_called_once()
    
    def test_build_dataloader(self):
        """Test dataloader building."""
        # Mock the required classes
        with patch('spikezoo.models.depth_estimation_model.DepthEstimationModel') as mock_model, \
             patch('spikezoo.datasets.depth_estimation_dataset.DepthEstimationDataset') as mock_dataset, \
             patch('torch.utils.data.DataLoader') as mock_dataloader_class:
            
            # Setup mocks
            mock_model_instance = Mock()
            mock_model_instance.net = Mock()
            mock_model_instance.build_network = Mock()
            mock_model.return_value = mock_model_instance
            
            mock_dataset_instance = Mock()
            mock_dataset_instance.build_source = Mock()
            mock_dataset.return_value = mock_dataset_instance
            
            mock_dataloader_instance = Mock()
            mock_dataloader_class.return_value = mock_dataloader_instance
            
            # Create training pipeline
            train_pipeline = DepthEstimationTrainPipeline(
                self.train_pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Test building train dataloader
            train_loader = train_pipeline._build_dataloader(mock_dataset_instance, "train")
            self.assertEqual(train_loader, mock_dataloader_instance)
            mock_dataloader_class.assert_any_call(
                mock_dataset_instance,
                batch_size=train_pipeline.cfg.bs_train,
                shuffle=True,
                num_workers=train_pipeline.cfg.nw_train,
                pin_memory=train_pipeline.cfg.pin_memory
            )
            
            # Test building test dataloader
            test_loader = train_pipeline._build_dataloader(mock_dataset_instance, "test")
            self.assertEqual(test_loader, mock_dataloader_instance)
            mock_dataloader_class.assert_any_call(
                mock_dataset_instance,
                batch_size=train_pipeline.cfg.bs_test,
                shuffle=False,
                num_workers=train_pipeline.cfg.nw_test,
                pin_memory=train_pipeline.cfg.pin_memory
            )


class TestPipelineFactoryFunctions(unittest.TestCase):
    """Test pipeline factory functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.pipeline_cfg = DepthEstimationPipelineConfig(
            version="local",
            save_folder=self.temp_dir
        )
        
        self.train_pipeline_cfg = DepthEstimationTrainPipelineConfig(
            version="local",
            save_folder=self.temp_dir,
            epochs=1
        )
        
        self.model_cfg = DepthEstimationModelConfig(
            model_name="depth_estimation",
            network_type="basic",
            input_channels=3,
            output_channels=1,
            hidden_dim=16
        )
        
        self.dataset_cfg = DepthEstimationDatasetConfig(
            dataset_name="depth_estimation",
            data_root=self.temp_dir,
            split="test",
            height=32,
            width=32
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_depth_estimation_pipeline(self):
        """Test creating depth estimation inference pipeline."""
        from spikezoo.pipeline.depth_estimation_pipeline import create_depth_estimation_pipeline
        
        # Mock the pipeline class
        with patch('spikezoo.pipeline.depth_estimation_pipeline.DepthEstimationPipeline') as mock_pipeline:
            mock_pipeline_instance = Mock()
            mock_pipeline.return_value = mock_pipeline_instance
            
            # Create pipeline using factory function
            pipeline = create_depth_estimation_pipeline(
                self.pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Verify pipeline was created
            self.assertEqual(pipeline, mock_pipeline_instance)
            mock_pipeline.assert_called_once_with(
                self.pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
    
    def test_create_depth_estimation_train_pipeline(self):
        """Test creating depth estimation training pipeline."""
        from spikezoo.pipeline.depth_estimation_pipeline import create_depth_estimation_train_pipeline
        
        # Mock the pipeline class
        with patch('spikezoo.pipeline.depth_estimation_pipeline.DepthEstimationTrainPipeline') as mock_pipeline:
            mock_pipeline_instance = Mock()
            mock_pipeline.return_value = mock_pipeline_instance
            
            # Create training pipeline using factory function
            train_pipeline = create_depth_estimation_train_pipeline(
                self.train_pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )
            
            # Verify training pipeline was created
            self.assertEqual(train_pipeline, mock_pipeline_instance)
            mock_pipeline.assert_called_once_with(
                self.train_pipeline_cfg,
                self.model_cfg,
                self.dataset_cfg
            )


if __name__ == '__main__':
    unittest.main()