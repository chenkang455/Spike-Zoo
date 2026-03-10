"""
Depth estimation pipeline for SpikeZoo.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import functools
from spikezoo.pipeline.base_pipeline import Pipeline, PipelineConfig
from spikezoo.pipeline.train_pipeline import TrainPipeline, TrainPipelineConfig
from spikezoo.models import BaseModel, BaseModelConfig
from spikezoo.models.depth_estimation_model import DepthEstimationModel, DepthEstimationModelConfig
from spikezoo.datasets import BaseDataset, BaseDatasetConfig
from spikezoo.datasets.depth_estimation_dataset import DepthEstimationDataset, DepthEstimationDatasetConfig
from spikezoo.utils.visualization_utils import log_scalar, log_image
from spikezoo.archs.depth_estimation.utils import depth_to_colormap, depth_metrics


@dataclass
class DepthEstimationPipelineConfig(PipelineConfig):
    """Configuration for depth estimation pipeline."""
    
    # Override mode
    _mode: str = "depth_estimation_mode"
    
    # Metrics for depth estimation
    metric_names: List[str] = field(default_factory=lambda: ["abs_rel", "sq_rel", "rmse", "accuracy"])


@dataclass
class DepthEstimationTrainPipelineConfig(TrainPipelineConfig):
    """Configuration for depth estimation training pipeline."""
    
    # Override mode
    _mode: str = "depth_estimation_train_mode"
    
    # Depth estimation specific training parameters
    steps_per_log_metrics: int = 100
    steps_per_save_depth_vis: int = 500
    
    # Loss weights for depth estimation
    loss_weight_dict: Dict[str, float] = field(default_factory=lambda: {"l1_loss": 1.0, "l2_loss": 0.5})
    
    # Metrics for depth estimation
    metric_names: List[str] = field(default_factory=lambda: ["abs_rel", "sq_rel", "rmse", "accuracy"])


class DepthEstimationPipeline(Pipeline):
    """Depth estimation inference pipeline."""
    
    def __init__(
        self,
        cfg: DepthEstimationPipelineConfig,
        model_cfg: Union[str, DepthEstimationModelConfig],
        dataset_cfg: Union[str, DepthEstimationDatasetConfig],
    ):
        """Initialize depth estimation pipeline.
        
        Args:
            cfg: Pipeline configuration
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
        """
        super().__init__(cfg, model_cfg, dataset_cfg)
    
    def _setup_model_data(self, model_cfg, dataset_cfg):
        """Setup model and data for depth estimation pipeline."""
        print("Setting up depth estimation model and data...")
        
        # Model setup
        if isinstance(model_cfg, str):
            from spikezoo.models import build_model_name
            self.model: DepthEstimationModel = build_model_name(model_cfg)
        else:
            self.model = DepthEstimationModel(model_cfg)
        
        self.model.build_network(mode="eval", version=self.cfg.version)
        torch.set_grad_enabled(False)
        
        # Dataset setup
        if isinstance(dataset_cfg, str):
            from spikezoo.datasets import build_dataset_name
            self.dataset: DepthEstimationDataset = build_dataset_name(dataset_cfg)
        else:
            self.dataset = DepthEstimationDataset(dataset_cfg)
        
        self.dataset.build_source(split="test")
        
        from spikezoo.datasets import build_dataloader
        self.dataloader = build_dataloader(self.dataset, self.cfg)
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def process(self):
        """Process depth estimation inference pipeline."""
        print("Starting depth estimation inference pipeline...")
        
        # Move model to device
        self.model.net = self.model.net.to(self.device)
        
        # Metrics storage
        metrics_storage = {
            "abs_rel": [],
            "sq_rel": [],
            "rmse": [],
            "accuracy": []
        }
        
        # Process each batch
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Processing")):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model.get_outputs_dict(batch)
            
            # Compute metrics
            metrics = self._compute_batch_metrics(outputs, batch)
            
            # Store metrics
            for key, value in metrics.items():
                metrics_storage[key].append(value)
            
            # Save results if requested
            if self.cfg.save_img:
                self._save_depth_visualizations(batch_idx, batch, outputs)
        
        # Compute final metrics
        final_metrics = {}
        for key, values in metrics_storage.items():
            if values:
                final_metrics[key] = np.mean(values)
        
        # Print results
        print("\nFinal Results:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save metrics if requested
        if self.cfg.save_metric:
            self._save_metrics(final_metrics)
        
        return final_metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Batch moved to device
        """
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch
    
    def _compute_batch_metrics(self, outputs: Dict, batch: Dict) -> Dict:
        """Compute metrics for batch.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Computed metrics
        """
        try:
            depth_pred = outputs.get("depth_pred")
            depth_gt = batch.get("depth_gt")
            
            if depth_pred is not None and depth_gt is not None:
                from spikezoo.archs.depth_estimation.utils import depth_metrics
                metrics = depth_metrics(depth_pred, depth_gt)
                return metrics
            else:
                return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "accuracy": 0.0}
        except Exception as e:
            print(f"Warning: Failed to compute metrics: {e}")
            return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "accuracy": 0.0}
    
    def _save_depth_visualizations(self, batch_idx: int, batch: Dict, outputs: Dict):
        """Save depth visualization images.
        
        Args:
            batch_idx: Batch index
            batch: Input batch
            outputs: Model outputs
        """
        try:
            # Get depth predictions and ground truth
            depth_pred = outputs.get("depth_pred")
            depth_gt = batch.get("depth_gt")
            
            if depth_pred is not None:
                # Convert to colormap
                from spikezoo.archs.depth_estimation.utils import depth_to_colormap
                pred_colormap = depth_to_colormap(depth_pred)
                pred_colormap = pred_colormap.cpu().numpy()
                
                # Save prediction
                save_path = Path(self.save_folder) / "depth_predictions" / f"pred_{batch_idx:06d}.npy"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(save_path, pred_colormap)
            
            if depth_gt is not None:
                # Convert to colormap
                from spikezoo.archs.depth_estimation.utils import depth_to_colormap
                gt_colormap = depth_to_colormap(depth_gt)
                gt_colormap = gt_colormap.cpu().numpy()
                
                # Save ground truth
                save_path = Path(self.save_folder) / "depth_ground_truth" / f"gt_{batch_idx:06d}.npy"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(save_path, gt_colormap)
                
        except Exception as e:
            print(f"Warning: Failed to save depth visualizations: {e}")
    
    def _save_metrics(self, metrics: Dict):
        """Save computed metrics.
        
        Args:
            metrics: Computed metrics dictionary
        """
        try:
            metrics_file = Path(self.save_folder) / "metrics.txt"
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.6f}\n")
            
            print(f"Metrics saved to {metrics_file}")
        except Exception as e:
            print(f"Warning: Failed to save metrics: {e}")


class DepthEstimationTrainPipeline(TrainPipeline):
    """Depth estimation training pipeline."""
    
    def __init__(
        self,
        cfg: DepthEstimationTrainPipelineConfig,
        model_cfg: Union[str, DepthEstimationModelConfig],
        dataset_cfg: Union[str, DepthEstimationDatasetConfig],
    ):
        """Initialize depth estimation training pipeline.
        
        Args:
            cfg: Training pipeline configuration
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
        """
        # Initialize parent class
        super().__init__(cfg, model_cfg, dataset_cfg)
    
    def _setup_model_data(self, model_cfg, dataset_cfg):
        """Setup model and data for depth estimation training pipeline."""
        print("Setting up depth estimation model and data for training...")
        
        # Set random seed
        from spikezoo.utils.other_utils import set_random_seed
        set_random_seed(self.cfg.seed)
        
        # Model setup
        if isinstance(model_cfg, str):
            from spikezoo.models import build_model_name
            self.model: DepthEstimationModel = build_model_name(model_cfg)
        else:
            self.model = DepthEstimationModel(model_cfg)
        
        self.model.build_network(mode="train", version=self.cfg.version)
        torch.set_grad_enabled(True)
        
        # Dataset setup
        if isinstance(dataset_cfg, str):
            from spikezoo.datasets import build_dataset_name
            self.dataset: DepthEstimationDataset = build_dataset_name(dataset_cfg)
        else:
            self.dataset = DepthEstimationDataset(dataset_cfg)
        
        # Build train and test datasets
        self.dataset.build_source(split="train")
        self.train_dataloader = self._build_dataloader(self.dataset, "train")
        
        # Build test dataset if metrics are enabled
        if self.cfg.steps_per_cal_metrics > 0:
            test_dataset = DepthEstimationDataset(dataset_cfg)
            test_dataset.build_source(split="test")
            self.test_dataloader = self._build_dataloader(test_dataset, "test")
        else:
            self.test_dataloader = None
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def _build_dataloader(self, dataset: BaseDataset, split: str):
        """Build dataloader for specific split.
        
        Args:
            dataset: Dataset instance
            split: Data split ('train' or 'test')
            
        Returns:
            Dataloader instance
        """
        from torch.utils.data import DataLoader
        
        if split == "train":
            return DataLoader(
                dataset,
                batch_size=self.cfg.bs_train,
                shuffle=True,
                num_workers=self.cfg.nw_train,
                pin_memory=self.cfg.pin_memory
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.cfg.bs_test,
                shuffle=False,
                num_workers=self.cfg.nw_test,
                pin_memory=self.cfg.pin_memory
            )
    
    def _setup_training(self):
        """Setup training components."""
        print("Setting up training components...")
        
        # Move model to device
        self.model.net = self.model.net.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler if specified
        self.scheduler = self._create_scheduler()
        
        # Setup visualization if enabled
        if self.cfg.enable_visualization:
            from spikezoo.utils.visualization_utils import setup_visualization
            setup_visualization(self.cfg.visualization_cfg or {})
        
        # Setup checkpoint manager if enabled
        if self.cfg.enable_checkpoint:
            from spikezoo.utils.checkpoint_utils import CheckpointManager
            self.checkpoint_manager = CheckpointManager(
                save_dir=Path(self.save_folder) / "checkpoints"
            )
        
        print("Training components setup complete.")
    
    def _create_optimizer(self):
        """Create optimizer for training.
        
        Returns:
            Optimizer instance
        """
        from spikezoo.utils.optimizer_utils import create_optimizer
        return create_optimizer(self.model.net.parameters(), self.cfg.optimizer_cfg)
    
    def _create_scheduler(self):
        """Create learning rate scheduler.
        
        Returns:
            Scheduler instance or None
        """
        if self.cfg.scheduler_cfg is not None:
            from spikezoo.utils.scheduler_utils import create_scheduler
            return create_scheduler(self.optimizer, self.cfg.scheduler_cfg)
        return None
    
    def train(self):
        """Train the depth estimation model."""
        print("Starting depth estimation training...")
        
        # Training loop
        for epoch in range(self.cfg.epochs):
            print(f"\nEpoch {epoch + 1}/{self.cfg.epochs}")
            
            # Train one epoch
            train_metrics = self._train_epoch(epoch)
            
            # Log training metrics
            for key, value in train_metrics.items():
                print(f"  Train {key}: {value:.6f}")
                if self.cfg.enable_visualization:
                    log_scalar(f"train/{key}", value, epoch)
            
            # Validate if enabled
            if self.test_dataloader is not None and (epoch + 1) % self.cfg.steps_per_cal_metrics == 0:
                val_metrics = self._validate_epoch(epoch)
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    print(f"  Val {key}: {value:.6f}")
                    if self.cfg.enable_visualization:
                        log_scalar(f"val/{key}", value, epoch)
            
            # Save checkpoint if enabled
            if self.cfg.enable_checkpoint and (epoch + 1) % self.cfg.steps_per_save_ckpt == 0:
                checkpoint_path = Path(self.save_folder) / "checkpoints" / f"epoch_{epoch + 1}.pth"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                self.checkpoint_manager.save_checkpoint(
                    self.model.net, self.optimizer, epoch + 1, str(checkpoint_path)
                )
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Update scheduler if available
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Flush visualization if enabled
        if self.cfg.enable_visualization:
            from spikezoo.utils.visualization_utils import flush_visualization
            flush_visualization()
        
        print("Training completed.")
    
    def _train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Training metrics
        """
        self.model.net.train()
        
        # Metrics storage
        metrics_storage = {
            "loss": [],
            "l1_loss": [],
            "l2_loss": []
        }
        
        # Progress bar
        pbar = tqdm(self.train_dataloader, desc=f"Train Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model.get_outputs_dict(batch)
            
            # Compute loss
            loss_dict, loss_values_dict = self.model.get_loss_dict(
                outputs, batch, self.cfg.loss_weight_dict
            )
            
            # Total loss
            total_loss = sum(loss_dict.values())
            
            # Backward pass
            total_loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.cfg.steps_grad_accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Store metrics
            metrics_storage["loss"].append(total_loss.item())
            for key, value in loss_values_dict.items():
                if key in metrics_storage:
                    metrics_storage[key].append(value)
                else:
                    metrics_storage[key] = [value]
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss.item():.6f}",
                "l1": f"{loss_values_dict.get('l1_loss', 0):.6f}",
                "l2": f"{loss_values_dict.get('l2_loss', 0):.6f}"
            })
            
            # Log metrics periodically
            if self.cfg.enable_visualization and (batch_idx + 1) % self.cfg.steps_per_log_metrics == 0:
                for key, values in metrics_storage.items():
                    if values:
                        avg_value = np.mean(values[-self.cfg.steps_per_log_metrics:])
                        log_scalar(f"train_step/{key}", avg_value, 
                                 epoch * len(self.train_dataloader) + batch_idx)
            
            # Save depth visualizations periodically
            if self.cfg.save_img and (batch_idx + 1) % self.cfg.steps_per_save_depth_vis == 0:
                self._save_training_depth_visualizations(batch_idx, batch, outputs)
        
        # Compute average metrics
        avg_metrics = {}
        for key, values in metrics_storage.items():
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def _validate_epoch(self, epoch: int) -> Dict:
        """Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Validation metrics
        """
        self.model.net.eval()
        
        # Metrics storage
        metrics_storage = {
            "abs_rel": [],
            "sq_rel": [],
            "rmse": [],
            "accuracy": []
        }
        
        # Progress bar
        pbar = tqdm(self.test_dataloader, desc=f"Val Epoch {epoch + 1}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model.get_outputs_dict(batch)
                
                # Compute metrics
                metrics = self._compute_batch_metrics(outputs, batch)
                
                # Store metrics
                for key, value in metrics.items():
                    metrics_storage[key].append(value)
                
                # Update progress bar
                pbar.set_postfix({
                    "abs_rel": f"{np.mean(metrics_storage['abs_rel']):.4f}" if metrics_storage['abs_rel'] else "0.0000"
                })
        
        # Compute average metrics
        avg_metrics = {}
        for key, values in metrics_storage.items():
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Batch moved to device
        """
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch
    
    def _compute_batch_metrics(self, outputs: Dict, batch: Dict) -> Dict:
        """Compute metrics for batch.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Computed metrics
        """
        try:
            depth_pred = outputs.get("depth_pred")
            depth_gt = batch.get("depth_gt")
            
            if depth_pred is not None and depth_gt is not None:
                from spikezoo.archs.depth_estimation.utils import depth_metrics
                metrics = depth_metrics(depth_pred, depth_gt)
                return metrics
            else:
                return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "accuracy": 0.0}
        except Exception as e:
            print(f"Warning: Failed to compute metrics: {e}")
            return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "accuracy": 0.0}
    
    def _save_training_depth_visualizations(self, batch_idx: int, batch: Dict, outputs: Dict):
        """Save depth visualization images during training.
        
        Args:
            batch_idx: Batch index
            batch: Input batch
            outputs: Model outputs
        """
        try:
            # Get depth predictions and ground truth
            depth_pred = outputs.get("depth_pred")
            depth_gt = batch.get("depth_gt")
            
            if depth_pred is not None:
                # Convert to colormap
                from spikezoo.archs.depth_estimation.utils import depth_to_colormap
                pred_colormap = depth_to_colormap(depth_pred)
                
                # Save prediction visualization
                save_path = Path(self.save_folder) / "training_visualizations" / f"pred_{batch_idx:06d}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert to numpy and save as image
                pred_np = pred_colormap.cpu().numpy()
                if pred_np.shape[1] == 3:  # RGB format
                    pred_np = np.transpose(pred_np, (0, 2, 3, 1))  # CHW to HWC
                pred_np = (pred_np * 255).astype(np.uint8)
                
                for i in range(pred_np.shape[0]):
                    cv2.imwrite(str(save_path.with_name(f"pred_{batch_idx:06d}_{i}.png")), 
                              cv2.cvtColor(pred_np[i], cv2.COLOR_RGB2BGR))
            
            if depth_gt is not None:
                # Convert to colormap
                from spikezoo.archs.depth_estimation.utils import depth_to_colormap
                gt_colormap = depth_to_colormap(depth_gt)
                
                # Save ground truth visualization
                save_path = Path(self.save_folder) / "training_visualizations" / f"gt_{batch_idx:06d}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert to numpy and save as image
                gt_np = gt_colormap.cpu().numpy()
                if gt_np.shape[1] == 3:  # RGB format
                    gt_np = np.transpose(gt_np, (0, 2, 3, 1))  # CHW to HWC
                gt_np = (gt_np * 255).astype(np.uint8)
                
                for i in range(gt_np.shape[0]):
                    cv2.imwrite(str(save_path.with_name(f"gt_{batch_idx:06d}_{i}.png")), 
                              cv2.cvtColor(gt_np[i], cv2.COLOR_RGB2BGR))
                
        except Exception as e:
            print(f"Warning: Failed to save training depth visualizations: {e}")


# Factory functions for creating pipelines
def create_depth_estimation_pipeline(
    cfg: DepthEstimationPipelineConfig,
    model_cfg: Union[str, DepthEstimationModelConfig],
    dataset_cfg: Union[str, DepthEstimationDatasetConfig]
) -> DepthEstimationPipeline:
    """Create depth estimation inference pipeline.
    
    Args:
        cfg: Pipeline configuration
        model_cfg: Model configuration
        dataset_cfg: Dataset configuration
        
    Returns:
        Depth estimation pipeline instance
    """
    return DepthEstimationPipeline(cfg, model_cfg, dataset_cfg)


def create_depth_estimation_train_pipeline(
    cfg: DepthEstimationTrainPipelineConfig,
    model_cfg: Union[str, DepthEstimationModelConfig],
    dataset_cfg: Union[str, DepthEstimationDatasetConfig]
) -> DepthEstimationTrainPipeline:
    """Create depth estimation training pipeline.
    
    Args:
        cfg: Training pipeline configuration
        model_cfg: Model configuration
        dataset_cfg: Dataset configuration
        
    Returns:
        Depth estimation training pipeline instance
    """
    return DepthEstimationTrainPipeline(cfg, model_cfg, dataset_cfg)