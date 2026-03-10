#!/usr/bin/env python3
"""
Example usage of depth estimation pipelines in SpikeZoo.
"""

import sys
import os
from pathlib import Path

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


def example_inference_pipeline():
    """Demonstrate depth estimation inference pipeline."""
    print("=== Depth Estimation Inference Pipeline ===\n")
    
    # Create pipeline configuration
    pipeline_cfg = DepthEstimationPipelineConfig(
        version="local",
        save_folder="./results/depth_estimation_inference",
        save_metric=True,
        save_img=True,
        bs_test=1,
        nw_test=0
    )
    
    print("1. Created pipeline configuration:")
    print(f"   Version: {pipeline_cfg.version}")
    print(f"   Save folder: {pipeline_cfg.save_folder}")
    print(f"   Save metrics: {pipeline_cfg.save_metric}")
    print(f"   Save images: {pipeline_cfg.save_img}")
    print()
    
    # Create model configuration
    model_cfg = DepthEstimationModelConfig(
        model_name="depth_estimation",
        network_type="basic",
        input_channels=3,
        output_channels=1,
        hidden_dim=32,
        num_layers=3,
        load_state=False
    )
    
    print("2. Created model configuration:")
    print(f"   Model name: {model_cfg.model_name}")
    print(f"   Network type: {model_cfg.network_type}")
    print(f"   Input channels: {model_cfg.input_channels}")
    print(f"   Output channels: {model_cfg.output_channels}")
    print()
    
    # Create dataset configuration
    dataset_cfg = DepthEstimationDatasetConfig(
        dataset_name="depth_estimation",
        data_root="./data/depth_estimation",
        split="test",
        height=128,
        width=128,
        sequence_length=5,
        augment=False
    )
    
    print("3. Created dataset configuration:")
    print(f"   Dataset name: {dataset_cfg.dataset_name}")
    print(f"   Data root: {dataset_cfg.data_root}")
    print(f"   Split: {dataset_cfg.split}")
    print(f"   Height: {dataset_cfg.height}")
    print(f"   Width: {dataset_cfg.width}")
    print()
    
    # Create pipeline
    try:
        pipeline = DepthEstimationPipeline(pipeline_cfg, model_cfg, dataset_cfg)
        print("4. Created depth estimation pipeline successfully")
        
        # Note: We won't actually run the pipeline in this example
        # as it would require actual data and trained weights
        print("   Pipeline created but not executed (would require data and weights)")
    except Exception as e:
        print(f"4. Failed to create pipeline: {e}")
    print()


def example_training_pipeline():
    """Demonstrate depth estimation training pipeline."""
    print("=== Depth Estimation Training Pipeline ===\n")
    
    # Create training pipeline configuration
    train_pipeline_cfg = DepthEstimationTrainPipelineConfig(
        version="local",
        save_folder="./results/depth_estimation_training",
        epochs=5,
        bs_train=2,
        bs_test=1,
        nw_train=0,
        nw_test=0,
        steps_per_save_ckpt=2,
        steps_per_cal_metrics=2,
        steps_per_log_metrics=1,
        steps_per_save_depth_vis=2,
        optimizer_cfg=AdamOptimizerConfig(lr=1e-4),
        scheduler_cfg=CosineAnnealingLRConfig(T_max=5, eta_min=1e-6),
        loss_weight_dict={"l1_loss": 1.0, "l2_loss": 0.5},
        enable_visualization=False,  # Disable for example
        enable_checkpoint=True,
        seed=42
    )
    
    print("1. Created training pipeline configuration:")
    print(f"   Version: {train_pipeline_cfg.version}")
    print(f"   Save folder: {train_pipeline_cfg.save_folder}")
    print(f"   Epochs: {train_pipeline_cfg.epochs}")
    print(f"   Batch size (train): {train_pipeline_cfg.bs_train}")
    print(f"   Batch size (test): {train_pipeline_cfg.bs_test}")
    print(f"   Enable checkpoint: {train_pipeline_cfg.enable_checkpoint}")
    print(f"   Enable visualization: {train_pipeline_cfg.enable_visualization}")
    print()
    
    # Create model configuration
    model_cfg = DepthEstimationModelConfig(
        model_name="depth_estimation",
        network_type="basic",
        input_channels=3,
        output_channels=1,
        hidden_dim=32,
        num_layers=3,
        load_state=False
    )
    
    print("2. Created model configuration:")
    print(f"   Model name: {model_cfg.model_name}")
    print(f"   Network type: {model_cfg.network_type}")
    print()
    
    # Create dataset configuration
    dataset_cfg = DepthEstimationDatasetConfig(
        dataset_name="depth_estimation",
        data_root="./data/depth_estimation",
        split="train",
        height=128,
        width=128,
        sequence_length=5,
        augment=False
    )
    
    print("3. Created dataset configuration:")
    print(f"   Dataset name: {dataset_cfg.dataset_name}")
    print(f"   Data root: {dataset_cfg.data_root}")
    print(f"   Split: {dataset_cfg.split}")
    print()
    
    # Create training pipeline
    try:
        train_pipeline = DepthEstimationTrainPipeline(train_pipeline_cfg, model_cfg, dataset_cfg)
        print("4. Created depth estimation training pipeline successfully")
        
        # Note: We won't actually run the training in this example
        # as it would require actual data and would take time
        print("   Training pipeline created but not executed (would require data)")
    except Exception as e:
        print(f"4. Failed to create training pipeline: {e}")
    print()


def example_pipeline_comparison():
    """Compare different pipeline configurations."""
    print("=== Pipeline Configuration Comparison ===\n")
    
    # Different configurations
    configs = [
        ("Basic Inference", DepthEstimationPipelineConfig()),
        ("Training with Visualization", DepthEstimationTrainPipelineConfig(
            enable_visualization=True,
            steps_per_log_metrics=50
        )),
        ("Memory Efficient Training", DepthEstimationTrainPipelineConfig(
            bs_train=1,
            nw_train=0,
            steps_grad_accumulation=8
        ))
    ]
    
    for name, cfg in configs:
        print(f"{name}:")
        print(f"  Type: {type(cfg).__name__}")
        if hasattr(cfg, 'bs_train'):
            print(f"  Train batch size: {cfg.bs_train}")
        if hasattr(cfg, 'nw_train'):
            print(f"  Train workers: {cfg.nw_train}")
        if hasattr(cfg, 'enable_visualization'):
            print(f"  Visualization: {cfg.enable_visualization}")
        if hasattr(cfg, 'steps_grad_accumulation'):
            print(f"  Gradient accumulation: {cfg.steps_grad_accumulation}")
        print()


def main():
    """Run all examples."""
    print("Depth Estimation Pipeline Examples in SpikeZoo\n")
    print("=" * 50)
    
    example_inference_pipeline()
    example_training_pipeline()
    example_pipeline_comparison()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()