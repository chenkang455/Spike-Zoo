#!/usr/bin/env python3
"""
Example usage of optical flow components in SpikeZoo.
"""

import sys
import os
import torch
from dataclasses import dataclass
from typing import Dict, Any

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.models.optical_flow_model import OpticalFlowModel, OpticalFlowModelConfig
from spikezoo.datasets.optical_flow_dataset import OpticalFlowDataset, OpticalFlowDatasetConfig
from spikezoo.archs.optical_flow.nets import create_optical_flow_network
from spikezoo.archs.optical_flow.utils import flow_to_image, compute_flow_metrics


def example_model_creation():
    """Demonstrate optical flow model creation."""
    print("=== Optical Flow Model Creation ===\n")
    
    # Create configuration
    config = OpticalFlowModelConfig(
        model_name="optical_flow",
        network_type="basic",
        input_channels=2,
        output_channels=2,
        hidden_dim=64,
        num_layers=4,
        load_state=False
    )
    
    print("1. Created model configuration:")
    print(f"   Model name: {config.model_name}")
    print(f"   Network type: {config.network_type}")
    print(f"   Input channels: {config.input_channels}")
    print(f"   Output channels: {config.output_channels}")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Num layers: {config.num_layers}")
    print()
    
    # Create model
    model = OpticalFlowModel(config)
    print("2. Created model instance:")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Config type: {type(model.cfg).__name__}")
    print()
    
    # Test network building
    print("3. Building network:")
    try:
        model.build_network(mode="train", version="local")
        print("   Network built successfully")
    except Exception as e:
        print(f"   Warning: Network building failed (expected without actual weights): {e}")
    print()


def example_network_architectures():
    """Demonstrate different optical flow network architectures."""
    print("=== Optical Flow Network Architectures ===\n")
    
    # Test different network types
    network_types = ['basic', 'event', 'pyramid']
    
    for net_type in network_types:
        print(f"1. Creating {net_type} network:")
        try:
            if net_type == 'event':
                # Event network has additional parameters
                network = create_optical_flow_network(
                    net_type, 
                    input_channels=2, 
                    output_channels=2,
                    hidden_dim=32,
                    num_layers=3,
                    temporal_bins=10
                )
            else:
                network = create_optical_flow_network(
                    net_type, 
                    input_channels=2, 
                    output_channels=2,
                    hidden_dim=32,
                    num_layers=3
                )
            
            print(f"   Created {net_type} network: {type(network).__name__}")
            
            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 2, 64, 64)
            with torch.no_grad():
                output = network(dummy_input)
                print(f"   Input shape: {dummy_input.shape}")
                print(f"   Output shape: {output.shape}")
        except Exception as e:
            print(f"   Failed to create {net_type} network: {e}")
        print()


def example_dataset_usage():
    """Demonstrate optical flow dataset usage."""
    print("=== Optical Flow Dataset Usage ===\n")
    
    # Create dataset configuration
    cfg = OpticalFlowDatasetConfig(
        data_root="./data/optical_flow",
        split="train",
        height=128,
        width=128,
        sequence_length=5,
        augment=False  # Disable augmentation for example
    )
    
    print("1. Created dataset configuration:")
    print(f"   Data root: {cfg.data_root}")
    print(f"   Split: {cfg.split}")
    print(f"   Height: {cfg.height}")
    print(f"   Width: {cfg.width}")
    print(f"   Sequence length: {cfg.sequence_length}")
    print()
    
    # Create dataset
    try:
        dataset = OpticalFlowDataset(cfg)
        print(f"2. Created dataset with {len(dataset)} samples")
        
        # Get a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print("3. Sample contents:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape}")
                else:
                    print(f"   {key}: {type(value)}")
        else:
            print("2. Dataset is empty (expected in this example)")
    except Exception as e:
        print(f"2. Failed to create dataset: {e}")
    print()


def example_utility_functions():
    """Demonstrate optical flow utility functions."""
    print("=== Optical Flow Utility Functions ===\n")
    
    # Create dummy flow fields
    batch_size, height, width = 1, 64, 64
    flow_pred = torch.randn(batch_size, 2, height, width)
    flow_gt = torch.randn(batch_size, 2, height, width)
    
    print("1. Computing flow metrics:")
    try:
        metrics = compute_flow_metrics(flow_pred, flow_gt)
        print(f"   EPE: {metrics['epe']:.4f}")
        print(f"   Angular error: {metrics['angular_error']:.4f}°")
        print(f"   Outlier rate: {metrics['outlier_rate']:.4f}")
    except Exception as e:
        print(f"   Failed to compute metrics: {e}")
    print()
    
    print("2. Converting flow to image:")
    try:
        flow_image = flow_to_image(flow_pred)
        print(f"   Flow image shape: {flow_image.shape}")
        print(f"   Flow image range: [{flow_image.min():.4f}, {flow_image.max():.4f}]")
    except Exception as e:
        print(f"   Failed to convert flow to image: {e}")
    print()


def example_end_to_end_pipeline():
    """Demonstrate end-to-end optical flow pipeline."""
    print("=== End-to-End Optical Flow Pipeline ===\n")
    
    # Create model
    config = OpticalFlowModelConfig(
        model_name="optical_flow",
        network_type="basic",
        input_channels=2,
        output_channels=2,
        hidden_dim=32,
        num_layers=3,
        load_state=False
    )
    
    model = OpticalFlowModel(config)
    
    print("1. Created optical flow model")
    
    # Create dummy batch
    batch = {
        "events": torch.randn(2, 2, 128, 128),  # Batch of event frames
        "flow_gt": torch.randn(2, 2, 128, 128),  # Ground truth flow
        "image1": torch.randn(2, 3, 128, 128),   # First images
        "image2": torch.randn(2, 3, 128, 128)    # Second images
    }
    
    print("2. Created dummy batch:")
    for key, value in batch.items():
        print(f"   {key}: {value.shape}")
    print()
    
    # Test forward pass
    print("3. Testing forward pass:")
    try:
        outputs = model.get_outputs_dict(batch)
        print(f"   Outputs keys: {list(outputs.keys())}")
        print(f"   Flow prediction shape: {outputs['flow_pred'].shape}")
    except Exception as e:
        print(f"   Forward pass failed: {e}")
    print()
    
    # Test loss computation
    print("4. Testing loss computation:")
    try:
        loss_weight_dict = {"epe_loss": 1.0}
        loss_dict, loss_values_dict = model.get_loss_dict(outputs, batch, loss_weight_dict)
        print(f"   Loss dict keys: {list(loss_dict.keys())}")
        print(f"   Loss values dict: {loss_values_dict}")
    except Exception as e:
        print(f"   Loss computation failed: {e}")
    print()
    
    # Test visualization
    print("5. Testing visualization:")
    try:
        visual_dict = model.get_visual_dict(batch, outputs)
        print(f"   Visual dict keys: {list(visual_dict.keys())}")
        for key, value in visual_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
    except Exception as e:
        print(f"   Visualization failed: {e}")
    print()


def main():
    """Run all examples."""
    print("Optical Flow Examples in SpikeZoo\n")
    print("=" * 50)
    
    example_model_creation()
    example_network_architectures()
    example_dataset_usage()
    example_utility_functions()
    example_end_to_end_pipeline()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()