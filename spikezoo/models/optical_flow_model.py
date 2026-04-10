"""
Optical flow model interface for SpikeZoo.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from spikezoo.models.architecture_loader import load_architecture_class, create_architecture


@dataclass
class OpticalFlowModelConfig(BaseModelConfig):
    """Configuration for optical flow model."""
    
    # Model-specific parameters
    input_channels: int = 2
    output_channels: int = 2
    hidden_dim: int = 64
    num_layers: int = 4
    network_type: str = "basic"  # 'basic', 'event', 'pyramid'
    
    # Override inherited parameters
    model_name: str = "optical_flow"
    model_file_name: str = "nets"
    model_cls_name: str = "OpticalFlowNet"


class OpticalFlowModel(BaseModel):
    """Optical flow model interface."""
    
    def __init__(self, cfg: OpticalFlowModelConfig):
        """Initialize optical flow model.
        
        Args:
            cfg: Model configuration
        """
        # Ensure we have the right config type
        if not isinstance(cfg, OpticalFlowModelConfig):
            raise TypeError(f"Expected OpticalFlowModelConfig, got {type(cfg)}")
        
        super().__init__(cfg)
        self.cfg: OpticalFlowModelConfig = cfg  # For type hinting
    
    def build_network(self, mode: str = "train", version: str = "local"):
        """Build the network.
        
        Args:
            mode: Model mode ("train" or "eval")
            version: Model version identifier
            
        Returns:
            Self for method chaining
        """
        # Call parent build_network for standard setup
        super().build_network(mode, version)
        
        # Additional model-specific setup can go here
        print(f"Built OpticalFlowModel in {mode} mode")
        
        return self
    
    def get_outputs_dict(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Get model outputs as dictionary.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Dictionary of model outputs
        """
        # Extract events from batch
        events = batch.get("events", None)
        if events is None:
            raise ValueError("Events not found in batch")
        
        # Forward pass
        flow_pred = self.net(events)
        
        # Create outputs dictionary
        outputs = {
            "flow_pred": flow_pred
        }
        
        return outputs
    
    def get_loss_dict(self, outputs: Dict[str, Any], batch: Dict[str, Any], 
                     loss_weight_dict: Dict[str, float]) -> tuple:
        """Compute loss dictionary.
        
        Args:
            outputs: Model outputs dictionary
            batch: Input batch dictionary
            loss_weight_dict: Dictionary mapping loss names to weights
            
        Returns:
            Tuple of (loss_dict, loss_values_dict)
        """
        # Extract predicted and ground truth flow
        flow_pred = outputs.get("flow_pred", None)
        flow_gt = batch.get("flow_gt", None)
        
        if flow_pred is None or flow_gt is None:
            raise ValueError("Flow prediction or ground truth not found")
        
        # Compute endpoint error (EPE)
        epe = torch.norm(flow_pred - flow_gt, p=2, dim=1)
        epe_loss = epe.mean()
        
        # Create loss dictionaries
        loss_dict = {
            "epe_loss": epe_loss * loss_weight_dict.get("epe_loss", 1.0)
        }
        
        loss_values_dict = {
            "epe_loss": epe_loss.item()
        }
        
        return loss_dict, loss_values_dict
    
    def get_visual_dict(self, batch: Dict[str, Any], 
                       outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get visualization dictionary.
        
        Args:
            batch: Input batch dictionary
            outputs: Model outputs dictionary
            
        Returns:
            Dictionary for visualization
        """
        # Extract data
        events = batch.get("events", None)
        flow_gt = batch.get("flow_gt", None)
        flow_pred = outputs.get("flow_pred", None)
        image1 = batch.get("image1", None)
        image2 = batch.get("image2", None)
        
        # Create visualization dictionary
        visual_dict = {
            "events": events,
            "flow_gt": flow_gt,
            "flow_pred": flow_pred,
            "image1": image1,
            "image2": image2
        }
        
        return visual_dict
    
    def spk2img(self, spike: torch.Tensor) -> torch.Tensor:
        """Convert spikes to images (if applicable).
        
        Args:
            spike: Input spike tensor
            
        Returns:
            Reconstructed image tensor
        """
        # For optical flow, this might not be directly applicable
        # but we can provide a placeholder implementation
        if self.net is not None:
            # If the network has a spk2img capability, use it
            if hasattr(self.net, 'spk2img'):
                return self.net.spk2img(spike)
            else:
                # Simple pass-through for now
                return spike
        else:
            return spike


# Factory function for creating optical flow models
def create_optical_flow_model(config: OpticalFlowModelConfig) -> OpticalFlowModel:
    """Create optical flow model with specified configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Optical flow model instance
    """
    return OpticalFlowModel(config)


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = OpticalFlowModelConfig(
        model_name="optical_flow",
        network_type="basic",
        input_channels=2,
        output_channels=2,
        hidden_dim=64,
        num_layers=4
    )
    
    # Create model
    model = OpticalFlowModel(config)
    
    # Print model info
    print(f"Created optical flow model: {type(model).__name__}")
    print(f"Configuration: {model.cfg}")
    
    # Example of building network
    try:
        model.build_network(mode="train", version="local")
        print("Network built successfully")
    except Exception as e:
        print(f"Failed to build network: {e}")