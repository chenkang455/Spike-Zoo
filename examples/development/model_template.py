#!/usr/bin/env python3
"""
Template for creating new models in SpikeZoo.

This template demonstrates the standard structure for implementing new models
following the SpikeZoo development guidelines.
"""

import sys
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Add the spikezoo package to the path
# In practice, this would be installed as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from spikezoo.models.base_model import BaseModel, BaseModelConfig
from spikezoo.models.model_registry import register_model


# =============================================================================
# 1. Model Architecture Definition
# =============================================================================

class TemplateNet(nn.Module):
    """Template neural network architecture.
    
    This is where you define your actual neural network architecture.
    Place this in archs/template/nets.py in a real implementation.
    """
    
    def __init__(self, input_channels: int = 3, hidden_dim: int = 64, num_classes: int = 10):
        """Initialize the template network.
        
        Args:
            input_channels: Number of input channels
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Example architecture
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =============================================================================
# 2. Model Configuration
# =============================================================================

@dataclass
class TemplateModelConfig(BaseModelConfig):
    """Configuration for TemplateModel.
    
    Extend BaseModelConfig with model-specific parameters.
    """
    
    # Model-specific parameters
    input_channels: int = 3
    hidden_dim: int = 64
    num_classes: int = 10
    
    # Override inherited parameters
    model_file_name: str = "nets"
    model_cls_name: str = "TemplateNet"


# =============================================================================
# 3. Model Interface Implementation
# =============================================================================

class TemplateModel(BaseModel):
    """Template model interface implementation.
    
    This class implements the standard SpikeZoo model interface.
    """
    
    def __init__(self, cfg: TemplateModelConfig):
        """Initialize template model.
        
        Args:
            cfg: Model configuration
        """
        # Ensure we have the right config type
        if not isinstance(cfg, TemplateModelConfig):
            raise TypeError(f"Expected TemplateModelConfig, got {type(cfg)}")
        
        super().__init__(cfg)
        self.cfg: TemplateModelConfig = cfg  # For type hinting
    
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
        print(f"Built TemplateModel in {mode} mode")
        
        return self
    
    def get_outputs_dict(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Get model outputs as dictionary.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Dictionary of model outputs
        """
        # This is a simplified example
        # In practice, you'd process the batch through your model
        outputs = super().get_outputs_dict(batch)
        
        # Add model-specific outputs
        # outputs["attention_weights"] = attention_weights
        # outputs["intermediate_features"] = intermediate_features
        
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
        # Use the parent implementation for standard losses
        return super().get_loss_dict(outputs, batch, loss_weight_dict)
    
    def get_visual_dict(self, batch: Dict[str, Any], 
                       outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get visualization dictionary.
        
        Args:
            batch: Input batch dictionary
            outputs: Model outputs dictionary
            
        Returns:
            Dictionary for visualization
        """
        # Use the parent implementation for standard visualization
        return super().get_visual_dict(batch, outputs)
    
    def spk2img(self, spike: torch.Tensor) -> torch.Tensor:
        """Convert spikes to images (if applicable).
        
        Args:
            spike: Input spike tensor
            
        Returns:
            Reconstructed image tensor
        """
        # This is model-specific functionality
        # For this template, we'll just pass through
        return self.net(spike) if self.net is not None else spike


# =============================================================================
# 4. Model Registration
# =============================================================================

# Register the model with the global registry
register_model("template", TemplateModel, TemplateModelConfig)


# =============================================================================
# 5. Usage Examples
# =============================================================================

def example_usage():
    """Demonstrate usage of the template model."""
    print("=== Template Model Usage Example ===\n")
    
    # Create configuration
    config = TemplateModelConfig(
        model_name="template",
        load_state=False,
        input_channels=3,
        hidden_dim=128,
        num_classes=10
    )
    
    print("1. Created model configuration:")
    print(f"   Model name: {config.model_name}")
    print(f"   Input channels: {config.input_channels}")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Num classes: {config.num_classes}")
    print()
    
    # Create model
    model = TemplateModel(config)
    print("2. Created model instance:")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Config type: {type(model.cfg).__name__}")
    print()
    
    # Register and create through registry
    from spikezoo.models import create_model, list_models
    
    print("3. Model registry operations:")
    models = list_models()
    print(f"   Available models: {models}")
    
    # Create model through registry
    registry_model = create_model("template")
    print(f"   Created through registry: {type(registry_model).__name__}")
    print()
    
    # Test network building
    print("4. Building network:")
    try:
        # This would normally work with actual weights
        # For demo, we'll catch the expected error
        model.build_network(mode="train", version="local")
        print("   Network built successfully")
    except RuntimeError as e:
        print(f"   Expected error (no weights): {e}")
    print()


def example_configuration():
    """Demonstrate configuration usage."""
    print("=== Configuration Example ===\n")
    
    # Create configuration with custom parameters
    config = TemplateModelConfig(
        model_name="my_custom_model",
        input_channels=1,  # Grayscale
        hidden_dim=256,
        num_classes=100,
        load_state=True,
        ckpt_path="/path/to/checkpoint.pth"
    )
    
    print("Created custom configuration:")
    print(f"  Model name: {config.model_name}")
    print(f"  Input channels: {config.input_channels}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num classes: {config.num_classes}")
    print(f"  Load state: {config.load_state}")
    print(f"  Checkpoint path: {config.ckpt_path}")
    print()


if __name__ == "__main__":
    example_usage()
    example_configuration()
    
    print("Template model examples completed!")
    print("\nTo use this template for your own model:")
    print("1. Copy this file to models/your_model_name_model.py")
    print("2. Replace 'Template' with your model name")
    print("3. Implement your neural network in archs/your_model_name/nets.py")
    print("4. Customize the configuration parameters")
    print("5. Implement model-specific methods")
    print("6. Register your model with register_model()")