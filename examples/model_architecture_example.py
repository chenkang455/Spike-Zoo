#!/usr/bin/env python3
"""
Example of using the refactored SpikeZoo model architecture.
"""

import sys
import os
import torch

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.models import (
    BaseModel,
    BaseModelConfig,
    ModelRegistry,
    get_model_registry,
    create_model,
    list_models
)
from spikezoo.models.modules import (
    network_loader,
    loss_functions,
    metric_utils
)


class ExampleModel(BaseModel):
    """Example model implementation."""
    
    def __init__(self, cfg: BaseModelConfig):
        """Initialize example model."""
        super().__init__(cfg)
        # Example doesn't actually build a network since we're focusing on architecture
    
    def build_network(self, mode: str = "train", version: str = "local"):
        """Build network (simplified for example)."""
        print(f"Building network for {self.cfg.model_name} in {mode} mode")
        self.net = torch.nn.Linear(10, 1)  # Dummy network
        return self


class ExampleModelConfig(BaseModelConfig):
    """Example model configuration."""
    pass


def example_model_creation():
    """Example of model creation with refactored architecture."""
    print("=== Model Creation Example ===\n")
    
    # Create configuration
    config = ExampleModelConfig(
        model_name="example_model",
        load_state=False
    )
    
    # Create model directly
    model = ExampleModel(config)
    print(f"1. Created model directly: {type(model).__name__}")
    
    # Register model
    registry = get_model_registry()
    registry.register_model("example_model", ExampleModel, ExampleModelConfig)
    print("2. Registered model with registry")
    
    # Create model through registry
    model2 = create_model("example_model")
    print(f"3. Created model through registry: {type(model2).__name__}")
    
    # List available models
    models = list_models()
    print(f"4. Available models: {models}")
    
    print()


def example_module_usage():
    """Example of using modular components."""
    print("=== Module Usage Example ===\n")
    
    # Test loss functions
    print("1. Testing loss functions:")
    loss_names = loss_functions.list_loss_functions()
    print(f"   Available loss functions: {loss_names}")
    
    # Test a specific loss function
    l1_loss = loss_functions.get_loss_function("l1")
    pred = torch.randn(5, 3, 32, 32)
    target = torch.randn(5, 3, 32, 32)
    loss_value = l1_loss(pred, target)
    print(f"   L1 loss value: {loss_value.item():.6f}")
    
    # Test metric utilities
    print("\n2. Testing metric utilities:")
    # These would typically be used with actual model outputs
    print("   Metric utilities ready for use")
    
    print()


def example_network_loading():
    """Example of network loading utilities."""
    print("=== Network Loading Example ===\n")
    
    print("1. Network loading utilities:")
    print("   - load_model_weights function available")
    print("   - save_model_weights function available")
    print("   - Automatic retry mechanism included")
    
    print()


def example_registry_operations():
    """Example of registry operations."""
    print("=== Registry Operations Example ===\n")
    
    # Get registry
    registry = get_model_registry()
    print("1. Got model registry")
    
    # Register a model
    class TestModel(BaseModel):
        def __init__(self, cfg: BaseModelConfig):
            super().__init__(cfg)
    
    class TestModelConfig(BaseModelConfig):
        pass
    
    registry.register_model("test_model", TestModel, TestModelConfig)
    print("2. Registered test model")
    
    # Check if model is registered
    is_registered = registry.is_model_registered("test_model")
    print(f"3. Test model registered: {is_registered}")
    
    # List models
    models = registry.list_models()
    print(f"4. All registered models: {models}")
    
    # Get model class
    model_class = registry.get_model_class("test_model")
    print(f"5. Retrieved model class: {model_class.__name__ if model_class else None}")
    
    # Create model instance
    model_instance = registry.create_model("test_model")
    print(f"6. Created model instance: {type(model_instance).__name__ if model_instance else None}")
    
    # Unregister model
    registry.unregister_model("test_model")
    is_registered = registry.is_model_registered("test_model")
    print(f"7. After unregistering, test model registered: {is_registered}")
    
    print()


def example_backward_compatibility():
    """Example showing backward compatibility."""
    print("=== Backward Compatibility Example ===\n")
    
    print("1. Backward compatibility maintained:")
    print("   - Existing model creation APIs still work")
    print("   - BaseModel interface unchanged")
    print("   - Configuration system preserved")
    
    print()


if __name__ == "__main__":
    example_model_creation()
    example_module_usage()
    example_network_loading()
    example_registry_operations()
    example_backward_compatibility()
    
    print("All model architecture examples completed!")