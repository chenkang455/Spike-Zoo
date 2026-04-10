#!/usr/bin/env python3
"""
Example of using the SpikeZoo model registration system.
"""

import sys
import os
import torch
import torch.nn as nn

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.core import (
    ModelRegistry,
    ModelInfo,
    register_model,
    unregister_model,
    get_model_info,
    list_models,
    create_model,
    create_model_with_config,
    get_model_registry,
    discover_models_from_directory
)


class SimpleModel(nn.Module):
    """Simple model for demonstration."""
    
    def __init__(self, input_size=10, output_size=1):
        """Initialize simple model."""
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        """Forward pass."""
        return self.linear(x)


class SimpleModelConfig:
    """Configuration for simple model."""
    
    def __init__(self, input_size=10, output_size=1):
        """Initialize configuration."""
        self.input_size = input_size
        self.output_size = output_size


def create_simple_model(config=None):
    """
    Factory function to create simple model.
    
    Args:
        config: Model configuration (optional)
        
    Returns:
        SimpleModel instance
    """
    if config is None:
        return SimpleModel()
    else:
        return SimpleModel(
            input_size=config.input_size,
            output_size=config.output_size
        )


def example_basic_registration():
    """Example of basic model registration."""
    print("=== Basic Model Registration Example ===\n")
    
    # Create registry
    registry = ModelRegistry()
    
    # Register model with class
    registry.register_model(
        name="simple_model",
        model_class=SimpleModel,
        config_class=SimpleModelConfig,
        version="1.0.0",
        description="Simple linear model",
        author="Example Author",
        category="regression",
        tags=["linear", "simple"]
    )
    
    print("1. Registered model with class:")
    model_info = registry.get_model_info("simple_model")
    if model_info:
        print(f"   Name: {model_info.name}")
        print(f"   Version: {model_info.version}")
        print(f"   Description: {model_info.description}")
        print(f"   Category: {model_info.category}")
        print(f"   Tags: {model_info.tags}")
    print()
    
    # Register model with factory function
    registry.register_model(
        name="factory_model",
        factory_function=create_simple_model,
        version="1.0.0",
        description="Model created with factory function",
        author="Factory Author",
        category="regression",
        tags=["factory", "functional"]
    )
    
    print("2. Registered model with factory function:")
    model_info = registry.get_model_info("factory_model")
    if model_info:
        print(f"   Name: {model_info.name}")
        print(f"   Description: {model_info.description}")
        print(f"   Has factory function: {model_info.factory_function is not None}")
    print()


def example_model_creation():
    """Example of model creation."""
    print("=== Model Creation Example ===\n")
    
    # Create registry and register models
    registry = ModelRegistry()
    registry.register_model(
        name="simple_model",
        model_class=SimpleModel,
        config_class=SimpleModelConfig
    )
    
    registry.register_model(
        name="factory_model",
        factory_function=create_simple_model
    )
    
    # Create model instances
    print("1. Creating models:")
    
    # Create model with class
    try:
        model1 = registry.create_model("simple_model", input_size=20, output_size=2)
        print(f"   Created simple_model: {type(model1).__name__}")
        print(f"   Model type: {next(model1.parameters()).dtype}")
    except Exception as e:
        print(f"   Error creating simple_model: {e}")
    
    # Create model with factory function
    try:
        model2 = registry.create_model("factory_model")
        print(f"   Created factory_model: {type(model2).__name__}")
        print(f"   Model type: {next(model2.parameters()).dtype}")
    except Exception as e:
        print(f"   Error creating factory_model: {e}")
    
    # Create model with config
    try:
        config = SimpleModelConfig(input_size=15, output_size=3)
        model3 = registry.create_model_with_config("simple_model", config)
        print(f"   Created model with config: {type(model3).__name__}")
        print(f"   Input size: {model3.linear.in_features}")
        print(f"   Output size: {model3.linear.out_features}")
    except Exception as e:
        print(f"   Error creating model with config: {e}")
    
    print()


def example_global_registry():
    """Example of using global registry."""
    print("=== Global Registry Example ===\n")
    
    # Register models with global registry
    register_model(
        name="global_simple_model",
        model_class=SimpleModel,
        config_class=SimpleModelConfig,
        version="1.0.0",
        description="Simple model registered globally",
        author="Global Author",
        category="regression",
        tags=["global", "simple"]
    )
    
    print("1. Registered model with global registry:")
    model_names = list_models()
    print(f"   Registered models: {model_names}")
    
    model_info = get_model_info("global_simple_model")
    if model_info:
        print(f"   Model info: {model_info.name} - {model_info.description}")
    
    # Create model using global registry
    try:
        model = create_model("global_simple_model", input_size=25, output_size=5)
        print(f"2. Created model using global registry: {type(model).__name__}")
        print(f"   Input size: {model.linear.in_features}")
        print(f"   Output size: {model.linear.out_features}")
    except Exception as e:
        print(f"2. Error creating model: {e}")
    
    # Create model with config using global registry
    try:
        config = SimpleModelConfig(input_size=30, output_size=1)
        model = create_model_with_config("global_simple_model", config)
        print(f"3. Created model with config using global registry: {type(model).__name__}")
    except Exception as e:
        print(f"3. Error creating model with config: {e}")
    
    print()


def example_model_listing():
    """Example of listing models."""
    print("=== Model Listing Example ===\n")
    
    # Create registry and register various models
    registry = ModelRegistry()
    
    # Register models in different categories
    registry.register_model(
        name="linear_regression",
        model_class=SimpleModel,
        category="regression",
        tags=["linear", "statistics"]
    )
    
    registry.register_model(
        name="cnn_classifier",
        model_class=SimpleModel,  # Using SimpleModel for demo
        category="classification",
        tags=["cnn", "image"]
    )
    
    registry.register_model(
        name="rnn_sequence",
        model_class=SimpleModel,  # Using SimpleModel for demo
        category="sequence",
        tags=["rnn", "nlp"]
    )
    
    print("1. All registered models:")
    all_models = registry.list_models()
    for model_name in all_models:
        print(f"   - {model_name}")
    print()
    
    print("2. Models by category:")
    categories = registry.get_model_categories()
    for category in categories:
        models = registry.list_models_by_category(category)
        print(f"   {category}: {models}")
    print()
    
    print("3. Models by tag:")
    tags = registry.get_model_tags()
    for tag in tags:
        models = registry.list_models_by_tag(tag)
        print(f"   {tag}: {models}")
    print()


def example_model_discovery():
    """Example of model discovery from directory."""
    print("=== Model Discovery Example ===\n")
    
    # This example shows how to discover models from a directory
    # In practice, you would have actual model files in the directory
    
    # Get the global registry
    registry = get_model_registry()
    
    # Discover models from plugins directory (if it exists)
    plugins_dir = os.path.join(os.path.dirname(__file__), "..", "spikezoo", "plugins")
    if os.path.exists(plugins_dir):
        print("1. Discovering models from plugins directory:")
        print(f"   Searching in: {plugins_dir}")
        # Note: This would actually import and register models if they exist
        # For this example, we'll just show the concept
        print("   Discovery would import and register models from plugin files")
    else:
        print("1. Plugins directory not found, skipping discovery")
    
    print()


def example_error_handling():
    """Example of error handling."""
    print("=== Error Handling Example ===\n")
    
    registry = ModelRegistry()
    
    print("1. Trying to create unregistered model:")
    try:
        model = registry.create_model("nonexistent_model")
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    print("\n2. Registering model without class or factory:")
    registry.register_model(name="incomplete_model", description="Model without creation method")
    
    print("   Trying to create incomplete model:")
    try:
        model = registry.create_model("incomplete_model")
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    print()


if __name__ == "__main__":
    example_basic_registration()
    example_model_creation()
    example_global_registry()
    example_model_listing()
    example_model_discovery()
    example_error_handling()