#!/usr/bin/env python3
"""
Example of using the SpikeZoo model registry system.
"""

import sys
import os

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.models import (
    get_model_registry,
    list_models,
    is_model_registered,
    get_model_class,
    get_config_class,
    create_model,
    build_model_name,
    BaseModelConfig
)
# Import build_model_cfg for backward compatibility example only
from spikezoo.models.model_registry import create_model as create_model_registry


def example_model_registry_usage():
    """Example of using the model registry."""
    print("=== SpikeZoo Model Registry Example ===\n")
    
    # Get the global registry
    registry = get_model_registry()
    
    # List available models
    print("1. Available models:")
    models = list_models()
    for model_name in models:
        print(f"   - {model_name}")
    print()
    
    # Check if specific models are registered
    print("2. Model registration status:")
    test_models = ["base", "spk2imgnet", "ssir"]
    for model_name in test_models:
        is_registered = is_model_registered(model_name)
        print(f"   {model_name}: {'Registered' if is_registered else 'Not registered'}")
    print()
    
    # Get model classes
    print("3. Getting model classes:")
    for model_name in ["base", "spk2imgnet"]:
        if is_model_registered(model_name):
            model_class = get_model_class(model_name)
            config_class = get_config_class(model_name)
            print(f"   {model_name}:")
            print(f"     Model class: {model_class}")
            print(f"     Config class: {config_class}")
        else:
            print(f"   {model_name}: Not registered")
    print()
    
    # Create model instances
    print("4. Creating model instances:")
    try:
        # Create model by name
        base_model = create_model("base")
        if base_model:
            print(f"   Created base model: {type(base_model).__name__}")
        
        # Create model with custom config
        base_config = BaseModelConfig(model_name="base", load_state=False)
        base_model2 = create_model("base", base_config)
        if base_model2:
            print(f"   Created base model with custom config: {type(base_model2).__name__}")
            print(f"   Config: load_state={base_model2.cfg.load_state}")
        print()
    except Exception as e:
        print(f"   Error creating models: {e}")
        print()


def example_backward_compatibility():
    """Example showing backward compatibility with old APIs."""
    print("=== Backward Compatibility Example ===\n")
    
    # Using old build_model_name function
    print("1. Using build_model_name (old API):")
    try:
        model = build_model_name("base")
        print(f"   Created model: {type(model).__name__}")
        print(f"   Model name: {model.cfg.model_name}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    # Note: build_model_cfg is deprecated, showing how to use the new registry instead
    print("2. Using model registry (new API replacement for build_model_cfg):")
    try:
        config = BaseModelConfig(model_name="base", load_state=True)
        model = create_model_registry("base", config)
        print(f"   Created model: {type(model).__name__}")
        print(f"   Model name: {model.cfg.model_name}")
        print(f"   Load state: {model.cfg.load_state}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()


def example_new_api_usage():
    """Example showing new API usage."""
    print("=== New API Usage Example ===\n")
    
    # Direct registry usage
    registry = get_model_registry()
    
    print("1. Direct registry operations:")
    # List models
    models = registry.list_models()
    print(f"   Available models: {len(models)}")
    
    # Check registration
    is_registered = registry.is_model_registered("base")
    print(f"   Is 'base' registered: {is_registered}")
    
    # Get classes
    model_class = registry.get_model_class("base")
    config_class = registry.get_config_class("base")
    print(f"   Base model class: {model_class}")
    print(f"   Base config class: {config_class}")
    
    # Create model
    model = registry.create_model("base")
    if model:
        print(f"   Created model instance: {type(model).__name__}")
    print()


if __name__ == "__main__":
    example_model_registry_usage()
    example_backward_compatibility()
    example_new_api_usage()