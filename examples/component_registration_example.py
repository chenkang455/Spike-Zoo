#!/usr/bin/env python3
"""
Example of using the SpikeZoo component registration system.
"""

import sys
import os
import torch

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.core import (
    ComponentRegistry,
    ComponentInfo,
    ComponentType,
    register_component,
    unregister_component,
    get_component_info,
    list_components,
    list_components_by_type,
    create_component,
    create_component_with_config,
    get_component_registry,
    discover_components_from_directory,
    check_component_dependencies
)


class SimplePreprocessor:
    """Simple preprocessor for demonstration."""
    
    def __init__(self, normalize=True):
        """Initialize preprocessor."""
        self.normalize = normalize
    
    def preprocess(self, data):
        """Preprocess data."""
        if self.normalize:
            data = (data - data.mean()) / (data.std() + 1e-8)
        return data


class SimplePostprocessor:
    """Simple postprocessor for demonstration."""
    
    def __init__(self, clip_values=True):
        """Initialize postprocessor."""
        self.clip_values = clip_values
    
    def postprocess(self, data):
        """Postprocess data."""
        if self.clip_values:
            data = torch.clamp(data, 0, 1)
        return data


class SimpleMetric:
    """Simple metric for demonstration."""
    
    def __init__(self, reduction='mean'):
        """Initialize metric."""
        self.reduction = reduction
    
    def compute(self, pred, target):
        """Compute metric."""
        if self.reduction == 'mean':
            return torch.mean((pred - target) ** 2).item()
        else:
            return torch.sum((pred - target) ** 2).item()


class SimpleLoss:
    """Simple loss for demonstration."""
    
    def __init__(self, reduction='mean'):
        """Initialize loss."""
        self.reduction = reduction
    
    def compute(self, pred, target):
        """Compute loss."""
        if self.reduction == 'mean':
            return torch.mean((pred - target) ** 2)
        else:
            return torch.sum((pred - target) ** 2)


def create_simple_preprocessor(config=None):
    """Factory function for simple preprocessor."""
    if config is None:
        return SimplePreprocessor()
    else:
        return SimplePreprocessor(normalize=config.get('normalize', True))


def create_simple_postprocessor(config=None):
    """Factory function for simple postprocessor."""
    if config is None:
        return SimplePostprocessor()
    else:
        return SimplePostprocessor(clip_values=config.get('clip_values', True))


def create_simple_metric(config=None):
    """Factory function for simple metric."""
    if config is None:
        return SimpleMetric()
    else:
        return SimpleMetric(reduction=config.get('reduction', 'mean'))


def create_simple_loss(config=None):
    """Factory function for simple loss."""
    if config is None:
        return SimpleLoss()
    else:
        return SimpleLoss(reduction=config.get('reduction', 'mean'))


def example_basic_registration():
    """Example of basic component registration."""
    print("=== Basic Component Registration Example ===\n")
    
    # Create registry
    registry = ComponentRegistry()
    
    # Register preprocessor component
    registry.register_component(
        name="simple_preprocessor",
        component_type=ComponentType.PREPROCESSOR,
        component_class=SimplePreprocessor,
        factory_function=create_simple_preprocessor,
        version="1.0.0",
        description="Simple data preprocessor",
        author="Example Author",
        category="preprocessing",
        tags=["normalization", "simple"]
    )
    
    print("1. Registered preprocessor component:")
    component_info = registry.get_component_info("simple_preprocessor")
    if component_info:
        print(f"   Name: {component_info.name}")
        print(f"   Type: {component_info.component_type.value}")
        print(f"   Version: {component_info.version}")
        print(f"   Description: {component_info.description}")
        print(f"   Category: {component_info.category}")
        print(f"   Tags: {component_info.tags}")
    print()
    
    # Register postprocessor component
    registry.register_component(
        name="simple_postprocessor",
        component_type=ComponentType.POSTPROCESSOR,
        component_class=SimplePostprocessor,
        factory_function=create_simple_postprocessor,
        version="1.0.0",
        description="Simple data postprocessor",
        author="Example Author",
        category="postprocessing",
        tags=["clipping", "simple"]
    )
    
    print("2. Registered postprocessor component:")
    component_info = registry.get_component_info("simple_postprocessor")
    if component_info:
        print(f"   Name: {component_info.name}")
        print(f"   Type: {component_info.component_type.value}")
        print(f"   Description: {component_info.description}")
        print(f"   Has factory function: {component_info.factory_function is not None}")
    print()


def example_component_creation():
    """Example of component creation."""
    print("=== Component Creation Example ===\n")
    
    # Create registry and register components
    registry = ComponentRegistry()
    registry.register_component(
        name="simple_preprocessor",
        component_type=ComponentType.PREPROCESSOR,
        component_class=SimplePreprocessor,
        factory_function=create_simple_preprocessor
    )
    
    registry.register_component(
        name="simple_postprocessor",
        component_type=ComponentType.POSTPROCESSOR,
        component_class=SimplePostprocessor,
        factory_function=create_simple_postprocessor
    )
    
    registry.register_component(
        name="simple_metric",
        component_type=ComponentType.METRIC,
        component_class=SimpleMetric,
        factory_function=create_simple_metric
    )
    
    # Create component instances
    print("1. Creating components:")
    
    # Create preprocessor with class
    try:
        preprocessor = registry.create_component("simple_preprocessor", normalize=False)
        print(f"   Created simple_preprocessor: {type(preprocessor).__name__}")
        print(f"   Normalize: {preprocessor.normalize}")
    except Exception as e:
        print(f"   Error creating simple_preprocessor: {e}")
    
    # Create postprocessor with factory function
    try:
        postprocessor = registry.create_component("simple_postprocessor")
        print(f"   Created simple_postprocessor: {type(postprocessor).__name__}")
        print(f"   Clip values: {postprocessor.clip_values}")
    except Exception as e:
        print(f"   Error creating simple_postprocessor: {e}")
    
    # Create metric with config
    try:
        config = {'reduction': 'sum'}
        metric = registry.create_component_with_config("simple_metric", config)
        print(f"   Created metric with config: {type(metric).__name__}")
        print(f"   Reduction: {metric.reduction}")
        
        # Test metric computation
        pred = torch.randn(10, 5)
        target = torch.randn(10, 5)
        result = metric.compute(pred, target)
        print(f"   Computed metric result: {result:.6f}")
    except Exception as e:
        print(f"   Error creating metric with config: {e}")
    
    print()


def example_global_registry():
    """Example of using global registry."""
    print("=== Global Registry Example ===\n")
    
    # Register components with global registry
    register_component(
        name="global_preprocessor",
        component_type=ComponentType.PREPROCESSOR,
        component_class=SimplePreprocessor,
        factory_function=create_simple_preprocessor,
        version="1.0.0",
        description="Preprocessor registered globally",
        author="Global Author",
        category="preprocessing",
        tags=["global", "simple"]
    )
    
    print("1. Registered component with global registry:")
    component_names = list_components()
    print(f"   Registered components: {component_names}")
    
    component_info = get_component_info("global_preprocessor")
    if component_info:
        print(f"   Component info: {component_info.name} - {component_info.description}")
    
    # Create component using global registry
    try:
        preprocessor = create_component("global_preprocessor", normalize=True)
        print(f"2. Created component using global registry: {type(preprocessor).__name__}")
        print(f"   Normalize: {preprocessor.normalize}")
    except Exception as e:
        print(f"2. Error creating component: {e}")
    
    # Create component with config using global registry
    try:
        config = {'clip_values': False}
        register_component(
            name="global_postprocessor",
            component_type=ComponentType.POSTPROCESSOR,
            component_class=SimplePostprocessor,
            factory_function=create_simple_postprocessor
        )
        
        postprocessor = create_component_with_config("global_postprocessor", config)
        print(f"3. Created component with config using global registry: {type(postprocessor).__name__}")
        print(f"   Clip values: {postprocessor.clip_values}")
    except Exception as e:
        print(f"3. Error creating component with config: {e}")
    
    print()


def example_component_listing():
    """Example of listing components."""
    print("=== Component Listing Example ===\n")
    
    # Create registry and register various components
    registry = ComponentRegistry()
    
    # Register components of different types
    registry.register_component(
        name="data_normalizer",
        component_type=ComponentType.PREPROCESSOR,
        component_class=SimplePreprocessor,
        category="preprocessing",
        tags=["normalization", "data"]
    )
    
    registry.register_component(
        name="result_formatter",
        component_type=ComponentType.POSTPROCESSOR,
        component_class=SimplePostprocessor,
        category="postprocessing",
        tags=["formatting", "output"]
    )
    
    registry.register_component(
        name="accuracy_calculator",
        component_type=ComponentType.METRIC,
        component_class=SimpleMetric,
        category="evaluation",
        tags=["accuracy", "classification"]
    )
    
    registry.register_component(
        name="mse_loss",
        component_type=ComponentType.LOSS,
        component_class=SimpleLoss,
        category="training",
        tags=["mse", "regression"]
    )
    
    print("1. All registered components:")
    all_components = registry.list_components()
    for component_name in all_components:
        info = registry.get_component_info(component_name)
        print(f"   - {component_name} ({info.component_type.value})")
    print()
    
    print("2. Components by type:")
    component_types = registry.get_component_types()
    for comp_type in component_types:
        components = registry.list_components_by_type(comp_type)
        print(f"   {comp_type.value}: {components}")
    print()
    
    print("3. Components by category:")
    categories = registry.get_component_categories()
    for category in categories:
        components = registry.list_components_by_category(category)
        print(f"   {category}: {components}")
    print()
    
    print("4. Components by tag:")
    tags = registry.get_component_tags()
    for tag in tags:
        components = registry.list_components_by_tag(tag)
        print(f"   {tag}: {components}")
    print()


def example_component_usage():
    """Example of using components in a pipeline."""
    print("=== Component Usage Example ===\n")
    
    # Register and create components
    registry = ComponentRegistry()
    registry.register_component(
        name="pipeline_preprocessor",
        component_type=ComponentType.PREPROCESSOR,
        component_class=SimplePreprocessor,
        factory_function=create_simple_preprocessor
    )
    
    registry.register_component(
        name="pipeline_postprocessor",
        component_type=ComponentType.POSTPROCESSOR,
        component_class=SimplePostprocessor,
        factory_function=create_simple_postprocessor
    )
    
    registry.register_component(
        name="pipeline_metric",
        component_type=ComponentType.METRIC,
        component_class=SimpleMetric,
        factory_function=create_simple_metric
    )
    
    try:
        # Create components
        preprocessor = registry.create_component("pipeline_preprocessor", normalize=True)
        postprocessor = registry.create_component("pipeline_postprocessor", clip_values=True)
        metric = registry.create_component("pipeline_metric", reduction='mean')
        
        print("1. Created pipeline components:")
        print(f"   Preprocessor: {type(preprocessor).__name__}")
        print(f"   Postprocessor: {type(postprocessor).__name__}")
        print(f"   Metric: {type(metric).__name__}")
        
        # Simulate pipeline usage
        print("\n2. Simulating pipeline usage:")
        
        # Raw data
        raw_data = torch.randn(100, 10) * 10 + 5  # Large values
        print(f"   Raw data range: [{raw_data.min():.2f}, {raw_data.max():.2f}]")
        
        # Preprocess
        processed_data = preprocessor.preprocess(raw_data)
        print(f"   Processed data range: [{processed_data.min():.2f}, {processed_data.max():.2f}]")
        
        # Simulate model output (add some noise)
        model_output = processed_data + torch.randn_like(processed_data) * 0.1
        
        # Postprocess
        final_output = postprocessor.postprocess(model_output)
        print(f"   Final output range: [{final_output.min():.2f}, {final_output.max():.2f}]")
        
        # Evaluate
        target = processed_data  # Ideal target
        metric_result = metric.compute(final_output, target)
        print(f"   Metric result: {metric_result:.6f}")
        
    except Exception as e:
        print(f"   Error using components: {e}")
    
    print()


def example_component_dependencies():
    """Example of component dependencies."""
    print("=== Component Dependencies Example ===\n")
    
    registry = ComponentRegistry()
    
    # Register a component with dependencies
    registry.register_component(
        name="dependent_component",
        component_type=ComponentType.CUSTOM,
        description="Component with dependencies",
        dependencies=["base_component", "helper_component"]
    )
    
    print("1. Registered component with dependencies:")
    component_info = registry.get_component_info("dependent_component")
    if component_info:
        print(f"   Name: {component_info.name}")
        print(f"   Dependencies: {component_info.dependencies}")
    
    # Check dependencies (should fail since dependencies are not registered)
    deps_satisfied = registry.check_dependencies("dependent_component")
    print(f"2. Dependencies satisfied: {deps_satisfied}")
    
    # Register dependencies
    registry.register_component(
        name="base_component",
        component_type=ComponentType.CUSTOM,
        description="Base component"
    )
    
    registry.register_component(
        name="helper_component",
        component_type=ComponentType.CUSTOM,
        description="Helper component"
    )
    
    print("3. Registered dependencies:")
    print("   - base_component")
    print("   - helper_component")
    
    # Check dependencies again (should pass now)
    deps_satisfied = registry.check_dependencies("dependent_component")
    print(f"4. Dependencies satisfied: {deps_satisfied}")
    
    print()


def example_error_handling():
    """Example of error handling."""
    print("=== Error Handling Example ===\n")
    
    registry = ComponentRegistry()
    
    print("1. Trying to create unregistered component:")
    try:
        component = registry.create_component("nonexistent_component")
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    print("\n2. Registering component without class or factory:")
    registry.register_component(
        name="incomplete_component",
        component_type=ComponentType.CUSTOM,
        description="Component without creation method"
    )
    
    print("   Trying to create incomplete component:")
    try:
        component = registry.create_component("incomplete_component")
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    print()


if __name__ == "__main__":
    example_basic_registration()
    example_component_creation()
    example_global_registry()
    example_component_listing()
    example_component_usage()
    example_component_dependencies()
    example_error_handling()