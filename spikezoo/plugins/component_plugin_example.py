from spikezoo.core import register_component, ComponentType
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ExamplePreprocessor:
    """Example preprocessor component."""
    
    def __init__(self, config=None):
        """Initialize preprocessor."""
        self.config = config or {}
        self.normalize = self.config.get('normalize', True)
    
    def preprocess(self, data):
        """Preprocess data."""
        if self.normalize:
            data = (data - data.mean()) / (data.std() + 1e-8)
        return data


class ExamplePostprocessor:
    """Example postprocessor component."""
    
    def __init__(self, config=None):
        """Initialize postprocessor."""
        self.config = config or {}
        self.clip_values = self.config.get('clip_values', True)
    
    def postprocess(self, data):
        """Postprocess data."""
        if self.clip_values:
            data = torch.clamp(data, 0, 1)
        return data


class ExampleMetric:
    """Example metric component."""
    
    def __init__(self, config=None):
        """Initialize metric."""
        self.config = config or {}
        self.reduction = self.config.get('reduction', 'mean')
    
    def compute(self, pred, target):
        """Compute metric."""
        mse = torch.mean((pred - target) ** 2)
        return mse.item()


class ExampleLoss:
    """Example loss component."""
    
    def __init__(self, config=None):
        """Initialize loss."""
        self.config = config or {}
        self.reduction = self.config.get('reduction', 'mean')
    
    def compute(self, pred, target):
        """Compute loss."""
        loss = torch.mean((pred - target) ** 2)
        return loss


def create_example_preprocessor(config=None):
    """Factory function for example preprocessor."""
    return ExamplePreprocessor(config)


def create_example_postprocessor(config=None):
    """Factory function for example postprocessor."""
    return ExamplePostprocessor(config)


def create_example_metric(config=None):
    """Factory function for example metric."""
    return ExampleMetric(config)


def create_example_loss(config=None):
    """Factory function for example loss."""
    return ExampleLoss(config)


def register_components(registry):
    """
    Register components with the registry.
    
    Args:
        registry: ComponentRegistry instance
    """
    # Register preprocessor
    registry.register_component(
        name="example_preprocessor",
        component_type=ComponentType.PREPROCESSOR,
        component_class=ExamplePreprocessor,
        factory_function=create_example_preprocessor,
        version="1.0.0",
        description="Example preprocessor component",
        author="SpikeZoo Team",
        category="preprocessing",
        tags=["example", "normalization"]
    )
    
    # Register postprocessor
    registry.register_component(
        name="example_postprocessor",
        component_type=ComponentType.POSTPROCESSOR,
        component_class=ExamplePostprocessor,
        factory_function=create_example_postprocessor,
        version="1.0.0",
        description="Example postprocessor component",
        author="SpikeZoo Team",
        category="postprocessing",
        tags=["example", "clipping"]
    )
    
    # Register metric
    registry.register_component(
        name="example_metric",
        component_type=ComponentType.METRIC,
        component_class=ExampleMetric,
        factory_function=create_example_metric,
        version="1.0.0",
        description="Example metric component",
        author="SpikeZoo Team",
        category="evaluation",
        tags=["example", "mse"]
    )
    
    # Register loss
    registry.register_component(
        name="example_loss",
        component_type=ComponentType.LOSS,
        component_class=ExampleLoss,
        factory_function=create_example_loss,
        version="1.0.0",
        description="Example loss component",
        author="SpikeZoo Team",
        category="training",
        tags=["example", "mse"],
        dependencies=[]
    )


# Register with global registry when module is imported
register_components(register_component.__globals__['registry'] if 'registry' in register_component.__globals__ else 
                   __import__('spikezoo.core.component_registry').core.component_registry._component_registry)