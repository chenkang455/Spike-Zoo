from spikezoo.core import register_model
import torch
import torch.nn as nn


class ExampleModel(nn.Module):
    """Example model for plugin demonstration."""
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """Initialize example model."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Simple feedforward network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass."""
        x = x.view(-1, self.input_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ExampleModelConfig:
    """Configuration for example model."""
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """Initialize configuration."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


def create_example_model(config=None):
    """
    Factory function to create example model.
    
    Args:
        config: Model configuration (optional)
        
    Returns:
        ExampleModel instance
    """
    if config is None:
        return ExampleModel()
    else:
        return ExampleModel(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size
        )


def register_models(registry):
    """
    Register models with the registry.
    
    Args:
        registry: ModelRegistry instance
    """
    registry.register_model(
        name="example_model",
        model_class=ExampleModel,
        config_class=ExampleModelConfig,
        factory_function=create_example_model,
        version="1.0.0",
        description="Example model for plugin demonstration",
        author="SpikeZoo Team",
        category="classification",
        tags=["example", "demo", "classification"]
    )


# Register with global registry when module is imported
register_models(register_model.__globals__['registry'] if 'registry' in register_model.__globals__ else 
                __import__('spikezoo.core.model_registry').core.model_registry._model_registry)