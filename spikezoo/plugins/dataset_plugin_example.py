from spikezoo.core import register_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ExampleDataset(Dataset):
    """Example dataset for plugin demonstration."""
    
    def __init__(self, size=1000, input_size=784, output_size=10):
        """Initialize example dataset."""
        self.size = size
        self.input_size = input_size
        self.output_size = output_size
        
        # Generate random data for demonstration
        self.data = torch.randn(size, input_size)
        self.labels = torch.randint(0, output_size, (size,))
    
    def __len__(self):
        """Return dataset size."""
        return self.size
    
    def __getitem__(self, idx):
        """Get item by index."""
        return self.data[idx], self.labels[idx]


class ExampleDatasetConfig:
    """Configuration for example dataset."""
    
    def __init__(self, size=1000, input_size=784, output_size=10):
        """Initialize configuration."""
        self.size = size
        self.input_size = input_size
        self.output_size = output_size


def create_example_dataset(config=None):
    """
    Factory function to create example dataset.
    
    Args:
        config: Dataset configuration (optional)
        
    Returns:
        ExampleDataset instance
    """
    if config is None:
        return ExampleDataset()
    else:
        return ExampleDataset(
            size=config.size,
            input_size=config.input_size,
            output_size=config.output_size
        )


def register_datasets(registry):
    """
    Register datasets with the registry.
    
    Args:
        registry: DatasetRegistry instance
    """
    registry.register_dataset(
        name="example_dataset",
        dataset_class=ExampleDataset,
        config_class=ExampleDatasetConfig,
        factory_function=create_example_dataset,
        version="1.0.0",
        description="Example dataset for plugin demonstration",
        author="SpikeZoo Team",
        category="classification",
        tags=["example", "demo", "classification"]
    )


# Register with global registry when module is imported
register_datasets(register_dataset.__globals__['registry'] if 'registry' in register_dataset.__globals__ else 
                  __import__('spikezoo.core.dataset_registry').core.dataset_registry._dataset_registry)