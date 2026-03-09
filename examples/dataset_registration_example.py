#!/usr/bin/env python3
"""
Example of using the SpikeZoo dataset registration system.
"""

import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.core import (
    DatasetRegistry,
    DatasetInfo,
    register_dataset,
    unregister_dataset,
    get_dataset_info,
    list_datasets,
    create_dataset,
    create_dataset_with_config,
    get_dataset_registry,
    discover_datasets_from_directory
)


class SimpleDataset(Dataset):
    """Simple dataset for demonstration."""
    
    def __init__(self, size=100, input_size=10, output_size=1):
        """Initialize simple dataset."""
        self.size = size
        self.input_size = input_size
        self.output_size = output_size
        
        # Generate random data
        self.data = torch.randn(size, input_size)
        self.targets = torch.randn(size, output_size)
    
    def __len__(self):
        """Return dataset size."""
        return self.size
    
    def __getitem__(self, idx):
        """Get item by index."""
        return self.data[idx], self.targets[idx]


class SimpleDatasetConfig:
    """Configuration for simple dataset."""
    
    def __init__(self, size=100, input_size=10, output_size=1):
        """Initialize configuration."""
        self.size = size
        self.input_size = input_size
        self.output_size = output_size


def create_simple_dataset(config=None):
    """
    Factory function to create simple dataset.
    
    Args:
        config: Dataset configuration (optional)
        
    Returns:
        SimpleDataset instance
    """
    if config is None:
        return SimpleDataset()
    else:
        return SimpleDataset(
            size=config.size,
            input_size=config.input_size,
            output_size=config.output_size
        )


def example_basic_registration():
    """Example of basic dataset registration."""
    print("=== Basic Dataset Registration Example ===\n")
    
    # Create registry
    registry = DatasetRegistry()
    
    # Register dataset with class
    registry.register_dataset(
        name="simple_dataset",
        dataset_class=SimpleDataset,
        config_class=SimpleDatasetConfig,
        version="1.0.0",
        description="Simple dataset",
        author="Example Author",
        category="regression",
        tags=["linear", "simple"]
    )
    
    print("1. Registered dataset with class:")
    dataset_info = registry.get_dataset_info("simple_dataset")
    if dataset_info:
        print(f"   Name: {dataset_info.name}")
        print(f"   Version: {dataset_info.version}")
        print(f"   Description: {dataset_info.description}")
        print(f"   Category: {dataset_info.category}")
        print(f"   Tags: {dataset_info.tags}")
    print()
    
    # Register dataset with factory function
    registry.register_dataset(
        name="factory_dataset",
        factory_function=create_simple_dataset,
        version="1.0.0",
        description="Dataset created with factory function",
        author="Factory Author",
        category="regression",
        tags=["factory", "functional"]
    )
    
    print("2. Registered dataset with factory function:")
    dataset_info = registry.get_dataset_info("factory_dataset")
    if dataset_info:
        print(f"   Name: {dataset_info.name}")
        print(f"   Description: {dataset_info.description}")
        print(f"   Has factory function: {dataset_info.factory_function is not None}")
    print()


def example_dataset_creation():
    """Example of dataset creation."""
    print("=== Dataset Creation Example ===\n")
    
    # Create registry and register datasets
    registry = DatasetRegistry()
    registry.register_dataset(
        name="simple_dataset",
        dataset_class=SimpleDataset,
        config_class=SimpleDatasetConfig
    )
    
    registry.register_dataset(
        name="factory_dataset",
        factory_function=create_simple_dataset
    )
    
    # Create dataset instances
    print("1. Creating datasets:")
    
    # Create dataset with class
    try:
        dataset1 = registry.create_dataset("simple_dataset", size=200, input_size=20, output_size=2)
        print(f"   Created simple_dataset: {type(dataset1).__name__}")
        print(f"   Dataset size: {len(dataset1)}")
        print(f"   Sample item shape: {dataset1[0][0].shape}")
    except Exception as e:
        print(f"   Error creating simple_dataset: {e}")
    
    # Create dataset with factory function
    try:
        dataset2 = registry.create_dataset("factory_dataset")
        print(f"   Created factory_dataset: {type(dataset2).__name__}")
        print(f"   Dataset size: {len(dataset2)}")
    except Exception as e:
        print(f"   Error creating factory_dataset: {e}")
    
    # Create dataset with config
    try:
        config = SimpleDatasetConfig(size=150, input_size=15, output_size=3)
        dataset3 = registry.create_dataset_with_config("simple_dataset", config)
        print(f"   Created dataset with config: {type(dataset3).__name__}")
        print(f"   Configured size: {dataset3.size}")
        print(f"   Configured input size: {dataset3.input_size}")
        print(f"   Configured output size: {dataset3.output_size}")
    except Exception as e:
        print(f"   Error creating dataset with config: {e}")
    
    print()


def example_global_registry():
    """Example of using global registry."""
    print("=== Global Registry Example ===\n")
    
    # Register datasets with global registry
    register_dataset(
        name="global_simple_dataset",
        dataset_class=SimpleDataset,
        config_class=SimpleDatasetConfig,
        version="1.0.0",
        description="Simple dataset registered globally",
        author="Global Author",
        category="regression",
        tags=["global", "simple"]
    )
    
    print("1. Registered dataset with global registry:")
    dataset_names = list_datasets()
    print(f"   Registered datasets: {dataset_names}")
    
    dataset_info = get_dataset_info("global_simple_dataset")
    if dataset_info:
        print(f"   Dataset info: {dataset_info.name} - {dataset_info.description}")
    
    # Create dataset using global registry
    try:
        dataset = create_dataset("global_simple_dataset", size=250, input_size=25, output_size=5)
        print(f"2. Created dataset using global registry: {type(dataset).__name__}")
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Input size: {dataset.input_size}")
        print(f"   Output size: {dataset.output_size}")
    except Exception as e:
        print(f"2. Error creating dataset: {e}")
    
    # Create dataset with config using global registry
    try:
        config = SimpleDatasetConfig(size=300, input_size=30, output_size=1)
        dataset = create_dataset_with_config("global_simple_dataset", config)
        print(f"3. Created dataset with config using global registry: {type(dataset).__name__}")
        print(f"   Configured size: {dataset.size}")
    except Exception as e:
        print(f"3. Error creating dataset with config: {e}")
    
    print()


def example_dataset_listing():
    """Example of listing datasets."""
    print("=== Dataset Listing Example ===\n")
    
    # Create registry and register various datasets
    registry = DatasetRegistry()
    
    # Register datasets in different categories
    registry.register_dataset(
        name="linear_regression_data",
        dataset_class=SimpleDataset,
        category="regression",
        tags=["linear", "statistics"]
    )
    
    registry.register_dataset(
        name="image_classification_data",
        dataset_class=SimpleDataset,  # Using SimpleDataset for demo
        category="classification",
        tags=["image", "cnn"]
    )
    
    registry.register_dataset(
        name="sequence_data",
        dataset_class=SimpleDataset,  # Using SimpleDataset for demo
        category="sequence",
        tags=["nlp", "rnn"]
    )
    
    print("1. All registered datasets:")
    all_datasets = registry.list_datasets()
    for dataset_name in all_datasets:
        print(f"   - {dataset_name}")
    print()
    
    print("2. Datasets by category:")
    categories = registry.get_dataset_categories()
    for category in categories:
        datasets = registry.list_datasets_by_category(category)
        print(f"   {category}: {datasets}")
    print()
    
    print("3. Datasets by tag:")
    tags = registry.get_dataset_tags()
    for tag in tags:
        datasets = registry.list_datasets_by_tag(tag)
        print(f"   {tag}: {datasets}")
    print()


def example_dataset_usage():
    """Example of using datasets with DataLoader."""
    print("=== Dataset Usage Example ===\n")
    
    # Register and create dataset
    registry = DatasetRegistry()
    registry.register_dataset(
        name="usage_dataset",
        dataset_class=SimpleDataset
    )
    
    try:
        dataset = registry.create_dataset("usage_dataset", size=50, input_size=10, output_size=1)
        print(f"1. Created dataset: {type(dataset).__name__}")
        print(f"   Dataset size: {len(dataset)}")
        
        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Iterate through batches
        print("2. Using DataLoader:")
        for batch_idx, (data, targets) in enumerate(dataloader):
            print(f"   Batch {batch_idx}: data shape {data.shape}, targets shape {targets.shape}")
            if batch_idx >= 2:  # Just show first few batches
                break
        
        print(f"   Total batches: {len(dataloader)}")
    except Exception as e:
        print(f"   Error using dataset: {e}")
    
    print()


def example_error_handling():
    """Example of error handling."""
    print("=== Error Handling Example ===\n")
    
    registry = DatasetRegistry()
    
    print("1. Trying to create unregistered dataset:")
    try:
        dataset = registry.create_dataset("nonexistent_dataset")
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    print("\n2. Registering dataset without class or factory:")
    registry.register_dataset(name="incomplete_dataset", description="Dataset without creation method")
    
    print("   Trying to create incomplete dataset:")
    try:
        dataset = registry.create_dataset("incomplete_dataset")
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    print()


if __name__ == "__main__":
    example_basic_registration()
    example_dataset_creation()
    example_global_registry()
    example_dataset_listing()
    example_dataset_usage()
    example_error_handling()