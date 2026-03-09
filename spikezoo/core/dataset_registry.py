from typing import Dict, Type, Optional, Callable, Any
from dataclasses import dataclass, field
import importlib
import logging
from pathlib import Path
import os


@dataclass
class DatasetInfo:
    """Dataset information."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    category: str = "general"
    tags: list = field(default_factory=list)
    config_class: Optional[Type] = None
    dataset_class: Optional[Type] = None
    factory_function: Optional[Callable] = None


class DatasetRegistry:
    """Registry for managing datasets and their plugins."""
    
    def __init__(self):
        """Initialize dataset registry."""
        self.datasets: Dict[str, DatasetInfo] = {}
        self.factories: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
        self._discovered_modules = set()
    
    def register_dataset(
        self, 
        name: str,
        dataset_class: Optional[Type] = None,
        config_class: Optional[Type] = None,
        factory_function: Optional[Callable] = None,
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        category: str = "general",
        tags: Optional[list] = None
    ):
        """
        Register a dataset.
        
        Args:
            name: Dataset name
            dataset_class: Dataset class (optional)
            config_class: Configuration class (optional)
            factory_function: Factory function to create dataset instance (optional)
            version: Dataset version
            description: Dataset description
            author: Dataset author
            category: Dataset category
            tags: Dataset tags
        """
        if name in self.datasets:
            self.logger.warning(f"Dataset {name} already registered, overwriting")
        
        dataset_info = DatasetInfo(
            name=name,
            version=version,
            description=description,
            author=author,
            category=category,
            tags=tags or [],
            config_class=config_class,
            dataset_class=dataset_class,
            factory_function=factory_function
        )
        
        self.datasets[name] = dataset_info
        
        # Register factory function if provided
        if factory_function:
            self.factories[name] = factory_function
        
        self.logger.info(f"Registered dataset: {name}")
    
    def unregister_dataset(self, name: str):
        """
        Unregister a dataset.
        
        Args:
            name: Dataset name
        """
        if name in self.datasets:
            del self.datasets[name]
            self.logger.info(f"Unregistered dataset: {name}")
        
        if name in self.factories:
            del self.factories[name]
    
    def get_dataset_info(self, name: str) -> Optional[DatasetInfo]:
        """
        Get dataset information.
        
        Args:
            name: Dataset name
            
        Returns:
            DatasetInfo or None if not found
        """
        return self.datasets.get(name)
    
    def list_datasets(self) -> list:
        """
        List all registered datasets.
        
        Returns:
            List of dataset names
        """
        return list(self.datasets.keys())
    
    def list_datasets_by_category(self, category: str) -> list:
        """
        List datasets by category.
        
        Args:
            category: Dataset category
            
        Returns:
            List of dataset names in the category
        """
        return [name for name, info in self.datasets.items() if info.category == category]
    
    def list_datasets_by_tag(self, tag: str) -> list:
        """
        List datasets by tag.
        
        Args:
            tag: Dataset tag
            
        Returns:
            List of dataset names with the tag
        """
        return [name for name, info in self.datasets.items() if tag in info.tags]
    
    def create_dataset(self, name: str, *args, **kwargs) -> Any:
        """
        Create dataset instance.
        
        Args:
            name: Dataset name
            *args: Positional arguments for dataset constructor
            **kwargs: Keyword arguments for dataset constructor
            
        Returns:
            Dataset instance
            
        Raises:
            ValueError: If dataset not found or cannot be created
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found in registry")
        
        dataset_info = self.datasets[name]
        
        # Try factory function first
        if name in self.factories:
            try:
                return self.factories[name](*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to create dataset {name} using factory: {e}")
                raise
        
        # Try dataset class
        if dataset_info.dataset_class:
            try:
                return dataset_info.dataset_class(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to create dataset {name} using class: {e}")
                raise
        
        raise ValueError(f"Cannot create dataset {name}: no factory function or dataset class available")
    
    def create_dataset_with_config(self, name: str, config: Any) -> Any:
        """
        Create dataset instance with configuration.
        
        Args:
            name: Dataset name
            config: Dataset configuration
            
        Returns:
            Dataset instance
            
        Raises:
            ValueError: If dataset not found or cannot be created
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found in registry")
        
        dataset_info = self.datasets[name]
        
        # Try factory function with config
        if name in self.factories:
            try:
                return self.factories[name](config)
            except Exception as e:
                self.logger.error(f"Failed to create dataset {name} using factory with config: {e}")
                raise
        
        # Try dataset class with config
        if dataset_info.dataset_class:
            try:
                return dataset_info.dataset_class(config)
            except Exception as e:
                self.logger.error(f"Failed to create dataset {name} using class with config: {e}")
                raise
        
        raise ValueError(f"Cannot create dataset {name}: no factory function or dataset class available")
    
    def discover_datasets_from_directory(self, directory: str, package_prefix: str = ""):
        """
        Discover and register datasets from a directory.
        
        Args:
            directory: Directory to search for datasets
            package_prefix: Package prefix for import
        """
        directory = Path(directory)
        if not directory.exists():
            self.logger.warning(f"Directory {directory} does not exist")
            return
        
        # Walk through directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("_dataset.py"):
                    module_name = file[:-3]  # Remove .py extension
                    full_module_name = f"{package_prefix}.{module_name}" if package_prefix else module_name
                    
                    # Skip if already discovered
                    if full_module_name in self._discovered_modules:
                        continue
                    
                    try:
                        # Import module
                        module = importlib.import_module(full_module_name)
                        self._discovered_modules.add(full_module_name)
                        
                        # Look for dataset registration function
                        if hasattr(module, "register_datasets"):
                            module.register_datasets(self)
                            self.logger.info(f"Discovered datasets from {full_module_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to import module {full_module_name}: {e}")
    
    def get_dataset_categories(self) -> list:
        """
        Get all dataset categories.
        
        Returns:
            List of unique categories
        """
        categories = set()
        for info in self.datasets.values():
            categories.add(info.category)
        return list(categories)
    
    def get_dataset_tags(self) -> list:
        """
        Get all dataset tags.
        
        Returns:
            List of unique tags
        """
        tags = set()
        for info in self.datasets.values():
            tags.update(info.tags)
        return list(tags)


# Global dataset registry instance
_dataset_registry = DatasetRegistry()


def register_dataset(
    name: str,
    dataset_class: Optional[Type] = None,
    config_class: Optional[Type] = None,
    factory_function: Optional[Callable] = None,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    category: str = "general",
    tags: Optional[list] = None
):
    """
    Register a dataset in the global registry.
    
    Args:
        name: Dataset name
        dataset_class: Dataset class (optional)
        config_class: Configuration class (optional)
        factory_function: Factory function to create dataset instance (optional)
        version: Dataset version
        description: Dataset description
        author: Dataset author
        category: Dataset category
        tags: Dataset tags
    """
    _dataset_registry.register_dataset(
        name, dataset_class, config_class, factory_function,
        version, description, author, category, tags
    )


def unregister_dataset(name: str):
    """
    Unregister a dataset from the global registry.
    
    Args:
        name: Dataset name
    """
    _dataset_registry.unregister_dataset(name)


def get_dataset_info(name: str) -> Optional[DatasetInfo]:
    """
    Get dataset information from the global registry.
    
    Args:
        name: Dataset name
        
    Returns:
        DatasetInfo or None if not found
    """
    return _dataset_registry.get_dataset_info(name)


def list_datasets() -> list:
    """
    List all registered datasets from the global registry.
    
    Returns:
        List of dataset names
    """
    return _dataset_registry.list_datasets()


def create_dataset(name: str, *args, **kwargs) -> Any:
    """
    Create dataset instance from the global registry.
    
    Args:
        name: Dataset name
        *args: Positional arguments for dataset constructor
        **kwargs: Keyword arguments for dataset constructor
        
    Returns:
        Dataset instance
    """
    return _dataset_registry.create_dataset(name, *args, **kwargs)


def create_dataset_with_config(name: str, config: Any) -> Any:
    """
    Create dataset instance with configuration from the global registry.
    
    Args:
        name: Dataset name
        config: Dataset configuration
        
    Returns:
        Dataset instance
    """
    return _dataset_registry.create_dataset_with_config(name, config)


def get_dataset_registry() -> DatasetRegistry:
    """
    Get the global dataset registry instance.
    
    Returns:
        DatasetRegistry instance
    """
    return _dataset_registry


def discover_datasets_from_directory(directory: str, package_prefix: str = ""):
    """
    Discover and register datasets from a directory in the global registry.
    
    Args:
        directory: Directory to search for datasets
        package_prefix: Package prefix for import
    """
    _dataset_registry.discover_datasets_from_directory(directory, package_prefix)