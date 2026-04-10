from typing import Dict, Type, Optional, Callable, Any
from dataclasses import dataclass, field
import importlib
import logging
from pathlib import Path
import os


@dataclass
class ModelInfo:
    """Model information."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    category: str = "general"
    tags: list = field(default_factory=list)
    config_class: Optional[Type] = None
    model_class: Optional[Type] = None
    factory_function: Optional[Callable] = None


class ModelRegistry:
    """Registry for managing models and their plugins."""
    
    def __init__(self):
        """Initialize model registry."""
        self.models: Dict[str, ModelInfo] = {}
        self.factories: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
        self._discovered_modules = set()
    
    def register_model(
        self, 
        name: str,
        model_class: Optional[Type] = None,
        config_class: Optional[Type] = None,
        factory_function: Optional[Callable] = None,
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        category: str = "general",
        tags: Optional[list] = None
    ):
        """
        Register a model.
        
        Args:
            name: Model name
            model_class: Model class (optional)
            config_class: Configuration class (optional)
            factory_function: Factory function to create model instance (optional)
            version: Model version
            description: Model description
            author: Model author
            category: Model category
            tags: Model tags
        """
        if name in self.models:
            self.logger.warning(f"Model {name} already registered, overwriting")
        
        model_info = ModelInfo(
            name=name,
            version=version,
            description=description,
            author=author,
            category=category,
            tags=tags or [],
            config_class=config_class,
            model_class=model_class,
            factory_function=factory_function
        )
        
        self.models[name] = model_info
        
        # Register factory function if provided
        if factory_function:
            self.factories[name] = factory_function
        
        self.logger.info(f"Registered model: {name}")
    
    def unregister_model(self, name: str):
        """
        Unregister a model.
        
        Args:
            name: Model name
        """
        if name in self.models:
            del self.models[name]
            self.logger.info(f"Unregistered model: {name}")
        
        if name in self.factories:
            del self.factories[name]
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """
        Get model information.
        
        Args:
            name: Model name
            
        Returns:
            ModelInfo or None if not found
        """
        return self.models.get(name)
    
    def list_models(self) -> list:
        """
        List all registered models.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def list_models_by_category(self, category: str) -> list:
        """
        List models by category.
        
        Args:
            category: Model category
            
        Returns:
            List of model names in the category
        """
        return [name for name, info in self.models.items() if info.category == category]
    
    def list_models_by_tag(self, tag: str) -> list:
        """
        List models by tag.
        
        Args:
            tag: Model tag
            
        Returns:
            List of model names with the tag
        """
        return [name for name, info in self.models.items() if tag in info.tags]
    
    def create_model(self, name: str, *args, **kwargs) -> Any:
        """
        Create model instance.
        
        Args:
            name: Model name
            *args: Positional arguments for model constructor
            **kwargs: Keyword arguments for model constructor
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model not found or cannot be created
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found in registry")
        
        model_info = self.models[name]
        
        # Try factory function first
        if name in self.factories:
            try:
                return self.factories[name](*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to create model {name} using factory: {e}")
                raise
        
        # Try model class
        if model_info.model_class:
            try:
                return model_info.model_class(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to create model {name} using class: {e}")
                raise
        
        raise ValueError(f"Cannot create model {name}: no factory function or model class available")
    
    def create_model_with_config(self, name: str, config: Any) -> Any:
        """
        Create model instance with configuration.
        
        Args:
            name: Model name
            config: Model configuration
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model not found or cannot be created
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found in registry")
        
        model_info = self.models[name]
        
        # Try factory function with config
        if name in self.factories:
            try:
                return self.factories[name](config)
            except Exception as e:
                self.logger.error(f"Failed to create model {name} using factory with config: {e}")
                raise
        
        # Try model class with config
        if model_info.model_class:
            try:
                return model_info.model_class(config)
            except Exception as e:
                self.logger.error(f"Failed to create model {name} using class with config: {e}")
                raise
        
        raise ValueError(f"Cannot create model {name}: no factory function or model class available")
    
    def discover_models_from_directory(self, directory: str, package_prefix: str = ""):
        """
        Discover and register models from a directory.
        
        Args:
            directory: Directory to search for models
            package_prefix: Package prefix for import
        """
        directory = Path(directory)
        if not directory.exists():
            self.logger.warning(f"Directory {directory} does not exist")
            return
        
        # Walk through directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("_model.py"):
                    module_name = file[:-3]  # Remove .py extension
                    full_module_name = f"{package_prefix}.{module_name}" if package_prefix else module_name
                    
                    # Skip if already discovered
                    if full_module_name in self._discovered_modules:
                        continue
                    
                    try:
                        # Import module
                        module = importlib.import_module(full_module_name)
                        self._discovered_modules.add(full_module_name)
                        
                        # Look for model registration function
                        if hasattr(module, "register_models"):
                            module.register_models(self)
                            self.logger.info(f"Discovered models from {full_module_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to import module {full_module_name}: {e}")
    
    def get_model_categories(self) -> list:
        """
        Get all model categories.
        
        Returns:
            List of unique categories
        """
        categories = set()
        for info in self.models.values():
            categories.add(info.category)
        return list(categories)
    
    def get_model_tags(self) -> list:
        """
        Get all model tags.
        
        Returns:
            List of unique tags
        """
        tags = set()
        for info in self.models.values():
            tags.update(info.tags)
        return list(tags)


# Global model registry instance
_model_registry = ModelRegistry()


def register_model(
    name: str,
    model_class: Optional[Type] = None,
    config_class: Optional[Type] = None,
    factory_function: Optional[Callable] = None,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    category: str = "general",
    tags: Optional[list] = None
):
    """
    Register a model in the global registry.
    
    Args:
        name: Model name
        model_class: Model class (optional)
        config_class: Configuration class (optional)
        factory_function: Factory function to create model instance (optional)
        version: Model version
        description: Model description
        author: Model author
        category: Model category
        tags: Model tags
    """
    _model_registry.register_model(
        name, model_class, config_class, factory_function,
        version, description, author, category, tags
    )


def unregister_model(name: str):
    """
    Unregister a model from the global registry.
    
    Args:
        name: Model name
    """
    _model_registry.unregister_model(name)


def get_model_info(name: str) -> Optional[ModelInfo]:
    """
    Get model information from the global registry.
    
    Args:
        name: Model name
        
    Returns:
        ModelInfo or None if not found
    """
    return _model_registry.get_model_info(name)


def list_models() -> list:
    """
    List all registered models from the global registry.
    
    Returns:
        List of model names
    """
    return _model_registry.list_models()


def create_model(name: str, *args, **kwargs) -> Any:
    """
    Create model instance from the global registry.
    
    Args:
        name: Model name
        *args: Positional arguments for model constructor
        **kwargs: Keyword arguments for model constructor
        
    Returns:
        Model instance
    """
    return _model_registry.create_model(name, *args, **kwargs)


def create_model_with_config(name: str, config: Any) -> Any:
    """
    Create model instance with configuration from the global registry.
    
    Args:
        name: Model name
        config: Model configuration
        
    Returns:
        Model instance
    """
    return _model_registry.create_model_with_config(name, config)


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Returns:
        ModelRegistry instance
    """
    return _model_registry


def discover_models_from_directory(directory: str, package_prefix: str = ""):
    """
    Discover and register models from a directory in the global registry.
    
    Args:
        directory: Directory to search for models
        package_prefix: Package prefix for import
    """
    _model_registry.discover_models_from_directory(directory, package_prefix)