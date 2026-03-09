from typing import Dict, Type, Optional, Union
import importlib
from pathlib import Path
import os
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from spikezoo.utils.other_utils import getattr_case_insensitive


class ModelRegistry:
    """Registry for managing model classes and their dynamic imports."""
    
    def __init__(self):
        """Initialize model registry."""
        self._model_classes: Dict[str, Type[BaseModel]] = {}
        self._model_configs: Dict[str, Type[BaseModelConfig]] = {}
        self._module_paths: Dict[str, str] = {}
        self._loaded_modules: Dict[str, object] = {}
        
        # Discover available models
        self._discover_models()
    
    def _discover_models(self):
        """Discover available models in the models directory."""
        models_dir = Path(__file__).parent
        files_list = os.listdir(models_dir)
        model_files = [file for file in files_list if file.endswith("_model.py")]
        
        for file in model_files:
            model_name = file.split("_")[0]
            module_name = f"spikezoo.models.{model_name}_model"
            self._module_paths[model_name] = module_name
    
    def register_model(
        self, 
        model_name: str, 
        model_class: Type[BaseModel],
        config_class: Optional[Type[BaseModelConfig]] = None,
        module_path: Optional[str] = None
    ):
        """
        Register a model class manually.
        
        Args:
            model_name: Name of the model
            model_class: Model class
            config_class: Configuration class for the model (optional)
            module_path: Module path for dynamic import (optional)
        """
        self._model_classes[model_name] = model_class
        if config_class:
            self._model_configs[model_name] = config_class
        if module_path:
            self._module_paths[model_name] = module_path
    
    def unregister_model(self, model_name: str):
        """
        Unregister a model class.
        
        Args:
            model_name: Name of the model to unregister
        """
        if model_name in self._model_classes:
            del self._model_classes[model_name]
        if model_name in self._model_configs:
            del self._model_configs[model_name]
        if model_name in self._module_paths:
            del self._module_paths[model_name]
        if model_name in self._loaded_modules:
            del self._loaded_modules[model_name]
    
    def get_model_class(self, model_name: str) -> Optional[Type[BaseModel]]:
        """
        Get model class by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model class or None if not found
        """
        # Return cached class if available
        if model_name in self._model_classes:
            return self._model_classes[model_name]
        
        # Try to dynamically import
        if model_name in self._module_paths:
            try:
                module_path = self._module_paths[model_name]
                module = importlib.import_module(module_path)
                self._loaded_modules[model_name] = module
                
                # Get model class
                class_name = model_name + 'Model' if model_name == "base" else model_name
                model_class = getattr_case_insensitive(module, class_name)
                
                if model_class and issubclass(model_class, BaseModel):
                    self._model_classes[model_name] = model_class
                    return model_class
            except Exception as e:
                print(f"Failed to import model {model_name}: {e}")
                return None
        
        return None
    
    def get_config_class(self, model_name: str) -> Optional[Type[BaseModelConfig]]:
        """
        Get configuration class by model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Configuration class or None if not found
        """
        # Return cached config class if available
        if model_name in self._model_configs:
            return self._model_configs[model_name]
        
        # Try to dynamically import
        if model_name in self._module_paths:
            try:
                module_path = self._module_paths[model_name]
                module = importlib.import_module(module_path)
                self._loaded_modules[model_name] = module
                
                # Get config class
                class_name = model_name + 'Model' if model_name == "base" else model_name
                config_class = getattr_case_insensitive(module, class_name + 'Config')
                
                if config_class and issubclass(config_class, BaseModelConfig):
                    self._model_configs[model_name] = config_class
                    return config_class
            except Exception as e:
                print(f"Failed to import config for model {model_name}: {e}")
                return None
        
        return None
    
    def create_model(
        self, 
        model_name: str, 
        config: Optional[BaseModelConfig] = None
    ) -> Optional[BaseModel]:
        """
        Create model instance.
        
        Args:
            model_name: Name of the model
            config: Configuration for the model (optional)
            
        Returns:
            Model instance or None if creation failed
        """
        model_class = self.get_model_class(model_name)
        if model_class is None:
            print(f"Model class for {model_name} not found")
            return None
        
        # Create config if not provided
        if config is None:
            config_class = self.get_config_class(model_name)
            if config_class is None:
                print(f"Config class for {model_name} not found")
                return None
            config = config_class()
        
        try:
            model = model_class(config)
            return model
        except Exception as e:
            print(f"Failed to create model {model_name}: {e}")
            return None
    
    def list_models(self) -> list:
        """
        List all available models.
        
        Returns:
            List of model names
        """
        return list(self._module_paths.keys())
    
    def is_model_registered(self, model_name: str) -> bool:
        """
        Check if model is registered.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is registered, False otherwise
        """
        return model_name in self._model_classes or model_name in self._module_paths


# Global model registry instance
_model_registry = ModelRegistry()


def register_model(
    model_name: str, 
    model_class: Type[BaseModel],
    config_class: Optional[Type[BaseModelConfig]] = None,
    module_path: Optional[str] = None
):
    """
    Register a model class globally.
    
    Args:
        model_name: Name of the model
        model_class: Model class
        config_class: Configuration class for the model (optional)
        module_path: Module path for dynamic import (optional)
    """
    _model_registry.register_model(model_name, model_class, config_class, module_path)


def unregister_model(model_name: str):
    """
    Unregister a model class globally.
    
    Args:
        model_name: Name of the model to unregister
    """
    _model_registry.unregister_model(model_name)


def get_model_class(model_name: str) -> Optional[Type[BaseModel]]:
    """
    Get model class by name from global registry.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model class or None if not found
    """
    return _model_registry.get_model_class(model_name)


def get_config_class(model_name: str) -> Optional[Type[BaseModelConfig]]:
    """
    Get configuration class by model name from global registry.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Configuration class or None if not found
    """
    return _model_registry.get_config_class(model_name)


def create_model(
    model_name: str, 
    config: Optional[BaseModelConfig] = None
) -> Optional[BaseModel]:
    """
    Create model instance from global registry.
    
    Args:
        model_name: Name of the model
        config: Configuration for the model (optional)
        
    Returns:
        Model instance or None if creation failed
    """
    return _model_registry.create_model(model_name, config)


def list_models() -> list:
    """
    List all available models from global registry.
    
    Returns:
        List of model names
    """
    return _model_registry.list_models()


def is_model_registered(model_name: str) -> bool:
    """
    Check if model is registered in global registry.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if model is registered, False otherwise
    """
    return _model_registry.is_model_registered(model_name)


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Returns:
        ModelRegistry instance
    """
    return _model_registry