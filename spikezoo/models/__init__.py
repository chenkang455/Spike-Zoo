import importlib
import inspect
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from spikezoo.models.model_registry import (
    ModelRegistry,
    register_model,
    unregister_model,
    get_model_class,
    get_config_class,
    create_model,
    list_models,
    is_model_registered,
    get_model_registry
)

from spikezoo.utils.other_utils import getattr_case_insensitive
import os
from pathlib import Path

# Initialize model registry
_model_registry = get_model_registry()

# Backward compatibility - list of available models
current_file_path = Path(__file__).parent
files_list = os.listdir(os.path.dirname(os.path.abspath(__file__)))
model_list = [file.split("_")[0] for file in files_list if file.endswith("_model.py")]


def build_model_cfg(cfg: BaseModelConfig):
    """
    Build the model from the given model config.
    
    Args:
        cfg: BaseModelConfig instance
        
    Returns:
        Model instance
    """
    # Use model_cls_local if provided
    if cfg.model_cls_local is not None:
        model_cls = cfg.model_cls_local
        model = model_cls(cfg)
        return model
    
    # Use registry to create model
    model = create_model(cfg.model_name, cfg)
    if model is not None:
        return model
    
    # Fallback to old method for backward compatibility
    module_name = cfg.model_name + "_model"
    assert cfg.model_name in model_list, f"Given model {cfg.model_name} not in our model zoo {model_list}."
    module_name = "spikezoo.models." + module_name
    module = importlib.import_module(module_name)
    
    # Get model class
    model_name = cfg.model_name
    model_name = model_name + 'Model' if model_name == "base" else model_name
    model_cls: BaseModel = getattr_case_insensitive(module, model_name)
    model = model_cls(cfg)
    return model


def build_model_name(model_name: str):
    """
    Build the default model from the given name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model instance
    """
    # Use registry to create model
    model = create_model(model_name)
    if model is not None:
        return model
    
    # Fallback to old method for backward compatibility
    assert model_name in model_list, f"Given model {model_name} not in our model zoo {model_list}."
    module_name = model_name + "_model"
    module_name = "spikezoo.models." + module_name
    module = importlib.import_module(module_name)
    
    # Get model class and config
    model_name = model_name + 'Model' if model_name == "base" else model_name
    model_cls: BaseModel = getattr_case_insensitive(module, model_name)
    model_cfg: BaseModelConfig = getattr_case_insensitive(module, model_name + 'Config')()
    model = model_cls(model_cfg)
    return model


# Export registry functions for direct use
__all__ = [
    "BaseModel",
    "BaseModelConfig",
    "build_model_cfg",
    "build_model_name",
    "ModelRegistry",
    "register_model",
    "unregister_model",
    "get_model_class",
    "get_config_class",
    "create_model",
    "list_models",
    "is_model_registered",
    "get_model_registry"
]
