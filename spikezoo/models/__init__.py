"""
Model package for SpikeZoo.
"""

from .base_model import BaseModel, BaseModelConfig
from .model_registry import (
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

__all__ = [
    "BaseModel",
    "BaseModelConfig",
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