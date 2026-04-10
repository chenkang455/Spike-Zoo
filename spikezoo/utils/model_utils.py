"""
Utility functions for model handling in SpikeZoo.
"""

import os
from typing import Optional


def get_suffix(model_name: str, version: str) -> str:
    """
    Get file suffix for model based on model name and version.
    
    Args:
        model_name: Name of the model
        version: Version identifier
        
    Returns:
        File suffix
    """
    # Default implementation - can be extended based on model requirements
    if version == "local":
        return "pth"
    else:
        # For remote versions, use the version string as part of suffix
        return f"{version}.pth"


def get_url_version(version: str) -> str:
    """
    Get URL version string for model downloading.
    
    Args:
        version: Version identifier
        
    Returns:
        URL version string
    """
    # Default implementation - can be extended based on model requirements
    if version == "local":
        return "latest"
    else:
        return version


def get_model_path(model_name: str, version: str, base_url: str, ckpt_path: Optional[str] = None) -> str:
    """
    Get full path or URL for model.
    
    Args:
        model_name: Name of the model
        version: Version identifier
        base_url: Base URL for remote models
        ckpt_path: Local checkpoint path (if provided)
        
    Returns:
        Full path or URL for model
    """
    if ckpt_path and version == "local":
        return ckpt_path
    else:
        url_version = get_url_version(version)
        suffix = get_suffix(model_name, version)
        return f"{base_url}/{model_name}/{url_version}/{model_name}.{suffix}"