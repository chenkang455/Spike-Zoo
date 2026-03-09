"""
Loss functions for SpikeZoo models.
"""

import torch
import torch.nn as nn
from typing import Dict, Callable
from typing_extensions import Literal


class LossFunctionRegistry:
    """Registry for loss functions."""
    
    def __init__(self):
        """Initialize loss function registry."""
        self._loss_funcs: Dict[str, Callable] = {}
        self._initialize_default_losses()
    
    def _initialize_default_losses(self):
        """Initialize default loss functions."""
        self._loss_funcs["l1"] = nn.L1Loss()
        self._loss_funcs["l2"] = nn.MSELoss()
        self._loss_funcs["smooth_l1"] = nn.SmoothL1Loss()
    
    def register_loss(self, name: str, loss_func: Callable) -> None:
        """
        Register a loss function.
        
        Args:
            name: Loss function name
            loss_func: Loss function callable
        """
        self._loss_funcs[name] = loss_func
    
    def get_loss_function(self, name: str) -> Callable:
        """
        Get loss function by name.
        
        Args:
            name: Loss function name
            
        Returns:
            Loss function callable
            
        Raises:
            KeyError: If loss function not found
        """
        if name not in self._loss_funcs:
            # Return zero loss as fallback
            return lambda x, y: torch.tensor(0.0, device=x.device)
        return self._loss_funcs[name]
    
    def list_losses(self) -> list:
        """
        List all registered loss functions.
        
        Returns:
            List of loss function names
        """
        return list(self._loss_funcs.keys())


# Global loss function registry
_loss_registry = LossFunctionRegistry()


def register_loss_function(name: str, loss_func: Callable) -> None:
    """
    Register a loss function globally.
    
    Args:
        name: Loss function name
        loss_func: Loss function callable
    """
    _loss_registry.register_loss(name, loss_func)


def get_loss_function(name: Literal["l1", "l2", "smooth_l1"]) -> Callable:
    """
    Get loss function by name.
    
    Args:
        name: Loss function name
        
    Returns:
        Loss function callable
    """
    return _loss_registry.get_loss_function(name)


def list_loss_functions() -> list:
    """
    List all registered loss functions.
    
    Returns:
        List of loss function names
    """
    return _loss_registry.list_losses()


def compute_loss_dict(
    outputs: dict,
    batch: dict,
    loss_weight_dict: dict,
    target_key: str = "recon_img",
    gt_key: str = "gt_img"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Compute loss dictionary from outputs and batch.
    
    Args:
        outputs: Model outputs dictionary
        batch: Input batch dictionary
        loss_weight_dict: Dictionary mapping loss names to weights
        target_key: Key for target in outputs
        gt_key: Key for ground truth in batch
        
    Returns:
        Tuple of (loss_dict, loss_values_dict)
    """
    # Data process
    gt_img = batch[gt_key]
    target_img = outputs[target_key]
    
    # Compute losses
    loss_dict = {}
    for loss_name, weight in loss_weight_dict.items():
        loss_func = get_loss_function(loss_name)
        loss_dict[loss_name] = weight * loss_func(target_img, gt_img)
    
    # Convert to values dictionary
    loss_values_dict = {k: v.item() for k, v in loss_dict.items()}
    
    return loss_dict, loss_values_dict