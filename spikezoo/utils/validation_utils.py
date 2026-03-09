from typing import Any, Optional, Union, List
import numpy as np
import torch
import os
from pathlib import Path


class ParameterValidator:
    """Parameter validator for infer_from_* series methods."""
    
    @staticmethod
    def validate_dataset_index(idx: int, dataset_length: int) -> None:
        """
        Validate dataset index.
        
        Args:
            idx: Dataset index
            dataset_length: Length of dataset
            
        Raises:
            ValueError: If index is invalid
            TypeError: If index is not an integer
        """
        if not isinstance(idx, int):
            raise TypeError(f"Dataset index must be an integer, got {type(idx).__name__}")
        
        if idx < 0:
            raise ValueError(f"Dataset index must be non-negative, got {idx}")
        
        if idx >= dataset_length:
            raise ValueError(f"Dataset index {idx} is out of range [0, {dataset_length - 1}]")
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path]) -> None:
        """
        Validate file path.
        
        Args:
            file_path: Path to file
            
        Raises:
            ValueError: If file path is invalid
            TypeError: If file path is not a string or Path
            FileNotFoundError: If file does not exist
        """
        if not isinstance(file_path, (str, Path)):
            raise TypeError(f"File path must be a string or Path, got {type(file_path).__name__}")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
    
    @staticmethod
    def validate_dimensions(height: int, width: int) -> None:
        """
        Validate height and width dimensions.
        
        Args:
            height: Height dimension
            width: Width dimension
            
        Raises:
            ValueError: If dimensions are invalid
            TypeError: If dimensions are not integers
        """
        if not isinstance(height, int):
            raise TypeError(f"Height must be an integer, got {type(height).__name__}")
        
        if not isinstance(width, int):
            raise TypeError(f"Width must be an integer, got {type(width).__name__}")
        
        if height <= 0:
            raise ValueError(f"Height must be positive, got {height}")
        
        if width <= 0:
            raise ValueError(f"Width must be positive, got {width}")
    
    @staticmethod
    def validate_rate(rate: float) -> None:
        """
        Validate rate parameter.
        
        Args:
            rate: Rate value
            
        Raises:
            ValueError: If rate is invalid
            TypeError: If rate is not a number
        """
        if not isinstance(rate, (int, float)):
            raise TypeError(f"Rate must be a number, got {type(rate).__name__}")
        
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
    
    @staticmethod
    def validate_image_path(img_path: Optional[Union[str, Path]]) -> None:
        """
        Validate image path.
        
        Args:
            img_path: Path to image file (can be None)
            
        Raises:
            ValueError: If image path is invalid
            TypeError: If image path is not a string, Path, or None
            FileNotFoundError: If image file does not exist
        """
        if img_path is None:
            return
            
        if not isinstance(img_path, (str, Path)):
            raise TypeError(f"Image path must be a string, Path, or None, got {type(img_path).__name__}")
        
        img_path = Path(img_path)
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        if not img_path.is_file():
            raise ValueError(f"Image path is not a file: {img_path}")
    
    @staticmethod
    def validate_spike_data(spike: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Validate spike data.
        
        Args:
            spike: Spike data as numpy array or torch tensor
            
        Raises:
            ValueError: If spike data is invalid
            TypeError: If spike data is not a numpy array or torch tensor
        """
        if not isinstance(spike, (np.ndarray, torch.Tensor)):
            raise TypeError(f"Spike data must be a numpy array or torch tensor, got {type(spike).__name__}")
        
        if spike.size == 0:
            raise ValueError("Spike data cannot be empty")
        
        if spike.ndim not in [3, 4]:
            raise ValueError(f"Spike data must be 3D or 4D, got {spike.ndim}D")
    
    @staticmethod
    def validate_image_data(img: Optional[np.ndarray]) -> None:
        """
        Validate image data.
        
        Args:
            img: Image data as numpy array (can be None)
            
        Raises:
            ValueError: If image data is invalid
            TypeError: If image data is not a numpy array or None
        """
        if img is None:
            return
            
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image data must be a numpy array or None, got {type(img).__name__}")
        
        if img.size == 0:
            raise ValueError("Image data cannot be empty")
        
        if img.ndim not in [2, 3]:
            raise ValueError(f"Image data must be 2D or 3D, got {img.ndim}D")
    
    @staticmethod
    def validate_remove_head(remove_head: bool) -> None:
        """
        Validate remove_head parameter.
        
        Args:
            remove_head: Remove head flag
            
        Raises:
            TypeError: If remove_head is not a boolean
        """
        if not isinstance(remove_head, bool):
            raise TypeError(f"Remove head flag must be a boolean, got {type(remove_head).__name__}")


def validate_infer_from_dataset_params(
    idx: int, 
    dataset_length: int
) -> None:
    """
    Validate parameters for infer_from_dataset method.
    
    Args:
        idx: Dataset index
        dataset_length: Length of dataset
        
    Raises:
        ValueError: If parameters are invalid
        TypeError: If parameters are of wrong type
    """
    ParameterValidator.validate_dataset_index(idx, dataset_length)


def validate_infer_from_file_params(
    file_path: Union[str, Path],
    height: int,
    width: int,
    rate: float,
    img_path: Optional[Union[str, Path]],
    remove_head: bool
) -> None:
    """
    Validate parameters for infer_from_file method.
    
    Args:
        file_path: Path to spike data file
        height: Height dimension
        width: Width dimension
        rate: Rate value
        img_path: Path to ground truth image (optional)
        remove_head: Remove header flag
        
    Raises:
        ValueError: If parameters are invalid
        TypeError: If parameters are of wrong type
        FileNotFoundError: If files do not exist
    """
    ParameterValidator.validate_file_path(file_path)
    ParameterValidator.validate_dimensions(height, width)
    ParameterValidator.validate_rate(rate)
    ParameterValidator.validate_image_path(img_path)
    ParameterValidator.validate_remove_head(remove_head)


def validate_infer_from_spk_params(
    spike: Union[np.ndarray, torch.Tensor],
    rate: float,
    img: Optional[np.ndarray]
) -> None:
    """
    Validate parameters for infer_from_spk method.
    
    Args:
        spike: Spike data
        rate: Rate value
        img: Ground truth image (optional)
        
    Raises:
        ValueError: If parameters are invalid
        TypeError: If parameters are of wrong type
    """
    ParameterValidator.validate_spike_data(spike)
    ParameterValidator.validate_rate(rate)
    ParameterValidator.validate_image_data(img)