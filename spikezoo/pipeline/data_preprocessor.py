import torch
import numpy as np
import cv2
from pathlib import Path
import os
from typing import Optional, Tuple, Union
from spikezoo.utils.spike_utils import load_vidar_dat


class DataPreprocessor:
    """Data preprocessor for infer_from_* series methods."""
    
    @staticmethod
    def preprocess_dataset_item(batch, device: str) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """
        Preprocess data from dataset item.
        
        Args:
            batch: Batch data from dataset
            device: Device to move tensors to
            
        Returns:
            Tuple of (spike, img, rate)
        """
        spike, img, rate = batch["spike"], batch["gt_img"], batch["rate"]
        spike = spike[None].to(device)
        
        if "with_img" in batch and batch["with_img"] == True:
            img = img[None].to(device)
        else:
            img = None
            
        return spike, img, rate
    
    @staticmethod
    def preprocess_file_input(
        file_path: str, 
        height: int, 
        width: int, 
        device: str,
        img_path: Optional[str] = None,
        remove_head: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Preprocess data from file input.
        
        Args:
            file_path: Path to spike data file (.dat or .npz)
            height: Height of spike data (for .dat files)
            width: Width of spike data (for .dat files)
            device: Device to move tensors to
            img_path: Path to ground truth image (optional)
            remove_head: Whether to remove header from .dat files
            
        Returns:
            Tuple of (spike, img)
        """
        # Load spike from .dat
        if file_path.endswith(".dat"):
            spike = load_vidar_dat(file_path, height, width, remove_head)
        # Load spike from .npz from UHSR
        elif file_path.endswith("npz"):
            spike = np.load(file_path)["spk"].astype(np.float32)[:, 13:237, 13:237]
        else:
            raise RuntimeError("Not recognized spike input file.")
            
        # Load img from .png/.jpg image file
        if img_path is not None:
            img = cv2.imread(img_path)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img / 255).astype(np.float32)
            img = torch.from_numpy(img)[None, None].to(device)
        else:
            img = None
            
        spike = torch.from_numpy(spike)[None].to(device)
        return spike, img
    
    @staticmethod
    def preprocess_spike_input(
        spike: Union[np.ndarray, torch.Tensor], 
        device: str,
        img: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Preprocess spike input data.
        
        Args:
            spike: Spike data as numpy array or torch tensor
            device: Device to move tensors to
            img: Ground truth image as numpy array (optional)
            
        Returns:
            Tuple of (spike, img)
        """
        # Spike process
        if isinstance(spike, np.ndarray):
            spike = torch.from_numpy(spike)
        spike = spike.to(device)
        # [c,h,w] -> [1,c,w,h]
        if spike.dim() == 3:
            spike = spike[None]
        spike = spike.float()
        
        # Img process
        if img is not None:
            if isinstance(img, np.ndarray):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                img = (img / 255).astype(np.float32)
                img = torch.from_numpy(img)[None, None].to(device)
            else:
                raise RuntimeError("Not recognized image input type.")
        else:
            img = None
            
        return spike, img
    
    @staticmethod
    def create_save_folder(base_save_folder: Path, method: str, *args) -> Path:
        """
        Create save folder for inference results.
        
        Args:
            base_save_folder: Base save folder path
            method: Method name ('dataset', 'file', 'spk')
            *args: Additional arguments for folder naming
            
        Returns:
            Save folder path
        """
        if method == "dataset":
            idx = args[0] if args else 0
            dataset_name = args[1] if len(args) > 1 else "unknown"
            split = args[2] if len(args) > 2 else "test"
            save_folder = base_save_folder / Path(f"infer_from_dataset/{dataset_name}_dataset/{split}/{idx:06d}")
        elif method == "file":
            file_path = args[0] if args else "unknown"
            save_folder = base_save_folder / Path(f"infer_from_file/{os.path.basename(file_path)}")
        elif method == "spk":
            save_folder = base_save_folder / Path(f"infer_from_spk")
        else:
            raise ValueError(f"Unknown method: {method}")
            
        os.makedirs(str(save_folder), exist_ok=True)
        return save_folder