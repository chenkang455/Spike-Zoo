"""
Metric utilities for SpikeZoo models.
"""

import torch
from typing import Tuple, Dict


def get_paired_images(
    batch: dict,
    outputs: dict,
    recon_key: str = "recon_img",
    gt_key: str = "gt_img"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get paired images for metric calculation.
    
    Args:
        batch: Input batch dictionary
        outputs: Model outputs dictionary
        recon_key: Key for reconstructed image in outputs
        gt_key: Key for ground truth image in batch
        
    Returns:
        Tuple of (reconstructed_image, ground_truth_image)
    """
    recon_img = outputs[recon_key]
    gt_img = batch[gt_key]
    return recon_img, gt_img


def prepare_visualization_dict(
    batch: dict,
    outputs: dict,
    recon_key: str = "recon_img",
    gt_key: str = "gt_img"
) -> Dict[str, torch.Tensor]:
    """
    Prepare visualization dictionary.
    
    Args:
        batch: Input batch dictionary
        outputs: Model outputs dictionary
        recon_key: Key for reconstructed image in outputs
        gt_key: Key for ground truth image in batch
        
    Returns:
        Visualization dictionary
    """
    visual_dict = {}
    visual_dict["recon_img"] = outputs[recon_key]
    visual_dict["gt_img"] = batch[gt_key]
    return visual_dict