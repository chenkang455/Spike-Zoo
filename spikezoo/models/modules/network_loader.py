"""
Network loading utilities for SpikeZoo models.
"""

import torch
import torch.nn as nn
import os
import time
from typing import Tuple
from spikezoo.utils.network_utils import (
    load_network_with_retry,
    download_file_with_retry
)
# Import these functions directly from utils to avoid circular import
from spikezoo.utils.model_utils import get_suffix, get_url_version


def load_model_weights(
    model: nn.Module,
    cfg,
    version: str,
    device: str
) -> nn.Module:
    """
    Load model weights based on configuration.
    
    Args:
        model: PyTorch model to load weights into
        cfg: Model configuration
        version: Model version
        device: Device to load model to
        
    Returns:
        Model with loaded weights
        
    Raises:
        RuntimeError: If model weights cannot be loaded
    """
    # Auto set the load_state to True under the eval mode
    if cfg.mode == "eval" and cfg.load_state == False:
        print(f"Method {cfg.model_name} on the evaluation mode, load_state is set to True automatically.")
        cfg.load_state = True
    
    # Load model weights if required
    if cfg.load_state and cfg.require_params:
        model = _load_weights_from_source(model, cfg, version)
    
    # Move to device
    model = model.to(device)
    model = nn.DataParallel(model) if cfg.multi_gpu == True else model
    
    return model


def _load_weights_from_source(
    model: nn.Module,
    cfg,
    version: str
) -> nn.Module:
    """
    Load weights from either local path or URL.
    
    Args:
        model: PyTorch model to load weights into
        cfg: Model configuration
        version: Model version
        
    Returns:
        Model with loaded weights
        
    Raises:
        RuntimeError: If model weights cannot be loaded
    """
    # Determine checkpoint path based on version
    if version != "local":
        load_folder = os.path.dirname(os.path.abspath(__file__))
        ckpt_name = f"{cfg.model_name}.{get_suffix(cfg.model_name, version)}"
        ckpt_path = os.path.join("weights", version, ckpt_name)
        ckpt_path = os.path.join(load_folder, ckpt_path)
        ckpt_path_url = os.path.join(cfg.base_url, get_url_version(version), ckpt_name)
    elif version == "local":
        ckpt_path = cfg.ckpt_path

    # Download checkpoint if not found locally (for non-local versions)
    if not os.path.isfile(ckpt_path) and version != "local":
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        success = download_file_with_retry(
            ckpt_path_url, 
            ckpt_path, 
            max_retries=cfg.max_retry_attempts,
            retry_delay=cfg.retry_delay
        )
        if not success:
            raise RuntimeError(
                f"Failed to download checkpoint from {ckpt_path_url} after {cfg.max_retry_attempts} attempts"
            )
        time.sleep(0.5)
    elif not os.path.isfile(ckpt_path) and version == "local":
        raise RuntimeError(
            f"For the method {cfg.model_name}, no ckpt can be found on the {ckpt_path} !!! Try set the version to get the model from the url."
        )
    
    # Load network with retry mechanism
    model = load_network_with_retry(
        ckpt_path, 
        model, 
        max_retries=cfg.max_retry_attempts,
        retry_delay=cfg.retry_delay
    )
    
    return model


def save_model_weights(model: nn.Module, save_path: str) -> None:
    """
    Save model weights to file.
    
    Args:
        model: PyTorch model to save
        save_path: Path to save weights
    """
    network = model
    if isinstance(network, nn.DataParallel):
        network = network.module
    
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    
    torch.save(state_dict, save_path)