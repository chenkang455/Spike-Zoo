"""
Base model for SpikeZoo.
"""

import torch
import torch.nn as nn
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Union, List, Tuple
import os
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import functools
import torch.nn as nn
from spikezoo.archs.base.nets import BaseNet
from spikezoo.models.modules.network_loader import (
    load_model_weights,
    save_model_weights
)
from spikezoo.models.modules.loss_functions import compute_loss_dict
from spikezoo.models.modules.metric_utils import (
    get_paired_images,
    prepare_visualization_dict
)
from spikezoo.utils import get_suffix, get_url_version


@dataclass
class BaseModelConfig:
    # ------------- Not Recommended to Change -------------
    "Registerd model name."
    model_name: str = "base"
    "File name of the specified model."
    model_file_name: str = "nets"
    "Class name of the specified model in spikezoo/archs/base/{model_file_name}.py."
    model_cls_name: str = "BaseNet"
    "Spike input length. (local mode)"
    model_length: int = 41
    "Spike input length for different versions."
    model_length_dict: dict = field(default_factory=lambda: {"v010": 41, "v023": 41})
    "Model require model parameters or not."
    require_params: bool = True
    "Model parameters. (local mode)"
    model_params: dict = field(default_factory=lambda: {})
    "Model parameters for different versions."
    model_params_dict: dict = field(default_factory=lambda: {"v010": {}, "v023": {}})
    # ------------- Config -------------
    "Load ckpt path. Used on the local mode."
    ckpt_path: str = ""
    "Load pretrained weights or not. (default false, set to true during the evaluation mode.)"
    load_state: bool = False
    "Multi-GPU setting."
    multi_gpu: bool = False
    "Base url."
    base_url: str = "https://github.com/chenkang455/Spike-Zoo/releases/download"
    "Load the model from local class or spikezoo lib. (None)"
    model_cls_local: Optional[nn.Module] = None
    "Load the arch from local class or spikezoo lib. (None)"
    arch_cls_local: Optional[nn.Module] = None
    # ------------- Retry Config -------------
    "Maximum number of retry attempts for network loading."
    max_retry_attempts: int = 3
    "Delay between retry attempts in seconds."
    retry_delay: float = 1.0
    # ------------- Mode Config -------------
    "Model mode (train/eval)."
    mode: str = "train"


class BaseModel:
    def __init__(self, cfg: BaseModelConfig):
        self.cfg = cfg
        self.net = None
        self.device = None

    def build_network(self, mode: str = "train", version: str = "local"):
        """
        Build the network.
        
        Args:
            mode: Model mode (train/eval)
            version: Model version
        """
        self.cfg.mode = mode
        
        # [1] build the model.
        if self.cfg.model_cls_local == None:
            module_name = self.cfg.model_file_name
            module_name = "spikezoo.archs." + self.cfg.model_name + "." + module_name
            module = importlib.import_module(module_name)
            model_cls: BaseNet = getattr(module, self.cfg.model_cls_name)
        else:
            model_cls: BaseNet = self.cfg.model_cls_local
        self.net = model_cls(**self.cfg.model_params)
        
        # [2] build the network.
        self.net = load_model_weights(self.net, self.cfg, version, self.device)
        return self

    def feed_to_device(self, batch):
        """Feed batch to device."""
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        return batch

    def get_outputs_dict(self, batch):
        """Get outputs dictionary."""
        outputs = {}
        spike = batch["spike"]
        recon_img = self.net(spike)
        outputs["recon_img"] = recon_img
        return outputs

    def get_loss_dict(self, outputs, batch, loss_weight_dict):
        """Get loss dictionary."""
        return compute_loss_dict(outputs, batch, loss_weight_dict)

    def get_visual_dict(self, batch, outputs):
        """Get visualization dictionary."""
        return prepare_visualization_dict(batch, outputs)

    def get_paired_images(self, batch, outputs):
        """Get paired images for metrics."""
        return get_paired_images(batch, outputs)

    def save_network(self, save_path, net=None):
        """Save network weights."""
        network = net if net is not None else self.net
        save_model_weights(network, save_path)

    def spk2img(self, spike):
        """Spike to image conversion."""
        return self.net(spike)