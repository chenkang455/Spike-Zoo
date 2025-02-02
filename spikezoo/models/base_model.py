import torch
import torch.nn as nn
import importlib
import inspect
from dataclasses import dataclass, field
from spikezoo.utils import load_network, download_file
import os
import time
from typing import Dict
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import functools


# todo private design
@dataclass
class BaseModelConfig:
    # default params for BaseModel
    "Registerd model name."
    model_name: str = "base"
    "File name of the specified model."
    model_file_name: str = "nets"
    "Class name of the specified model in spikezoo/archs/base/{model_file_name}.py."
    model_cls_name: str = "BaseNet"
    "Spike input length for the specified model."
    model_win_length: int = 41
    "Model require model parameters or not."
    require_params: bool = False
    "Model stored path."
    ckpt_path: str = ""
    "Load pretrained weights or not."
    load_state: bool = True
    "Base url for storing pretrained models."
    base_url: str = "https://github.com/chenkang455/Spike-Zoo/releases/download/v0.1/"
    "Multi-GPU setting."
    multi_gpu: bool = False
    "Model parameters."
    model_params: dict = field(default_factory=lambda: {})


class BaseModel(nn.Module):
    def __init__(self, cfg: BaseModelConfig):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = self.build_network().to(self.device)
        self.net = nn.DataParallel(self.net) if cfg.multi_gpu == True else self.net
        self.model_half_win_length: int = cfg.model_win_length // 2

    # ! Might lead to low speed training on the BSF.
    def forward(self, spike):
        """A simple implementation for the spike-to-image conversion, given the spike input and output the reconstructed image."""
        spike = self.preprocess_spike(spike)
        img = self.net(spike)
        img = self.postprocess_img(img)
        return img

    def build_network(self):
        """Build the network and load the pretrained weight."""
        # network
        module = importlib.import_module(f"spikezoo.archs.{self.cfg.model_name}.{self.cfg.model_file_name}")
        model_cls = getattr(module, self.cfg.model_cls_name)
        model = model_cls(**self.cfg.model_params)
        if self.cfg.load_state and self.cfg.require_params:
            load_folder = os.path.dirname(os.path.abspath(__file__))
            weight_path = os.path.join(load_folder, self.cfg.ckpt_path)
            if os.path.exists(weight_path) == False:
                os.makedirs(os.path.dirname(weight_path), exist_ok=True)
                self.download_weight(weight_path)
                time.sleep(0.5)
            model = load_network(weight_path, model)
        return model

    def save_network(self, save_path):
        """Save the network."""
        network = self.net
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def download_weight(self, weight_path):
        """Download the pretrained weight from the given url."""
        url = self.cfg.base_url + os.path.basename(self.cfg.ckpt_path)
        download_file(url, weight_path)

    def crop_spike_length(self, spike):
        """Crop the spike length."""
        spike_length = spike.shape[1]
        spike_mid = spike_length // 2
        assert spike_length >= self.cfg.model_win_length, f"Spike input is not long enough, given {spike_length} frames < {self.cfg.model_win_length}."
        # even length
        if self.cfg.model_win_length == self.model_half_win_length * 2:
            spike = spike[
                :,
                spike_mid - self.model_half_win_length : spike_mid + self.model_half_win_length,
            ]
        # odd length
        else:
            spike = spike[
                :,
                spike_mid - self.model_half_win_length : spike_mid + self.model_half_win_length + 1,
            ]
        self.spike_size = (spike.shape[2], spike.shape[3])
        return spike

    def preprocess_spike(self, spike):
        """Preprocess the spike (length size)."""
        spike = self.crop_spike_length(spike)
        return spike

    def postprocess_img(self, image):
        """Postprocess the image."""
        return image

    # -------------------- Training Part --------------------
    def setup_training(self, pipeline_cfg):
        """Setup training optimizer and loss."""
        from spikezoo.pipeline import TrainPipelineConfig

        cfg: TrainPipelineConfig = pipeline_cfg
        self.optimizer = Adam(self.net.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs, eta_min=0)
        self.criterion = nn.L1Loss()

    def get_outputs_dict(self, batch):
        """Get the output dict for the given input batch. (Designed for the training mode considering possible auxiliary output.)"""
        # data process
        spike = batch["spike"]
        # outputs
        outputs = {}
        recon_img = self(spike)
        outputs["recon_img"] = recon_img
        return outputs

    def get_visual_dict(self, batch, outputs):
        """Get the visual dict from the given input batch and outputs."""
        visual_dict = {}
        visual_dict["recon"] = outputs["recon_img"]
        visual_dict["img"] = batch["img"]
        return visual_dict

    def get_loss_dict(self, outputs, batch):
        """Get the loss dict from the given input batch and outputs."""
        # data process
        gt_img = batch["img"]
        # recon image
        recon_img = outputs["recon_img"]
        # loss dict
        loss_dict = {}
        loss_dict["l1"] = self.criterion(recon_img, gt_img)
        loss_values_dict = {k: v.item() for k, v in loss_dict.items()}
        return loss_dict,loss_values_dict

    def get_paired_imgs(self, batch, outputs):
        recon_img = outputs["recon_img"]
        img = batch["img"]
        return recon_img, img

    def optimize_parameters(self, loss_dict):
        """Optimize the parameters from the loss_dict."""
        loss = functools.reduce(torch.add, loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        """Update the learning rate."""
        self.scheduler.step()

    def feed_to_device(self, batch):
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        return batch
