import torch
import torch.nn as nn
import importlib
import inspect
from dataclasses import dataclass, field
from spikezoo.utils import load_network, download_file
import os
import time
from typing import Dict, Literal
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import functools
import torch.nn as nn
from typing import Optional, Union, List
from spikezoo.archs.base.nets import BaseNet 


# todo private design
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

class BaseModel(nn.Module):
    def __init__(self, cfg: BaseModelConfig):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_func_cache = {}

    def forward(self, spike):
        return self.spk2img(spike)

    # ! Might lead to low speed training on the BSF.
    def spk2img(self, spike):
        """A simple implementation for the spike-to-image conversion (**tailored for the evaluation mode**), given the spike input and output the reconstructed image."""
        spike = self.preprocess_spike(spike)
        img = self.net(spike)
        img = self.postprocess_img(img)
        return img

    def build_network(
        self,
        mode: Literal["debug", "train", "eval"] = "debug",
        version: Literal["local", "v010", "v023"] = "local",
    ):
        """Build the network and load the pretrained weight."""
        # network
        if self.cfg.arch_cls_local == None:
            module = importlib.import_module(f"spikezoo.archs.{self.cfg.model_name}.{self.cfg.model_file_name}")
            model_cls = getattr(module, self.cfg.model_cls_name)
        else:
            model_cls = self.cfg.arch_cls_local
        # load model config parameters
        if version == "local":
            model = model_cls(**self.cfg.model_params)
            self.model_length = self.cfg.model_length
            self.model_half_length = self.model_length // 2
        else:
            model = model_cls(**self.cfg.model_params_dict[version])
            self.model_length = self.cfg.model_length_dict[version]
            self.model_half_length = self.model_length // 2
        model.train() if mode == "train" else model.eval()
        # auto set the load_state to True under the eval mode
        if mode == "eval" and self.cfg.load_state == False:
            print(f"Method {self.cfg.model_name} on the evaluation mode, load_state is set to True automatically.")
            self.cfg.load_state = True
        # load model
        if self.cfg.load_state and self.cfg.require_params:
            # load from the url version
            if version != "local":
                load_folder = os.path.dirname(os.path.abspath(__file__))
                ckpt_name = f"{self.cfg.model_name}.{get_suffix(self.cfg.model_name,version)}"
                ckpt_path = os.path.join("weights",version,ckpt_name)
                ckpt_path = os.path.join(load_folder, ckpt_path)
                ckpt_path_url = os.path.join(self.cfg.base_url,get_url_version(version),ckpt_name)
            elif version == "local":
                ckpt_path = self.cfg.ckpt_path

            # no ckpt found on the device, try to download from the url
            if os.path.isfile(ckpt_path) == False and version != "local":
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                download_file(ckpt_path_url, ckpt_path)
                time.sleep(0.5)
            elif os.path.isfile(ckpt_path) == False and version == "local":
                raise RuntimeError(
                    f"For the method {self.cfg.model_name}, no ckpt can be found on the {ckpt_path} !!! Try set the version to get the model from the url."
                )
            model = load_network(ckpt_path, model)
        # to device
        model = model.to(self.device)
        model = nn.DataParallel(model) if self.cfg.multi_gpu == True else model
        self.net = model
        return self

    def save_network(self, save_path):
        """Save the network."""
        network = self.net
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def crop_spike_length(self, spike):
        """Crop the spike length."""
        spike_length = spike.shape[1]
        spike_mid = spike_length // 2
        assert spike_length >= self.model_length, f"Spike input is not long enough, given {spike_length} frames < {self.cfg.model_length} required by the {self.cfg.model_name}."
        # even length
        if self.model_length == self.model_half_length * 2:
            spike = spike[
                :,
                spike_mid - self.model_half_length : spike_mid + self.model_half_length,
            ]
        # odd length
        else:
            spike = spike[
                :,
                spike_mid - self.model_half_length : spike_mid + self.model_half_length + 1,
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
    def get_outputs_dict(self, batch):
        """Get the output dict for the given input batch. (Designed for the training mode considering possible auxiliary output.)"""
        # data process
        spike = batch["spike"]
        rate = batch["rate"].view(-1, 1, 1, 1).float()
        # outputs
        outputs = {}
        recon_img = self.spk2img(spike)
        outputs["recon_img"] = recon_img / rate
        return outputs

    def get_visual_dict(self, batch, outputs):
        """Get the visual dict from the given input batch and outputs."""
        visual_dict = {}
        visual_dict["recon_img"] = outputs["recon_img"]
        visual_dict["gt_img"] = batch["gt_img"]
        return visual_dict

    def get_loss_dict(self, outputs, batch, loss_weight_dict):
        """Get the loss dict from the given input batch and outputs."""
        # data process
        gt_img = batch["gt_img"]
        # recon image
        recon_img = outputs["recon_img"]
        # loss dict
        loss_dict = {}
        for loss_name, weight in loss_weight_dict.items():
            loss_dict[loss_name] = weight * self.get_loss_func(loss_name)(recon_img, gt_img)

        # todo add your desired loss here by loss_dict["name"] = loss()

        loss_values_dict = {k: v.item() for k, v in loss_dict.items()}
        return loss_dict, loss_values_dict

    def get_loss_func(self, name: Literal["l1", "l2"]):
        """Get the loss function from the given loss name."""
        if name not in self.loss_func_cache:
            if name == "l1":
                self.loss_func_cache[name] = nn.L1Loss()
            elif name == "l2":
                self.loss_func_cache[name] = nn.MSELoss()
            else:
                self.loss_func_cache[name] = lambda x, y: 0
        loss_func = self.loss_func_cache[name]
        return loss_func

    def get_paired_imgs(self, batch, outputs):
        """Get paired images for the metric calculation."""
        recon_img = outputs["recon_img"]
        img = batch["gt_img"]
        return recon_img, img

    def feed_to_device(self, batch):
        """Feed the batch data to the given device."""
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        return batch


# functions
def get_suffix(model_name, version):
    if version == "v010":
        if model_name in ["ssml", "wgse"]:
            return "pt"
        else:
            return "pth"
    else:
        return "pth"


def get_url_version(version):
    major = version[1]
    minor = version[2]
    patch = version[3]
    return f"v{major}.{minor}.{patch}"
