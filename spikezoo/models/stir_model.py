import torch
from dataclasses import dataclass, field
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from typing import List
from spikezoo.archs.stir.metrics.losses import compute_per_loss_single
from spikezoo.archs.stir.models.Vgg19 import Vgg19
from spikezoo.archs.stir.models.networks_STIR import STIR

@dataclass
class STIRConfig(BaseModelConfig):
    # default params for SSIR
    model_name: str = "stir"
    model_file_name: str = "models.networks_STIR"
    model_cls_name: str = "STIR"
    model_length: int = 61
    model_length_dict: dict = field(default_factory=lambda: {"v010": 61, "v023": 41})
    require_params: bool = True
    model_params: dict = field(default_factory=lambda: {})
    model_params_dict: dict = field(default_factory=lambda: {"v010": {"spike_dim": 61}, "v023": {"spike_dim": 41}})


class STIR(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(STIR, self).__init__(cfg)

    def preprocess_spike(self, spike):
        # length
        spike = self.crop_spike_length(spike)
        # size
        if self.spike_size == (250, 400):
            spike = torch.cat([spike, spike[:, :, -6:]], dim=2)
        elif self.spike_size == (480, 854):
            spike = torch.cat([spike, spike[:, :, :, -10:]], dim=3)
        return spike

    def postprocess_img(self, image):
        if self.spike_size == (250, 400):
            image = image[:, :, :250, :]
        elif self.spike_size == (480, 854):
            image = image[:, :, :, :854]
        return image

    def get_outputs_dict(self, batch):
        # data process
        spike = batch["spike"]
        rate = batch["rate"].view(-1, 1, 1, 1).float()
        # outputs
        outputs = {}
        spike = self.preprocess_spike(spike)
        # pyramid loss is omitted owing to limited performance gain.
        img_pred_0, Fs_lv_0, Fs_lv_1, Fs_lv_2, Fs_lv_3, Fs_lv_4, Est = self.net(spike)
        img_pred_0 = self.postprocess_img(img_pred_0)
        outputs["recon_img"] = img_pred_0 / rate
        return outputs
