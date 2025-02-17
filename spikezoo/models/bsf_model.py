import torch
from dataclasses import dataclass, field
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from typing import List
from spikezoo.archs.bsf.models.bsf.bsf import BSF 


@dataclass
class BSFConfig(BaseModelConfig):
    # default params for BSF
    model_name: str = "bsf"
    model_file_name: str = "models.bsf.bsf"
    model_cls_name: str = "BSF"
    model_length: int = 61
    model_length_dict: dict = field(default_factory=lambda: {"v010": 61, "v023": 41})
    require_params: bool = True
    model_params: dict = field(default_factory=lambda: {})
    model_params_dict: dict = field(default_factory=lambda: {"v010": {"spike_dim": 61}, "v023": {"spike_dim": 41}})


class BSF(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(BSF, self).__init__(cfg)

    def preprocess_spike(self, spike):
        # length
        spike = self.crop_spike_length(spike)
        # size
        if self.spike_size == (250, 400):
            spike = torch.cat([spike, spike[:, :, -2:]], dim=2)
        elif self.spike_size == (480, 854):
            spike = torch.cat([spike, spike[:, :, :, -2:]], dim=3)
        # dsft
        dsft = self.compute_dsft_core(spike)
        dsft_dict = self.convert_dsft4(dsft, spike)
        input_dict = {
            "dsft_dict": dsft_dict,
            "spikes": spike,
        }
        return input_dict

    def postprocess_img(self, image):
        if self.spike_size == (250, 400):
            image = image[:, :, :250, :]
        elif self.spike_size == (480, 854):
            image = image[:, :, :, :854]
        return image

    def compute_dsft_core(self, spike):
        bs, T, H, W = spike.shape
        time = spike * torch.arange(T, device="cuda").reshape(1, T, 1, 1)
        l_idx, _ = time.cummax(dim=1)
        time[time == 0] = T
        r_idx, _ = torch.flip(time, [1]).cummin(dim=1)
        r_idx = torch.flip(r_idx, [1])
        r_idx = torch.cat([r_idx[:, 1:, :, :], torch.ones([bs, 1, H, W], device="cuda") * T], dim=1)
        res = r_idx - l_idx
        res = torch.clip(res, 0)
        return res

    def convert_dsft4(self, dsft, spike):
        b, T, h, w = spike.shape
        dmls1 = -1 * torch.ones(spike.shape, device=spike.device, dtype=torch.float32)
        dmrs1 = -1 * torch.ones(spike.shape, device=spike.device, dtype=torch.float32)
        flag = -2 * torch.ones([b, h, w], device=spike.device, dtype=torch.float32)
        for ii in range(T - 1, 0 - 1, -1):
            flag += spike[:, ii] == 1
            copy_pad_coord = flag < 0
            dmls1[:, ii][copy_pad_coord] = dsft[:, ii][copy_pad_coord]
            if ii < T - 1:
                update_coord = (spike[:, ii + 1] == 1) * (~copy_pad_coord)
                dmls1[:, ii][update_coord] = dsft[:, ii + 1][update_coord]
                non_update_coord = (spike[:, ii + 1] != 1) * (~copy_pad_coord)
                dmls1[:, ii][non_update_coord] = dmls1[:, ii + 1][non_update_coord]
        flag = -2 * torch.ones([b, h, w], device=spike.device, dtype=torch.float32)
        for ii in range(0, T, 1):
            flag += spike[:, ii] == 1
            copy_pad_coord = flag < 0
            dmrs1[:, ii][copy_pad_coord] = dsft[:, ii][copy_pad_coord]
            if ii > 0:
                update_coord = (spike[:, ii] == 1) * (~copy_pad_coord)
                dmrs1[:, ii][update_coord] = dsft[:, ii - 1][update_coord]
                non_update_coord = (spike[:, ii] != 1) * (~copy_pad_coord)
                dmrs1[:, ii][non_update_coord] = dmrs1[:, ii - 1][non_update_coord]
        dsft12 = dsft + dmls1
        dsft21 = dsft + dmrs1
        dsft22 = dsft + dmls1 + dmrs1
        dsft_dict = {
            "dsft11": dsft,
            "dsft12": dsft12,
            "dsft21": dsft21,
            "dsft22": dsft22,
        }
        return dsft_dict
