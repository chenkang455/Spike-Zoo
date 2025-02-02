import torch
from dataclasses import dataclass
from spikezoo.models.base_model import BaseModel, BaseModelConfig


@dataclass
class STIRConfig(BaseModelConfig):
    # default params for SSIR
    model_name: str = "stir"
    model_file_name: str = "models.networks_STIR"
    model_cls_name: str = "STIR"
    model_win_length: int = 61
    require_params: bool = True
    ckpt_path: str = "weights/stir.pth"


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
        # recon, Fs_lv_0, Fs_lv_1, Fs_lv_2, Fs_lv_3, Fs_lv_4, Est = image
        if self.spike_size == (250, 400):
            image = image[:, :, :250, :]
        elif self.spike_size == (480, 854):
            image = image[:, :, :, :854]
        return image
