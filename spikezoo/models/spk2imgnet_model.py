import torch
from dataclasses import dataclass, field
from spikezoo.models.base_model import BaseModel, BaseModelConfig


@dataclass
class Spk2ImgNetConfig(BaseModelConfig):
    # default params for Spk2ImgNet
    model_name: str = "spk2imgnet"
    model_file_name: str = "nets"
    model_cls_name: str = "SpikeNet"
    model_win_length: int = 41
    require_params: bool = True
    ckpt_path: str = "weights/spk2imgnet.pth"
    light_correction: bool = False

    # model params
    model_params: dict = field(
        default_factory=lambda: {
            "in_channels": 13,
            "features": 64,
            "out_channels": 1,
            "win_r": 6,
            "win_step": 7,
        }
    )


class Spk2ImgNet(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(Spk2ImgNet, self).__init__(cfg)

    def preprocess_spike(self, spike):
        # length
        spike = self.crop_spike_length(spike)
        # size
        if self.spike_size == (250, 400):
            spike = torch.cat([spike, spike[:, :, -2:]], dim=2)
        elif self.spike_size == (480, 854):
            spike = torch.cat([spike, spike[:, :, :, -2:]], dim=3)
        return spike

    def postprocess_img(self, image):
        if self.spike_size == (250, 400):
            image = image[:, :, :250, :]
        elif self.spike_size == (480, 854):
            image = image[:, :, :, :854]
        # used on the REDS_small dataset.
        if self.cfg.light_correction == True:
            image = torch.clamp(image / 0.6, 0, 1)
        return image
