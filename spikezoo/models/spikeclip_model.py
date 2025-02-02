from dataclasses import dataclass
from spikezoo.models.base_model import BaseModel, BaseModelConfig
import torch
import torch.nn.functional as F


@dataclass
class SpikeCLIPConfig(BaseModelConfig):
    # default params for SpikeCLIP
    model_name: str = "spikeclip"
    model_file_name: str = "nets"
    model_cls_name: str = "LRN"
    model_win_length: int = 200
    require_params: bool = True
    ckpt_path: str = "weights/spikeclip.pth"


class SpikeCLIP(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(SpikeCLIP, self).__init__(cfg)

    def preprocess_spike(self, spike):
        # length
        spike = self.crop_spike_length(spike)
        # voxel
        voxel = torch.sum(spike.reshape(-1, 50, 4, spike.shape[-2], spike.shape[-1]), axis=2)  # [200,224,224] -> [50,224,224]
        voxel = F.pad(voxel, pad=(20, 20, 20, 20), mode="reflect", value=0)
        return voxel

    def postprocess_img(self, image):
        image = image[:, :, 20:-20, 20:-20]
        return image
