import torch
from dataclasses import dataclass, field
from spikezoo.models.base_model import BaseModel, BaseModelConfig


@dataclass
class SpikeFormerConfig(BaseModelConfig):
    # default params for SpikeFormer
    model_name: str = "spikeformer"
    model_file_name: str = "Model.SpikeFormer"
    model_cls_name: str = "SpikeFormer"
    model_win_length: int = 65
    require_params: bool = True
    ckpt_path: str = "weights/spikeformer.pth"
    model_params: dict = field(
        default_factory=lambda: {
            "inputDim": 65,
            "dims": (32, 64, 160, 256),
            "heads": (1, 2, 5, 8),
            "ff_expansion": (8, 8, 4, 4),
            "reduction_ratio": (8, 4, 2, 1),
            "num_layers": 2,
            "decoder_dim": 256,
            "out_channel": 1,
        }
    )


class SpikeFormer(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(SpikeFormer, self).__init__(cfg)

    def preprocess_spike(self, spike):
        # length
        spike = self.crop_spike_length(spike)
        # size
        if self.spike_size == (250, 400):
            spike = torch.cat([spike[:, :, :3, :], spike, spike[:, :, -3:, :]], dim=2)
        elif self.spike_size == (480, 854):
            spike = torch.cat([spike, spike[:, :, :, -2:]], dim=3)
        # input
        spike = 2 * spike - 1
        return spike

    def postprocess_img(self, image):
        if self.spike_size == (250, 400):
            image = image[:, :, 3:-3, :]
        elif self.spike_size == (480, 854):
            image = image[:, :, :, :854]
        return image
