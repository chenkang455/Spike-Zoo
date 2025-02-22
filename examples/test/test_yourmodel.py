from torch.utils.data import Dataset
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Union
from typing import Optional
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from dataclasses import field
import torch.nn as nn


@dataclass
class YourModelConfig(BaseModelConfig):
    model_name: str = "yourmodel"
    model_length: int = 41
    require_params: bool = True
    model_params: dict = field(default_factory=lambda: {"inDim": 41})


class YourModel(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(YourModel, self).__init__(cfg)


def conv_layer(inDim, outDim, ks, s, p, norm_layer="none"):
    ## convolutional layer
    conv = nn.Conv2d(inDim, outDim, kernel_size=ks, stride=s, padding=p)
    relu = nn.ReLU(True)
    assert norm_layer in ("batch", "instance", "none")
    if norm_layer == "none":
        seq = nn.Sequential(*[conv, relu])
    else:
        if norm_layer == "instance":
            norm = nn.InstanceNorm2d(outDim, affine=False, track_running_stats=False)  # instance norm
        else:
            momentum = 0.1
            norm = nn.BatchNorm2d(outDim, momentum=momentum, affine=True, track_running_stats=True)
        seq = nn.Sequential(*[conv, norm, relu])
    return seq


class YourNet(nn.Module):
    """Borrow the structure from the SpikeCLIP. (https://arxiv.org/abs/2501.04477)"""

    def __init__(self, inDim=41):
        super(YourNet, self).__init__()
        norm = "none"
        outDim = 1
        convBlock1 = conv_layer(inDim, 64, 3, 1, 1)
        convBlock2 = conv_layer(64, 128, 3, 1, 1, norm)
        convBlock3 = conv_layer(128, 64, 3, 1, 1, norm)
        convBlock4 = conv_layer(64, 16, 3, 1, 1, norm)
        conv = nn.Conv2d(16, outDim, 3, 1, 1)
        self.seq = nn.Sequential(*[convBlock1, convBlock2, convBlock3, convBlock4, conv])

    def forward(self, x):
        return self.seq(x)


net = YourModel(
    YourModelConfig(
        model_cls_local=YourModel,
        arch_cls_local=YourNet,
        load_state=True,
        ckpt_path="spikezoo/models/weights/v023/base.pth",
    )
)
net.build_network(mode="eval")
