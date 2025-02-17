from dataclasses import dataclass, field
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from typing import List
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from typing import List
from spikezoo.archs.wgse.dwtnets import Dwt1dResnetX_TCN


@dataclass
class WGSEConfig(BaseModelConfig):
    # default params for WGSE
    model_name: str = "wgse"
    model_file_name: str = "dwtnets"
    model_cls_name: str = "Dwt1dResnetX_TCN"
    model_length: int = 41
    model_length_dict: dict = field(default_factory=lambda: {"v010": 41, "v023": 41})
    require_params: bool = True


class WGSE(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(WGSE, self).__init__(cfg)
