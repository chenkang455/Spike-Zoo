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
    model_name: str = "yourmodel"  # 需与文件名保持一致
    model_file_name: str = "arch.net"  # archs路径下的模块路径
    model_cls_name: str = "YourNet"  # 模型类名
    model_length: int = 41
    require_params: bool = True
    model_params: dict = field(default_factory=lambda: {"inDim": 41})

class YourModel(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(YourModel, self).__init__(cfg)