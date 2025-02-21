from spikezoo.models import BaseModel,BaseModelConfig
from dataclasses import dataclass
import torch.nn as nn
from typing import Optional

def get_model_cls():
    return YourModel

@dataclass
class YourModelConfig(BaseModelConfig):
    model_name: str = "yourmodel"

class YourModel(BaseModel):
    def __init__(self, cfg: YourModelConfig):
        super(YourModel, self).__init__(cfg)
    



