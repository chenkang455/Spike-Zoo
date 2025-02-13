from dataclasses import dataclass, field
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from spikezoo.archs.tfp.nets import TFPModel


@dataclass
class TFPConfig(BaseModelConfig):
    # default params for TFP
    model_name: str = "tfp"
    model_file_name: str = "nets"
    model_cls_name: str = "TFPModel"
    model_length: int = 41
    model_length_dict: dict = field(default_factory=lambda: {"v010": 41, "v023": 41})
    require_params: bool = False
    model_params: dict = field(default_factory=lambda: {"model_win_length": 41})


class TFP(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(TFP, self).__init__(cfg)
