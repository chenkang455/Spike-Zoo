from dataclasses import dataclass, field
from spikezoo.models.base_model import BaseModel, BaseModelConfig


@dataclass
class TFIConfig(BaseModelConfig):
    # default params for TFI
    model_name: str = "tfi"
    model_file_name: str = "nets"
    model_cls_name: str = "TFIModel"
    model_win_length: int = 41
    require_params: bool = False
    model_params: dict = field(default_factory=lambda: {"model_win_length": 41})


class TFI(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(TFI, self).__init__(cfg)
