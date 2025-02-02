from dataclasses import dataclass
from spikezoo.models.base_model import BaseModel, BaseModelConfig


@dataclass
class SSMLConfig(BaseModelConfig):
    # default params for SSML
    model_name: str = "ssml"
    model_file_name: str = "model"
    model_cls_name: str = "DoubleNet"
    model_win_length: int = 41
    require_params: bool = True
    ckpt_path: str = 'weights/ssml.pt'


class SSML(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(SSML, self).__init__(cfg)
