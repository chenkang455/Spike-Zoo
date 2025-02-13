from dataclasses import dataclass
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from dataclasses import field
from spikezoo.archs.ssir.models.networks import SSIR 


@dataclass
class SSIRConfig(BaseModelConfig):
    # default params for SSIR
    model_name: str = "ssir"
    model_file_name: str = "models.networks"
    model_cls_name: str = "SSIR"
    model_length: int = 41
    model_length_dict: dict = field(default_factory=lambda: {"v010": 41, "v023": 41})
    require_params: bool = True


class SSIR(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(SSIR, self).__init__(cfg)
