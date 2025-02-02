from dataclasses import dataclass, field
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from typing import List


@dataclass
class WGSEConfig(BaseModelConfig):
    # default params for WGSE
    model_name: str = "wgse"
    model_file_name: str = "dwtnets"
    model_cls_name: str = "Dwt1dResnetX_TCN"
    model_win_length: int = 41
    require_params: bool = True
    ckpt_path: str = "weights/wgse.pt"
    model_params: dict = field(
        default_factory=lambda: {
            "wvlname": "db8",
            "J": 5,
            "yl_size": "15",
            "yh_size": [28, 21, 18, 16, 15],
            "num_residual_blocks": 3,
            "norm": None,
            "ks": 3,
            "store_features": True,
        }
    )


class WGSE(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(WGSE, self).__init__(cfg)
