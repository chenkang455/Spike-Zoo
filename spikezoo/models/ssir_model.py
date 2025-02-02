from dataclasses import dataclass
from spikezoo.models.base_model import BaseModel, BaseModelConfig


@dataclass
class SSIRConfig(BaseModelConfig):
    # default params for SSIR
    model_name: str = "ssir"
    model_file_name: str = "models.networks"
    model_cls_name: str = "SSIR"
    model_win_length: int = 41
    require_params: bool = True
    ckpt_path: str = "weights/ssir.pth"


class SSIR(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(SSIR, self).__init__(cfg)

    def postprocess_img(self, image):
        # image = image[0]
        return image
