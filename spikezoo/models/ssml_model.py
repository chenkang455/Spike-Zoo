from dataclasses import dataclass
from spikezoo.models.base_model import BaseModel, BaseModelConfig
import torch
from dataclasses import field
from spikezoo.archs.ssml.model import DoubleNet 


@dataclass
class SSMLConfig(BaseModelConfig):
    # default params for SSML
    model_name: str = "ssml"
    model_file_name: str = "model"
    model_cls_name: str = "DoubleNet"
    model_length: int = 41
    model_length_dict: dict = field(default_factory=lambda: {"v010": 41, "v023": 41})
    tfp_label_length: int = 11
    require_params: bool = True


# ! A simple version of SSML rather than the full version
class SSML(BaseModel):
    def __init__(self, cfg: BaseModelConfig):
        super(SSML, self).__init__(cfg)

    def get_outputs_dict(self, batch):
        # data process
        spike = batch["spike"]
        spike = self.preprocess_spike(spike)
        rate = batch["rate"].view(-1, 1, 1, 1).float()
        # outputs
        outputs = {}
        bsn_pred, nbsn_pred = self.net(spike)
        bsn_pred = self.postprocess_img(bsn_pred)
        nbsn_pred = self.postprocess_img(nbsn_pred)
        outputs["recon_img"] = nbsn_pred / rate
        outputs["bsn_pred"] = bsn_pred / rate
        # tfp-label
        mid = spike.shape[1] // 2
        tfp_label = torch.mean(spike[:, mid - self.cfg.tfp_label_length // 2 : mid + self.cfg.tfp_label_length // 2 + 1], dim=1, keepdim=True)
        outputs["tfp_label"] = self.postprocess_img(tfp_label) / rate
        return outputs

    def get_visual_dict(self, batch, outputs):
        visual_dict = super().get_visual_dict(batch, outputs)
        visual_dict["bsn_pred"] = outputs["bsn_pred"]
        visual_dict["tfp_label"] = outputs["tfp_label"]
        return visual_dict

    def get_loss_dict(self, outputs, batch, loss_weight_dict):
        # recon image
        recon_img = outputs["recon_img"]
        bsn_pred = outputs["bsn_pred"]
        tfp_label = outputs["tfp_label"]
        # loss dict
        loss_dict = {}
        for loss_name, weight in loss_weight_dict.items():
            loss_dict["bsn_loss_" + loss_name] = weight * self.get_loss_func(loss_name)(bsn_pred, tfp_label)
            loss_dict["mutual_loss_" + loss_name] = 0.01 * weight * self.get_loss_func(loss_name)(recon_img, bsn_pred)
        loss_values_dict = {k: v.item() for k, v in loss_dict.items()}
        return loss_dict, loss_values_dict
