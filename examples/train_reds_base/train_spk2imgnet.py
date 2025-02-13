import os, sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from spikezoo.pipeline import TrainPipeline
from spikezoo.datasets.reds_base_dataset import REDS_BASEConfig
from spikezoo.models.spk2imgnet_model import Spk2ImgNetConfig
from spikezoo.pipeline.train_cfgs import REDS_BASE_TrainConfig

pipeline = TrainPipeline(
    cfg=REDS_BASE_TrainConfig(save_folder="results", exp_name="spk2imgnet"),
    dataset_cfg=REDS_BASEConfig(root_dir="spikezoo/data/REDS_BASE", use_aug=True, crop_size=(128, 128)),
    model_cfg=Spk2ImgNetConfig(),
)
pipeline.train()
