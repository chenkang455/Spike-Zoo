from spikezoo.pipeline import TrainPipeline
from spikezoo.datasets.reds_base_dataset import REDS_BASEConfig
from spikezoo.models.wgse_model import WGSEConfig
from spikezoo.pipeline.train_cfgs import REDS_BASE_TrainConfig

pipeline = TrainPipeline(
    cfg=REDS_BASE_TrainConfig(save_folder="results", exp_name="wgse"),
    dataset_cfg=REDS_BASEConfig(root_dir="spikezoo/data/reds_base", use_aug=True, crop_size=(128, 128)),
    model_cfg=WGSEConfig(),
)
pipeline.train()
