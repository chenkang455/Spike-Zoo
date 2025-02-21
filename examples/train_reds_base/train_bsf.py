from spikezoo.pipeline import TrainPipeline
from spikezoo.datasets.reds_base_dataset import REDS_BASEConfig
from spikezoo.models.bsf_model import BSFConfig
from spikezoo.pipeline.train_cfgs import REDS_BASE_TrainConfig

pipeline = TrainPipeline(
    cfg=REDS_BASE_TrainConfig(save_folder="results", exp_name="bsf"),
    dataset_cfg=REDS_BASEConfig(root_dir="spikezoo/data/reds_base", use_aug=True, crop_size=(128, 128)),
    model_cfg=BSFConfig(model_length=41, model_params={"spike_dim": 41}),
)
pipeline.train()
