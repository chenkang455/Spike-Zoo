from spikezoo.pipeline import TrainPipeline
from spikezoo.datasets.reds_base_dataset import REDS_BASEConfig
from spikezoo.models.stir_model import STIRConfig
from spikezoo.pipeline.train_cfgs import REDS_BASE_TrainConfig
from spikezoo.utils.optimizer_utils import AdamOptimizerConfig

# !: Can not achieve the desired performance
pipeline = TrainPipeline(
    cfg=REDS_BASE_TrainConfig(save_folder="results", exp_name="stir"),
    dataset_cfg=REDS_BASEConfig(root_dir="spikezoo/data/reds_base", use_aug=True, crop_size=(128, 128)),
    model_cfg=STIRConfig(model_win_length=41, model_params={"spike_dim": 41}),
)
pipeline.train()
