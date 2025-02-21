from spikezoo.pipeline import TrainPipeline
from spikezoo.datasets.reds_base_dataset import REDS_BASEConfig
from spikezoo.models.ssir_model import SSIRConfig
from spikezoo.pipeline.train_cfgs import REDS_BASE_TrainConfig
from spikezoo.utils.optimizer_utils import AdamOptimizerConfig
from spikezoo.utils.scheduler_utils import CosineAnnealingLRConfig

# !: Can not achieve the desired performance
pipeline = TrainPipeline(
    cfg=REDS_BASE_TrainConfig(save_folder="results", exp_name="ssir", steps_grad_accumulation=8),
    dataset_cfg=REDS_BASEConfig(root_dir="spikezoo/data/reds_base", use_aug=True, crop_size=(128, 128)),
    model_cfg=SSIRConfig(),
)
pipeline.train()
