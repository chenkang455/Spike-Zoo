from spikezoo.pipeline import TrainPipeline
from spikezoo.datasets.reds_base_dataset import REDS_BASEConfig
from spikezoo.models.ssml_model import SSMLConfig
from spikezoo.pipeline.train_cfgs import TrainPipelineConfig
from spikezoo.utils.optimizer_utils import AdamOptimizerConfig
from spikezoo.utils.scheduler_utils import CosineAnnealingLRConfig

# !: Can not achieve the desired performance
pipeline = TrainPipeline(
    cfg=TrainPipelineConfig(
        save_folder="results",
        exp_name="ssml",
        bs_train=4,
        epochs=400,
        steps_per_save_imgs=100,
        steps_per_save_ckpt=400,
        steps_per_cal_metrics=20,
        optimizer_cfg=AdamOptimizerConfig(lr=3e-4),
        scheduler_cfg=CosineAnnealingLRConfig(T_max=400),  # epochs
        loss_weight_dict={"l2": 1},
        save_img_norm=False, 
    ),
    dataset_cfg=REDS_BASEConfig(root_dir="spikezoo/data/reds_base", use_aug=True, crop_size=(40, 40)),
    model_cfg=SSMLConfig(tfp_label_length=7),
)
pipeline.train()
