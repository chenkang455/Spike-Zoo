import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    dataset_cfg=REDS_BASEConfig(root_dir="spikezoo/data/REDS_BASE", use_aug=True, crop_size=(40, 40)),
    model_cfg=SSMLConfig(tfp_label_length=7),
)
pipeline.train()


# pipeline = TrainPipeline(
#     cfg=TrainPipelineConfig(
#         save_folder="results",
#         exp_name="ssml/no_nbsn",
#         bs_train=4,
#         epochs=100,
#         steps_per_save_imgs=10,
#         steps_per_save_ckpt=50,
#         steps_per_cal_metrics=10,
#         optimizer_cfg=AdamOptimizerConfig(lr=2e-4),
#         loss_weight_dict = {"l2": 1},
#         save_img_norm=False,
#         scheduler_cfg=None,
#     ),
#     dataset_cfg=REDS_BASEConfig(root_dir="spikezoo/data/REDS_BASE", use_aug=True, crop_size=(128, 128)),
#     model_cfg=SSMLConfig(tfp_label_length = 7),
# )
