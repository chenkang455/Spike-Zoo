import torch.nn as nn
import torch.optim as optimizer
import torch.optim.lr_scheduler as lr_scheduler
import functools
from spikezoo.utils.optimizer_utils import OptimizerConfig, AdamOptimizerConfig
from spikezoo.utils.scheduler_utils import SchedulerConfig, MultiStepSchedulerConfig, CosineAnnealingLRConfig
from dataclasses import dataclass, field
from spikezoo.pipeline.train_pipeline import TrainPipelineConfig
from typing import Optional, Dict, List


@dataclass
class REDS_BASE_TrainConfig(TrainPipelineConfig):
    """Training setting for methods on the REDS-BASE dataset."""

    # parameters setting
    epochs: int = 600
    steps_per_save_imgs: int = 200
    steps_per_save_ckpt: int = 500
    steps_per_cal_metrics: int = 100
    metric_names: List[str] = field(default_factory=lambda: ["psnr", "ssim","lpips","niqe","brisque","piqe"])

    # dataloader setting
    bs_train: int = 8
    nw_train: int = 4
    pin_memory: bool = False

    # train setting - optimizer & scheduler & loss_dict
    optimizer_cfg: OptimizerConfig = AdamOptimizerConfig(lr=1e-4)
    scheduler_cfg: Optional[SchedulerConfig] = MultiStepSchedulerConfig(milestones=[400], gamma=0.2) # from wgse
    loss_weight_dict: Dict = field(default_factory=lambda: {"l1": 1})



# ! Train Config for each method on the official setting, not recommended to utilize their default parameters owing to the dataset setting.
@dataclass
class BSFTrainConfig(TrainPipelineConfig):
    """Training setting for BSF. https://github.com/ruizhao26/BSF"""

    optimizer_cfg: OptimizerConfig = AdamOptimizerConfig(lr=1e-4, weight_decay=0.0)
    scheduler_cfg: Optional[SchedulerConfig] = MultiStepSchedulerConfig(milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    loss_weight_dict: Dict = field(default_factory=lambda: {"l1": 1})


@dataclass
class WGSETrainConfig(TrainPipelineConfig):
    """Training setting for WGSE. https://github.com/Leozhangjiyuan/WGSE-SpikeCamera"""

    optimizer_cfg: OptimizerConfig = AdamOptimizerConfig(lr=1e-4, betas=(0.9, 0.99), weight_decay=0)
    scheduler_cfg: Optional[SchedulerConfig] = MultiStepSchedulerConfig(milestones=[400, 600], gamma=0.2)
    loss_weight_dict: Dict = field(default_factory=lambda: {"l1": 1})


@dataclass
class STIRTrainConfig(TrainPipelineConfig):
    """Training setting for STIR. https://github.com/GitCVfb/STIR"""

    optimizer_cfg: OptimizerConfig = AdamOptimizerConfig(lr=1e-4, betas=(0.9, 0.999))
    scheduler_cfg: Optional[SchedulerConfig] = MultiStepSchedulerConfig(milestones=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], gamma=0.7)
    loss_weight_dict: Dict = field(default_factory=lambda: {"l1": 1})


@dataclass
class Spk2ImgNetTrainConfig(TrainPipelineConfig):
    """Training setting for Spk2ImgNet. https://github.com/Vspacer/Spk2ImgNet"""

    optimizer_cfg: OptimizerConfig = AdamOptimizerConfig(lr=1e-4)
    scheduler_cfg: Optional[SchedulerConfig] = MultiStepSchedulerConfig(milestones=[20], gamma=0.1)
    loss_weight_dict: Dict = field(default_factory=lambda: {"l1": 1})
