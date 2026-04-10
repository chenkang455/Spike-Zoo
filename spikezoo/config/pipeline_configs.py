from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Union
from spikezoo.utils.optimizer_utils import OptimizerConfig, AdamOptimizerConfig
from spikezoo.utils.scheduler_utils import SchedulerConfig
import yaml


@dataclass
class PipelineConfig:
    """Base pipeline configuration."""
    # Model loading
    version: Literal["local", "v010", "v023"] = "local"
    
    # Save settings
    save_folder: str = ""
    exp_name: str = ""
    
    # Evaluation settings
    save_metric: bool = True
    metric_names: List[str] = field(default_factory=lambda: ["psnr", "ssim", "niqe", "brisque"])
    
    # Image saving
    save_img: bool = True
    img_norm: bool = False
    
    # Dataloader settings (test)
    bs_test: int = 1
    nw_test: int = 0
    pin_memory: bool = False
    
    # Mode
    _mode: Literal["single_mode", "multi_mode", "train_mode"] = "single_mode"
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create PipelineConfig from dictionary."""
        # Filter out keys that are not in the dataclass fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        return cls(**filtered_dict)
    
    def to_dict(self):
        """Convert PipelineConfig to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class TrainPipelineConfig(PipelineConfig):
    """Training pipeline configuration."""
    # Training parameters
    epochs: int = 10
    steps_per_save_imgs: int = 10
    steps_per_save_ckpt: int = 10
    steps_per_cal_metrics: int = 10
    steps_grad_accumulation: int = 4
    
    # Training mode
    _mode: Literal["single_mode", "multi_mode", "train_mode"] = "train_mode"
    
    # Tensorboard
    use_tensorboard: bool = True
    
    # Random seed
    seed: int = 521
    
    # Dataloader settings (train)
    bs_train: int = 8
    nw_train: int = 4
    
    # Optimizer and scheduler
    optimizer_cfg: OptimizerConfig = field(default_factory=lambda: AdamOptimizerConfig(lr=1e-3))
    scheduler_cfg: Optional[SchedulerConfig] = None
    
    # Loss weights
    loss_weight_dict: Dict[Literal["l1", "l2"], float] = field(default_factory=lambda: {"l1": 1})
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create TrainPipelineConfig from dictionary."""
        # Handle optimizer config
        if "optimizer_cfg" in config_dict and isinstance(config_dict["optimizer_cfg"], dict):
            opt_cfg_dict = config_dict["optimizer_cfg"]
            opt_type = opt_cfg_dict.get("type", "adam")
            if opt_type.lower() == "adam":
                config_dict["optimizer_cfg"] = AdamOptimizerConfig(**{k: v for k, v in opt_cfg_dict.items() if k != "type"})
            else:
                # For other optimizer types, use default
                config_dict["optimizer_cfg"] = AdamOptimizerConfig(lr=1e-3)
        
        # Filter out keys that are not in the dataclass fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        return cls(**filtered_dict)


@dataclass
class EnsemblePipelineConfig(PipelineConfig):
    """Ensemble pipeline configuration."""
    _mode: Literal["single_mode", "multi_mode", "train_mode"] = "multi_mode"
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create EnsemblePipelineConfig from dictionary."""
        # Filter out keys that are not in the dataclass fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        return cls(**filtered_dict)


def load_pipeline_config(config_path: str, config_type: str = "base") -> Union[PipelineConfig, TrainPipelineConfig, EnsemblePipelineConfig]:
    """
    Load pipeline configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        config_type: Type of configuration ("base", "train", "ensemble")
        
    Returns:
        PipelineConfig instance
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f) or {}
    
    if config_type == "train":
        return TrainPipelineConfig.from_dict(config_dict)
    elif config_type == "ensemble":
        return EnsemblePipelineConfig.from_dict(config_dict)
    else:
        return PipelineConfig.from_dict(config_dict)