from dataclasses import dataclass
from typing import Optional, Dict, Any
from src.config.base import BaseModelConfig, BaseDatasetConfig


@dataclass
class RecognitionModelConfig(BaseModelConfig):
    """Configuration for recognition models"""
    model_type: str = "dmer_net18"  # Default model type
    num_classes: int = 10
    T: int = 7  # Time steps for temporal data
    pretrained: bool = False
    checkpoint_path: Optional[str] = None
    model_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}


@dataclass
class RecognitionDatasetConfig(BaseDatasetConfig):
    """Configuration for recognition datasets"""
    dataset_type: str = "dmer"  # dmer, rps, vgg
    data_path: str = ""
    T: int = 7  # Time steps
    train_split: float = 0.8
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    transform_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.transform_config is None:
            self.transform_config = {}