from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from pathlib import Path


@dataclass
class BaseModelConfig:
    """Base configuration for models"""
    model_name: str = "base"
    model_file_name: str = "base_model"
    model_cls_name: str = "BaseModel"
    

@dataclass
class BaseDatasetConfig:
    """Base configuration for datasets"""
    dataset_name: str = "base"
    root_dir: Union[str, Path] = Path(__file__).parent.parent.parent / Path("data/base")
    width: int = 400
    height: int = 250
    with_img: bool = True
    spike_length_train: int = -1
    spike_length_test: int = -1
    spike_dir_name: str = "spike"
    img_dir_name: str = "gt"
    rate: float = 0.6
    use_aug: bool = False
    use_cache: bool = False
    crop_size: tuple = (-1, -1)