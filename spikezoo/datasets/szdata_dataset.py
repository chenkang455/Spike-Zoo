from pathlib import Path
from spikezoo.datasets.base_dataset import BaseDataset, BaseDatasetConfig
from dataclasses import dataclass
import cv2
import torch
import numpy as np

# todo tobe evaluated
@dataclass
class SZDataConfig(BaseDatasetConfig):
    dataset_name: str = "szdata"
    root_dir: Path = Path(__file__).parent.parent / Path("data/szdata")
    width: int = 400
    height: int = 250
    with_img: bool = True
    spike_length_train: int = -1
    spike_length_test: int = -1
    spike_dir_name: str = "spike_data"
    img_dir_name: str = "sharp_data"
    rate: float = 1

class SZData(BaseDataset):
    def __init__(self, cfg: BaseDatasetConfig):
        super(SZData, self).__init__(cfg)
