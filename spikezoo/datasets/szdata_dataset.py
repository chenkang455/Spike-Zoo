from pathlib import Path
from spikezoo.datasets.base_dataset import BaseDataset, BaseDatasetConfig
from dataclasses import dataclass
import cv2
import torch
import numpy as np

@dataclass
class SZData_Config(BaseDatasetConfig):
    dataset_name: str = "szdata"
    root_dir: Path = Path(__file__).parent.parent / Path("data/dataset")
    width: int = 400
    height: int = 250
    with_img: bool = True
    spike_length_train: int = -1
    spike_length_test: int = -1
    spike_dir_name: str = "spike_data"
    img_dir_name: str = "sharp_data"

class SZData(BaseDataset):
    def __init__(self, cfg: BaseDatasetConfig):
        super(SZData, self).__init__(cfg)

    def get_img(self, idx):
        if self.cfg.with_img:
            spike_name = self.spike_list[idx]
            img_name = str(spike_name).replace(self.cfg.spike_dir_name,self.cfg.img_dir_name).replace(".dat",".png")
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img / 255).astype(np.float32)
            img = img[None]
            img = torch.from_numpy(img)
        else:
            spike = self.get_spike(idx)
            img = torch.mean(spike, dim=0, keepdim=True)
        return img
    