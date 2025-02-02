from torch.utils.data import Dataset
from pathlib import Path
from spikezoo.datasets.base_dataset import BaseDataset, BaseDatasetConfig
from dataclasses import dataclass
import re

@dataclass
class REDS_Small_Config(BaseDatasetConfig):
    dataset_name: str = "reds_small"
    root_dir: Path = Path(__file__).parent.parent / Path("data/REDS_Small")
    width: int = 400
    height: int = 250
    with_img: bool = True
    spike_length_train: int = 41
    spike_length_test: int = 301
    spike_dir_name: str = "spike"
    img_dir_name: str = "gt"

class REDS_Small(BaseDataset):
    def __init__(self, cfg: BaseDatasetConfig):
        super(REDS_Small, self).__init__(cfg)

    def prepare_data(self):
        super().prepare_data()
        if self.cfg.split == "train":
            self.img_list = [self.img_dir / Path(str(s.name).replace('.dat','.png')) for s in self.spike_list]

