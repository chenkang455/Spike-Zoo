from torch.utils.data import Dataset
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Union
from typing import Optional
from spikezoo.datasets.base_dataset import BaseDatasetConfig,BaseDataset

@dataclass
class YourDatasetConfig(BaseDatasetConfig):
    dataset_name: str = "yourdataset"
    root_dir: Union[str, Path] = Path(__file__).parent.parent / Path("data/your_data_path")
    width: int = 400
    height: int = 250
    with_img: bool = True
    spike_length_train: int = -1
    spike_length_test: int = -1
    spike_dir_name: str = "spike_data"
    img_dir_name: str = "sharp_data"
    rate: float = 1

class YourDataset(BaseDataset):
    def __init__(self, cfg: BaseDatasetConfig):
        super(YourDataset, self).__init__(cfg)