from torch.utils.data import Dataset
from pathlib import Path
from spikezoo.datasets.base_dataset import BaseDataset, BaseDatasetConfig
from dataclasses import dataclass
import re


@dataclass
class REDS_BASEConfig(BaseDatasetConfig):
    dataset_name: str = "reds_base"
    root_dir: Path = Path(__file__).parent.parent / Path("data/reds_base")
    width: int = 400
    height: int = 250
    with_img: bool = True
    spike_length_train: int = 41
    spike_length_test: int = 301
    spike_dir_name: str = "spike"
    img_dir_name: str = "gt"
    rate: float = 0.6


class REDS_BASE(BaseDataset):
    def __init__(self, cfg: BaseDatasetConfig):
        super(REDS_BASE, self).__init__(cfg)

    def prepare_data(self):
        super().prepare_data()
        self.img_list = [self.img_dir / Path(str(s.name).replace(".dat", ".png")) for s in self.spike_list]
