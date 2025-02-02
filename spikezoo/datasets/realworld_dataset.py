from pathlib import Path
from spikezoo.datasets.base_dataset import BaseDataset, BaseDatasetConfig
from dataclasses import dataclass


@dataclass
class RealWorld_Config(BaseDatasetConfig):
    dataset_name: str = "realworld"
    root_dir: Path = Path(__file__).parent.parent / Path("data/recVidarReal2019")
    width: int = 400
    height: int = 250
    with_img: bool = False
    spike_length_train: int = -1
    spike_length_test: int = -1


class RealWorld(BaseDataset):
    def __init__(self, cfg: BaseDatasetConfig):
        super(RealWorld, self).__init__(cfg)

    def prepare_data(self):
        self.spike_dir = self.cfg.root_dir
        self.spike_list = self.get_spike_files(self.spike_dir)
        
    