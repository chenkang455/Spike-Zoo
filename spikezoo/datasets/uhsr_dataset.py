from pathlib import Path
from spikezoo.datasets.base_dataset import BaseDataset, BaseDatasetConfig
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class UHSRConfig(BaseDatasetConfig):
    dataset_name: str = "uhsr"
    root_dir: Path = Path(__file__).parent.parent / Path("data/u_caltech")
    width: int = 224
    height: int = 224
    with_img: bool = False
    spike_length_train: int = 200
    spike_length_test: int = 200
    spike_dir_name: str = "spike"
    img_dir_name: str = ""
    rate: float = 1

class UHSR(BaseDataset):
    def __init__(self, cfg: BaseDatasetConfig):
        super(UHSR, self).__init__(cfg)

    def prepare_data(self):
        self.spike_dir = self.cfg.root_dir / self.cfg.dataset_split
        self.spike_list = self.get_spike_files(self.spike_dir)

    def get_spike_files(self, path: Path):
        files = path.glob("**/*.npz")
        return sorted(files)

    def load_spike(self, idx):
        spike_name = str(self.spike_list[idx])
        data = np.load(spike_name)
        spike = data["spk"].astype(np.float32)
        spike = torch.from_numpy(spike)
        spike = spike[:, 13:237, 13:237]
        return spike
