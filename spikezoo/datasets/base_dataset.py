import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
from spikezoo.utils.spike_utils import load_vidar_dat
import re
from dataclasses import dataclass, replace
from typing import Literal, Union
import warnings
import torch
from tqdm import tqdm
from spikezoo.utils.data_utils import Augmentor
from typing import Optional


@dataclass
class BaseDatasetConfig:
    # ------------- Not Recommended to Change -------------
    "Dataset name."
    dataset_name: str = "base"
    "Directory specifying location of data."
    root_dir: Union[str, Path] = Path(__file__).parent.parent / Path("data/base")
    "Image width."
    width: int = 400
    "Image height."
    height: int = 250
    "Spike paried with the image or not."
    with_img: bool = True
    "Dataset spike length for the train data."
    spike_length_train: int = -1
    "Dataset spike length for the test data."
    spike_length_test: int = -1
    "Dir name for the spike."
    spike_dir_name: str = "spike"
    "Dir name for the image."
    img_dir_name: str = "gt"
    "Rate. (-1 denotes variant)"
    rate: float = 0.6

    # ------------- Config -------------
    "Use the data augumentation technique or not."
    use_aug: bool = False
    "Use cache mechanism."
    use_cache: bool = False
    "Crop size."
    crop_size: tuple = (-1, -1)
    "Load the dataset from local or spikezoo lib."
    dataset_cls_local: Optional[Dataset] = None
    "Spike load version. [python,cpp]"
    spike_load_version: Literal["python", "cpp"] = "python"


# todo cache mechanism
class BaseDataset(Dataset):
    def __init__(self, cfg: BaseDatasetConfig):
        super(BaseDataset, self).__init__()
        self.cfg = cfg

    def __len__(self):
        return len(self.spike_list)

    def __getitem__(self, idx: int):
        # load data
        if self.cfg.use_cache == True:
            spike, img = self.cache_spkimg[idx]
            spike = spike.to(torch.float32)
        else:
            spike = self.get_spike(idx)
            img = self.get_img(idx)

        # process data
        if self.cfg.use_aug == True and self.split == "train":
            spike, img = self.augmentor(spike, img)

        # rate
        rate = self.cfg.rate

        # ! spike and gt_img names are necessary
        batch = {"spike": spike, "gt_img": img, "rate": rate}
        return batch

    def build_source(self, split: Literal["train", "test"] = "test"):
        """Build the dataset source and prepare to be loaded files."""
        # spike length
        self.split = split
        self.spike_length = self.cfg.spike_length_train if self.split == "train" else self.cfg.spike_length_test
        # root dir
        self.cfg.root_dir = Path(self.cfg.root_dir) if isinstance(self.cfg.root_dir, str) else self.cfg.root_dir
        assert self.cfg.root_dir.exists(), f"No files found in {self.cfg.root_dir} for the specified dataset `{self.cfg.dataset_name}`."
        # prepare
        self.augmentor = Augmentor(self.cfg.crop_size) if self.cfg.use_aug == True and self.split == "train" else -1
        self.prepare_data()
        self.cache_data() if self.cfg.use_cache == True else -1
        warnings.warn("Lengths of the image list and the spike list should be equal.") if len(self.img_list) != len(self.spike_list) else -1

    # todo: To be overridden
    def prepare_data(self):
        """Specify the spike and image files to be loaded."""
        # spike
        self.spike_dir = self.cfg.root_dir / self.split / self.cfg.spike_dir_name
        self.spike_list = self.get_spike_files(self.spike_dir)
        # gt
        if self.cfg.with_img == True:
            self.img_dir = self.cfg.root_dir / self.split / self.cfg.img_dir_name
            self.img_list = self.get_image_files(self.img_dir)

    # todo: To be overridden
    def get_spike_files(self, path: Path):
        """Recognize spike files automatically (default .dat)."""
        files = path.glob("**/*.dat")
        return sorted(files)

    # todo: To be overridden
    def load_spike(self, idx):
        """Load the spike stream from the given idx."""
        spike_name = str(self.spike_list[idx])
        spike = load_vidar_dat(
            spike_name,
            height=self.cfg.height,
            width=self.cfg.width,
            out_format="tensor",
            version=self.cfg.spike_load_version
        )
        return spike

    def get_spike(self, idx):
        """Get and process the spike stream from the given idx."""
        spike_length = self.spike_length
        spike = self.load_spike(idx)
        assert spike.shape[0] >= spike_length, f"Given spike length {spike.shape[0]} smaller than the required length {spike_length}"
        spike_mid = spike.shape[0] // 2
        # spike length process
        if spike_length == -1:
            spike = spike
        elif spike_length % 2 == 1:
            spike = spike[spike_mid - spike_length // 2 : spike_mid + spike_length // 2 + 1]
        elif spike_length % 2 == 0:
            spike = spike[spike_mid - spike_length // 2 : spike_mid + spike_length // 2]
        return spike

    def get_image_files(self, path: Path):
        """Recognize image files automatically."""
        files = [f for f in path.glob("**/*") if re.match(r".*\.(jpg|jpeg|png)$", f.name, re.IGNORECASE)]
        return sorted(files)

    # todo: To be overridden
    def get_img(self, idx):
        """Get the image from the given idx."""
        if self.cfg.with_img:
            img_name = str(self.img_list[idx])
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img / 255).astype(np.float32)
            img = img[None]
            img = torch.from_numpy(img)
        else:
            spike = self.get_spike(idx)
            img = torch.mean(spike, dim=0, keepdim=True)
        return img

    def cache_data(self):
        """Cache the data."""
        self.cache_spkimg = []
        for idx in tqdm(range(len(self.spike_list)), desc="Caching data", unit="sample"):
            spike = self.get_spike(idx).to(torch.uint8)
            img = self.get_img(idx)
            self.cache_spkimg.append([spike, img])
