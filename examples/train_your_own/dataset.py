from spikezoo.datasets import BaseDataset,BaseDatasetConfig
from dataclasses import dataclass

@dataclass
class YourDatasetConfig(BaseDatasetConfig):
    dataset_name: str = "yourdataset"
    
class YourDataset(BaseDataset):
    def __init__(self, cfg: YourDatasetConfig):
        super(YourDataset, self).__init__(cfg)


