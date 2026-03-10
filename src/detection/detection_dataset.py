"""
Detection dataset for Spike-Zoo.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
from spikezoo.datasets.base_dataset import BaseDataset, BaseDatasetConfig
from dataclasses import dataclass, replace
from typing import Literal, Union, Optional
import warnings
import torch
from tqdm import tqdm
from spikezoo.utils.data_utils import Augmentor
import re


@dataclass
class DetectionDatasetConfig(BaseDatasetConfig):
    """Configuration for detection dataset."""
    # Dataset name
    dataset_name: str = "detection"
    # Directory specifying location of data
    root_dir: Union[str, Path] = Path(__file__).parent.parent.parent / Path("data/detection")
    # Image width
    width: int = 400
    # Image height
    height: int = 250
    # Spike paired with the image or not
    with_img: bool = True
    # Dataset spike length for the train data
    spike_length_train: int = -1
    # Dataset spike length for the test data
    spike_length_test: int = -1
    # Dir name for the spike
    spike_dir_name: str = "spike"
    # Dir name for the image
    img_dir_name: str = "gt"
    # Dir name for the annotations
    annotation_dir_name: str = "annotations"
    # Rate. (-1 denotes variant)
    rate: float = 0.6


class DetectionDataset(BaseDataset):
    """Detection dataset for spike data."""
    
    def __init__(self, cfg: DetectionDatasetConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
    def prepare_data(self):
        """Specify the spike, image and annotation files to be loaded."""
        # Spike
        self.spike_dir = self.cfg.root_dir / self.split / self.cfg.spike_dir_name
        self.spike_list = self.get_spike_files(self.spike_dir)
        
        # Ground truth images
        if self.cfg.with_img == True:
            self.img_dir = self.cfg.root_dir / self.split / self.cfg.img_dir_name
            self.img_list = self.get_image_files(self.img_dir)
            
        # Annotations
        self.annotation_dir = self.cfg.root_dir / self.split / self.cfg.annotation_dir_name
        self.annotation_list = self.get_annotation_files(self.annotation_dir)
        
        # Verify that we have the same number of spikes, images, and annotations
        if len(self.img_list) != len(self.spike_list) or len(self.annotation_list) != len(self.spike_list):
            warnings.warn(f"Lengths of image list ({len(self.img_list)}), spike list ({len(self.spike_list)}) and annotation list ({len(self.annotation_list)}) should be equal.")
            
    def get_annotation_files(self, path: Path):
        """Recognize annotation files automatically."""
        files = [f for f in path.glob("**/*") if re.match(r".*\.(txt|xml|json)$", f.name, re.IGNORECASE)]
        return sorted(files)
        
    def get_annotations(self, idx):
        """Get the annotations from the given idx."""
        annotation_name = str(self.annotation_list[idx])
        # For now, we'll just return the file path
        # In a more complete implementation, we would parse the annotation file
        return annotation_name
        
    def __getitem__(self, idx: int):
        # Build source if not already done
        if not hasattr(self, 'spike_length'):
            self.build_source(self.split)
            
        # Load data
        if self.cfg.use_cache == True:
            spike, img = self.cache_spkimg[idx]
            spike = spike.to(torch.float32)
        else:
            spike = self.get_spike(idx)
            img = self.get_img(idx)
            
        # Handle None values
        if spike is None:
            raise ValueError(f"Spike data at index {idx} is None")
        if img is None and self.cfg.with_img:
            raise ValueError(f"Image data at index {idx} is None")
            
        # Get annotations
        annotations = self.get_annotations(idx)
        
        # Handle empty annotations
        if annotations is None:
            warnings.warn(f"Annotations at index {idx} are None")

        # Process data
        if self.cfg.use_aug == True and self.split == "train":
            spike, img = self.augmentor(spike, img)

        # Rate
        rate = self.cfg.rate

        # Spike, gt_img, and annotations are necessary
        batch = {"spike": spike, "gt_img": img, "annotations": annotations, "rate": rate}
        return batch