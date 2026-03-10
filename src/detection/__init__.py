"""
Detection package for Spike-Zoo.
"""

from .detection_model import DetectionModel, DetectionModelConfig
from .detection_dataset import DetectionDataset, DetectionDatasetConfig

__all__ = [
    "DetectionModel",
    "DetectionModelConfig",
    "DetectionDataset",
    "DetectionDatasetConfig"
]