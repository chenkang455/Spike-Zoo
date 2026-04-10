import torch
from torch.utils.data import Dataset
import numpy as np


class RecognitionDataset(Dataset):
    """
    Base dataset class for recognition tasks.
    Handles different input formats and provides unified interface.
    """
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: Input data (spike streams, images, etc.)
            labels: Corresponding labels
            transform: Optional transform to be applied on data
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class DMERDataset(RecognitionDataset):
    """
    Dataset for DMER Net - handles temporal spike data with denoising and motion enhancement
    """
    def __init__(self, data, labels, T=7, transform=None):
        """
        Args:
            data: Temporal spike data with shape (N, T, H, W)
            labels: Labels for classification
            T: Number of time steps
            transform: Optional transform
        """
        super().__init__(data, labels, transform)
        self.T = T

    def __getitem__(self, idx):
        sample = self.data[idx]  # Shape: (T, H, W)
        label = self.labels[idx]

        # Ensure correct shape
        if sample.ndim == 3 and sample.shape[0] == self.T:
            # Add channel dimension if missing
            if sample.shape[0] == self.T:
                sample = sample.unsqueeze(1)  # Add channel dim: (T, 1, H, W)
        else:
            raise ValueError(f"Expected input shape (T, H, W) where T={self.T}, got {sample.shape}")

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class RPSDataset(RecognitionDataset):
    """
    Dataset for Rock-Paper-Scissors recognition task
    """
    def __init__(self, data, labels, T=None, transform=None):
        """
        Args:
            data: Spike data with shape (N, T, H, W)
            labels: Labels for RPS classification (0: Rock, 1: Paper, 2: Scissors)
            T: Not used specifically but kept for consistency
            transform: Optional transform
        """
        super().__init__(data, labels, transform)

    def __getitem__(self, idx):
        sample = self.data[idx]  # Shape: (T, H, W)
        label = self.labels[idx]

        # Ensure correct shape for RPSNet
        if sample.ndim == 3:
            # Add channel dimension
            sample = sample.unsqueeze(1)  # (T, 1, H, W)
        else:
            raise ValueError(f"Expected input shape (T, H, W), got {sample.shape}")

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class VGGRDataset(RecognitionDataset):
    """
    Dataset for VGG-based recognition models
    """
    def __init__(self, data, labels, T=5, transform=None):
        """
        Args:
            data: Spike data with shape (N, T, H, W)
            labels: Classification labels
            T: Number of time steps
            transform: Optional transform
        """
        super().__init__(data, labels, transform)
        self.T = T

    def __getitem__(self, idx):
        sample = self.data[idx]  # Shape: (T, H, W)
        label = self.labels[idx]

        # Process for VGG models
        if sample.ndim == 3 and sample.shape[0] == self.T:
            # Add channel dimension
            sample = sample.unsqueeze(1)  # (T, 1, H, W)
        else:
            raise ValueError(f"Expected input shape (T, H, W) where T={self.T}, got {sample.shape}")

        if self.transform:
            sample = self.transform(sample)

        return sample, label