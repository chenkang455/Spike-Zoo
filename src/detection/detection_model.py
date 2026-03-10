"""
Detection model for Spike-Zoo.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Union, List, Tuple
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from .attention_select import SaccadeInput


@dataclass
class DetectionModelConfig(BaseModelConfig):
    """Configuration for detection model."""
    # Model name
    model_name: str = "detection"
    # Model file name
    model_file_name: str = "detection_model"
    # Class name
    model_cls_name: str = "DetectionModel"
    # Additional parameters for detection
    box_size: int = 15
    attention_threshold: float = 40.0
    extend_edge: int = 7


class DetectionModel(BaseModel):
    """Detection model for spike data."""
    
    def __init__(self, cfg: DetectionModelConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.saccade_input = None
        
    def build_network(self, mode: str = "train", version: str = "local"):
        """
        Build the detection network.
        
        Args:
            mode: Model mode (train/eval)
            version: Model version
        """
        self.cfg.mode = mode
        # For detection, we don't build a traditional neural network
        # Instead, we initialize the saccade input processor
        self.saccade_input = None
        return self
    
    def process_frame(self, spike_frame: torch.Tensor):
        """
        Process a single spike frame for detection.
        
        Args:
            spike_frame: Input spike frame tensor of shape (H, W)
            
        Returns:
            attention_boxes: Tensor of attention boxes
        """
        if self.saccade_input is None:
            # Initialize saccade input processor
            height, width = spike_frame.shape
            self.saccade_input = SaccadeInput(
                spike_h=height,
                spike_w=width,
                box_size=self.cfg.box_size,
                device=spike_frame.device
            )
        elif spike_frame.shape != (self.saccade_input.spike_h, self.saccade_input.spike_w):
            # Reinitialize for different input size
            height, width = spike_frame.shape
            self.saccade_input = SaccadeInput(
                spike_h=height,
                spike_w=width,
                box_size=self.cfg.box_size,
                device=spike_frame.device
            )
            
        # Update DNF with the new spike frame
        self.saccade_input.update_dnf(spike_frame)
        
        # Get attention locations
        attention_boxes = self.saccade_input.get_attention_location()
        
        return attention_boxes
    
    def get_outputs_dict(self, batch):
        """Get outputs dictionary."""
        outputs = {}
        spike = batch["spike"]
        
        # Handle None input
        if spike is None:
            raise ValueError("Input spike data is None")
        
        # Process each frame in the spike sequence
        attention_boxes_list = []
        for t in range(spike.shape[0]):
            frame = spike[t]
            # Handle None frame
            if frame is None:
                warnings.warn(f"Frame at timestep {t} is None, skipping")
                continue
            attention_boxes = self.process_frame(frame)
            attention_boxes_list.append(attention_boxes)
            
        outputs["attention_boxes"] = attention_boxes_list
        return outputs