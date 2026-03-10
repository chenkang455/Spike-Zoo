"""
Unit tests for detection model.
"""

import unittest
import torch
from src.detection.detection_model import DetectionModel, DetectionModelConfig


class TestDetectionModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DetectionModelConfig()
        self.model = DetectionModel(self.config)
        
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, DetectionModel)
        self.assertEqual(self.model.cfg.model_name, "detection")
        
    def test_build_network(self):
        """Test building the network."""
        self.model.build_network()
        # For detection model, saccade_input should be None initially
        self.assertIsNone(self.model.saccade_input)
        
    def test_process_frame(self):
        """Test processing a single frame."""
        # Create a dummy spike frame
        spike_frame = torch.randn(250, 400)
        
        # Process the frame
        self.model.build_network()
        attention_boxes = self.model.process_frame(spike_frame)
        
        # Check that we got some attention boxes
        self.assertIsInstance(attention_boxes, torch.Tensor)
        
    def test_get_outputs_dict(self):
        """Test getting outputs dictionary."""
        # Create a dummy batch
        batch = {
            "spike": torch.randn(10, 250, 400),  # 10 frames of 250x400
            "gt_img": torch.randn(1, 250, 400),
            "rate": 0.6
        }
        
        # Get outputs
        self.model.build_network()
        outputs = self.model.get_outputs_dict(batch)
        
        # Check that we got attention boxes for each frame
        self.assertIn("attention_boxes", outputs)
        self.assertEqual(len(outputs["attention_boxes"]), 10)


if __name__ == '__main__':
    unittest.main()