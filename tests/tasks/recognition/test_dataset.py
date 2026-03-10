import torch
import unittest
import numpy as np
from src.tasks.recognition.dataset import (
    RecognitionDataset,
    DMERDataset,
    RPSDataset,
    VGGRDataset
)


class TestRecognitionDataset(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.T = 7
        self.H = 64
        self.W = 64
        self.num_classes = 10
        
        # Create sample data
        self.data = torch.randn(self.batch_size, self.T, self.H, self.W)
        self.labels = torch.randint(0, self.num_classes, (self.batch_size,))

    def test_recognition_dataset_base(self):
        dataset = RecognitionDataset(self.data, self.labels)
        self.assertEqual(len(dataset), self.batch_size)
        
        # Test get item
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (self.T, self.H, self.W))
        self.assertIsInstance(label, torch.Tensor)

    def test_dmer_dataset(self):
        dataset = DMERDataset(self.data, self.labels, T=self.T)
        self.assertEqual(len(dataset), self.batch_size)
        
        # Test get item
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (self.T, 1, self.H, self.W))  # Channel added
        self.assertIsInstance(label, torch.Tensor)

    def test_rps_dataset(self):
        # RPS dataset typically has different T (e.g., 5)
        rps_T = 5
        rps_data = torch.randn(self.batch_size, rps_T, 250, 400)
        rps_labels = torch.randint(0, 3, (self.batch_size,))  # 3 classes for RPS
        
        dataset = RPSDataset(rps_data, rps_labels)
        self.assertEqual(len(dataset), self.batch_size)
        
        # Test get item
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (rps_T, 1, 250, 400))  # Channel added
        self.assertIsInstance(label, torch.Tensor)

    def test_vgg_dataset(self):
        vgg_T = 5
        vgg_data = torch.randn(self.batch_size, vgg_T, 224, 224)
        vgg_labels = torch.randint(0, self.num_classes, (self.batch_size,))
        
        dataset = VGGRDataset(vgg_data, vgg_labels, T=vgg_T)
        self.assertEqual(len(dataset), self.batch_size)
        
        # Test get item
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (vgg_T, 1, 224, 224))  # Channel added
        self.assertIsInstance(label, torch.Tensor)

    def test_dataset_with_transform(self):
        def dummy_transform(x):
            return x * 2
            
        dataset = RecognitionDataset(self.data, self.labels, transform=dummy_transform)
        sample, label = dataset[0]
        expected = self.data[0] * 2
        self.assertTrue(torch.equal(sample, expected))


if __name__ == '__main__':
    unittest.main()