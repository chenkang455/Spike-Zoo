import torch
import unittest
from src.tasks.recognition.models import (
    dmer_net18, 
    dmer_net34, 
    RPSNet, 
    spike_stream_vgg11_bn,
    spike_stream_vgg13_bn,
    spike_stream_vgg16_bn,
    spike_stream_vgg19_bn
)


class TestRecognitionModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.T = 7  # Time steps
        self.H = 64  # Height
        self.W = 64  # Width
        self.num_classes = 10
        
        # Create sample input data
        self.sample_input_dmer = torch.randn(self.batch_size, self.T, 1, self.H, self.W)
        self.sample_input_rps = torch.randn(self.batch_size, 5, 1, 250, 400)  # Different size for RPS
        self.sample_input_vgg = torch.randn(self.batch_size, 5, 1, 224, 224)  # VGG input size

    def test_dmer_net18(self):
        model = dmer_net18(num_classes=self.num_classes, T=self.T)
        output = model(self.sample_input_dmer)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_dmer_net34(self):
        model = dmer_net34(num_classes=self.num_classes, T=self.T)
        output = model(self.sample_input_dmer)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_rps_net(self):
        model = RPSNet()
        output = model(self.sample_input_rps)
        self.assertEqual(output.shape, (self.batch_size, 3))  # 3 classes for RPS

    def test_spike_stream_vgg11_bn(self):
        model = spike_stream_vgg11_bn(num_classes=self.num_classes, T=5)
        output = model(self.sample_input_vgg)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_spike_stream_vgg13_bn(self):
        model = spike_stream_vgg13_bn(num_classes=self.num_classes, T=5)
        output = model(self.sample_input_vgg)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_spike_stream_vgg16_bn(self):
        model = spike_stream_vgg16_bn(num_classes=self.num_classes, T=5)
        output = model(self.sample_input_vgg)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_spike_stream_vgg19_bn(self):
        model = spike_stream_vgg19_bn(num_classes=self.num_classes, T=5)
        output = model(self.sample_input_vgg)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


if __name__ == '__main__':
    unittest.main()