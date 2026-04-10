"""
Unit tests for optical flow components in SpikeZoo.
"""

import unittest
import torch
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the spikezoo package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spikezoo.models.optical_flow_model import OpticalFlowModel, OpticalFlowModelConfig
from spikezoo.datasets.optical_flow_dataset import OpticalFlowDataset, OpticalFlowDatasetConfig
from spikezoo.archs.optical_flow.nets import (
    OpticalFlowNet, EventOpticalFlowNet, PyramidOpticalFlowNet, create_optical_flow_network
)
from spikezoo.archs.optical_flow.layers import (
    FlowRefinementLayer, WarpingLayer, FlowRegularizationLayer, 
    MultiScaleFlowFusion, EventFeatureExtractor, FlowConfidenceLayer
)
from spikezoo.archs.optical_flow.blocks import (
    ResidualFlowBlock, DenseFlowBlock, AttentionFlowBlock,
    PyramidProcessingBlock, FlowEstimationBlock, MultiScaleFlowBlock,
    FlowRefinementBlock
)
from spikezoo.archs.optical_flow.modules import (
    FlowPyramidGenerator, FlowWarper, MultiScaleFlowProcessor,
    FlowFeatureExtractor, FlowFusionModule, FlowRegularizationModule,
    FlowConfidenceModule, FlowUpsamplingModule
)
from spikezoo.archs.optical_flow.utils import (
    flow_to_image, flow_warp, compute_flow_metrics, resize_flow,
    flow_smoothness, flow_divergence, flow_curl, generate_flow_pyramid,
    flow_confidence
)


class TestOpticalFlowNetworks(unittest.TestCase):
    """Test optical flow network architectures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.height = 64
        self.width = 64
        self.input_channels = 2
        self.output_channels = 2
    
    def test_basic_optical_flow_net(self):
        """Test basic optical flow network."""
        # Create network
        net = OpticalFlowNet(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            hidden_dim=32,
            num_layers=3
        )
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.input_channels, self.height, self.width)
        with torch.no_grad():
            output = net(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_channels, self.height, self.width))
    
    def test_event_optical_flow_net(self):
        """Test event optical flow network."""
        # Create network
        net = EventOpticalFlowNet(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            hidden_dim=32,
            num_layers=3,
            temporal_bins=5
        )
        
        # Test forward pass with 4D input
        x = torch.randn(self.batch_size, 5, self.height, self.width)
        with torch.no_grad():
            output = net(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_channels, self.height, self.width))
    
    def test_pyramid_optical_flow_net(self):
        """Test pyramid optical flow network."""
        # Create network
        net = PyramidOpticalFlowNet(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            base_channels=16,
            num_levels=2
        )
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.input_channels, self.height, self.width)
        with torch.no_grad():
            output = net(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_channels, self.height, self.width))
    
    def test_network_factory(self):
        """Test network factory function."""
        # Test all network types
        network_types = ['basic', 'event', 'pyramid']
        
        for net_type in network_types:
            with self.subTest(network_type=net_type):
                if net_type == 'event':
                    net = create_optical_flow_network(
                        net_type,
                        input_channels=2,
                        output_channels=2,
                        hidden_dim=16,
                        num_layers=2,
                        temporal_bins=3
                    )
                else:
                    net = create_optical_flow_network(
                        net_type,
                        input_channels=2,
                        output_channels=2,
                        hidden_dim=16,
                        num_layers=2
                    )
                
                self.assertIsInstance(net, torch.nn.Module)


class TestOpticalFlowLayers(unittest.TestCase):
    """Test optical flow layers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.height = 32
        self.width = 32
        self.channels = 16
    
    def test_flow_refinement_layer(self):
        """Test flow refinement layer."""
        layer = FlowRefinementLayer(self.channels, self.channels)
        
        flow = torch.randn(self.batch_size, self.channels, self.height, self.width)
        features = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        with torch.no_grad():
            output = layer(flow, features)
        
        self.assertEqual(output.shape, flow.shape)
    
    def test_warping_layer(self):
        """Test warping layer."""
        layer = WarpingLayer()
        
        image = torch.randn(self.batch_size, 3, self.height, self.width)
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        with torch.no_grad():
            warped = layer(image, flow)
        
        self.assertEqual(warped.shape, image.shape)
    
    def test_flow_regularization_layer(self):
        """Test flow regularization layer."""
        layer = FlowRegularizationLayer()
        
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        with torch.no_grad():
            output = layer(flow)
        
        self.assertEqual(output.shape, flow.shape)
    
    def test_multi_scale_flow_fusion(self):
        """Test multi-scale flow fusion layer."""
        layer = MultiScaleFlowFusion(num_scales=3)
        
        flows = [
            torch.randn(self.batch_size, 2, 16, 16),
            torch.randn(self.batch_size, 2, 32, 32),
            torch.randn(self.batch_size, 2, 64, 64)
        ]
        
        with torch.no_grad():
            fused = layer(flows)
        
        self.assertEqual(fused.shape, (self.batch_size, 2, 64, 64))
    
    def test_event_feature_extractor(self):
        """Test event feature extractor."""
        extractor = EventFeatureExtractor(input_channels=2, base_channels=16, num_layers=2)
        
        x = torch.randn(self.batch_size, 2, self.height, self.width)
        
        with torch.no_grad():
            features = extractor(x)
        
        self.assertEqual(features.shape[0], self.batch_size)
        self.assertGreater(features.shape[1], 0)
    
    def test_flow_confidence_layer(self):
        """Test flow confidence layer."""
        layer = FlowConfidenceLayer(self.channels)
        
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        features = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        with torch.no_grad():
            flow_out, confidence = layer(flow, features)
        
        self.assertEqual(flow_out.shape, flow.shape)
        self.assertEqual(confidence.shape, (self.batch_size, 1, self.height, self.width))


class TestOpticalFlowBlocks(unittest.TestCase):
    """Test optical flow blocks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.height = 32
        self.width = 32
        self.channels = 16
    
    def test_residual_flow_block(self):
        """Test residual flow block."""
        block = ResidualFlowBlock(self.channels, self.channels)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        with torch.no_grad():
            output = block(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_dense_flow_block(self):
        """Test dense flow block."""
        block = DenseFlowBlock(self.channels, growth_rate=8, num_layers=2)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        with torch.no_grad():
            output = block(x)
        
        expected_channels = self.channels + 2 * 8
        self.assertEqual(output.shape, (self.batch_size, expected_channels, self.height, self.width))
    
    def test_attention_flow_block(self):
        """Test attention flow block."""
        block = AttentionFlowBlock(self.channels)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        with torch.no_grad():
            output = block(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_pyramid_processing_block(self):
        """Test pyramid processing block."""
        block = PyramidProcessingBlock(self.channels, self.channels, num_scales=2)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        with torch.no_grad():
            output = block(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_flow_estimation_block(self):
        """Test flow estimation block."""
        block = FlowEstimationBlock(self.channels)
        
        features1 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        features2 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        with torch.no_grad():
            flow = block(features1, features2)
        
        self.assertEqual(flow.shape, (self.batch_size, 2, self.height, self.width))
    
    def test_multi_scale_flow_block(self):
        """Test multi-scale flow block."""
        block = MultiScaleFlowBlock(base_channels=16, num_scales=2)
        
        x = torch.randn(self.batch_size, 2, self.height, self.width)
        
        with torch.no_grad():
            flows = block(x)
        
        self.assertEqual(len(flows), 2)
        for flow in flows:
            self.assertEqual(flow.shape[0], self.batch_size)
            self.assertEqual(flow.shape[1], 2)
    
    def test_flow_refinement_block(self):
        """Test flow refinement block."""
        block = FlowRefinementBlock(self.channels)
        
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        features = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        with torch.no_grad():
            refined_flow = block(flow, features)
        
        self.assertEqual(refined_flow.shape, flow.shape)


class TestOpticalFlowModules(unittest.TestCase):
    """Test optical flow modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.height = 32
        self.width = 32
    
    def test_flow_pyramid_generator(self):
        """Test flow pyramid generator."""
        module = FlowPyramidGenerator(num_levels=3)
        
        image = torch.randn(self.batch_size, 3, self.height, self.width)
        
        with torch.no_grad():
            pyramid = module(image)
        
        self.assertEqual(len(pyramid), 3)
        for i, level in enumerate(pyramid):
            self.assertEqual(level.shape[0], self.batch_size)
    
    def test_flow_warper(self):
        """Test flow warper."""
        module = FlowWarper()
        
        image = torch.randn(self.batch_size, 3, self.height, self.width)
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        with torch.no_grad():
            warped = module(image, flow)
        
        self.assertEqual(warped.shape, image.shape)
    
    def test_multi_scale_flow_processor(self):
        """Test multi-scale flow processor."""
        module = MultiScaleFlowProcessor(base_channels=16, num_scales=2)
        
        flows = [
            torch.randn(self.batch_size, 2, 16, 16),
            torch.randn(self.batch_size, 2, 32, 32)
        ]
        
        with torch.no_grad():
            processed = module(flows)
        
        self.assertEqual(len(processed), 2)
        for features in processed:
            self.assertEqual(features.shape[0], self.batch_size)
    
    def test_flow_feature_extractor(self):
        """Test flow feature extractor."""
        module = FlowFeatureExtractor(input_channels=2, base_channels=16, num_levels=2)
        
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        with torch.no_grad():
            features = module(flow)
        
        self.assertEqual(len(features), 2)
        for feat in features:
            self.assertEqual(feat.shape[0], self.batch_size)
    
    def test_flow_fusion_module(self):
        """Test flow fusion module."""
        module = FlowFusionModule(num_inputs=3, fusion_method='sum')
        
        flows = [
            torch.randn(self.batch_size, 2, self.height, self.width),
            torch.randn(self.batch_size, 2, self.height, self.width),
            torch.randn(self.batch_size, 2, self.height, self.width)
        ]
        
        with torch.no_grad():
            fused = module(flows)
        
        self.assertEqual(fused.shape, flows[0].shape)
    
    def test_flow_regularization_module(self):
        """Test flow regularization module."""
        module = FlowRegularizationModule(regularization_type='smoothness')
        
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        with torch.no_grad():
            regularized = module(flow)
        
        self.assertEqual(regularized.shape, flow.shape)
    
    def test_flow_confidence_module(self):
        """Test flow confidence module."""
        module = FlowConfidenceModule(in_channels=18)  # 2 for flow + 16 for features
        
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        features = torch.randn(self.batch_size, 16, self.height, self.width)
        
        with torch.no_grad():
            flow_out, confidence = module(flow, features)
        
        self.assertEqual(flow_out.shape, flow.shape)
        self.assertEqual(confidence.shape, (self.batch_size, 1, self.height, self.width))
    
    def test_flow_upsampling_module(self):
        """Test flow upsampling module."""
        module = FlowUpsamplingModule(upscale_factor=2)
        
        flow = torch.randn(self.batch_size, 2, 16, 16)
        
        with torch.no_grad():
            upsampled = module(flow)
        
        self.assertEqual(upsampled.shape, (self.batch_size, 2, 32, 32))


class TestOpticalFlowUtils(unittest.TestCase):
    """Test optical flow utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.height = 32
        self.width = 32
    
    def test_flow_to_image(self):
        """Test flow to image conversion."""
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        image = flow_to_image(flow)
        
        self.assertEqual(image.shape, (self.batch_size, 3, self.height, self.width))
    
    def test_flow_warp(self):
        """Test flow warping."""
        image = torch.randn(self.batch_size, 3, self.height, self.width)
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        warped = flow_warp(image, flow)
        
        self.assertEqual(warped.shape, image.shape)
    
    def test_compute_flow_metrics(self):
        """Test flow metrics computation."""
        flow_pred = torch.randn(self.batch_size, 2, self.height, self.width)
        flow_gt = torch.randn(self.batch_size, 2, self.height, self.width)
        
        metrics = compute_flow_metrics(flow_pred, flow_gt)
        
        self.assertIn('epe', metrics)
        self.assertIn('angular_error', metrics)
        self.assertIn('outlier_rate', metrics)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(metrics['epe'], 0)
        self.assertGreaterEqual(metrics['angular_error'], 0)
        self.assertGreaterEqual(metrics['outlier_rate'], 0)
        self.assertLessEqual(metrics['outlier_rate'], 1)
    
    def test_resize_flow(self):
        """Test flow resizing."""
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        target_size = (64, 64)
        
        resized = resize_flow(flow, target_size)
        
        self.assertEqual(resized.shape, (self.batch_size, 2, 64, 64))
    
    def test_flow_smoothness(self):
        """Test flow smoothness computation."""
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        smoothness = flow_smoothness(flow)
        
        self.assertIsInstance(smoothness, torch.Tensor)
        self.assertEqual(smoothness.shape, torch.Size([]))
    
    def test_flow_divergence(self):
        """Test flow divergence computation."""
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        divergence = flow_divergence(flow)
        
        self.assertEqual(divergence.shape, (self.batch_size, 1, self.height, self.width))
    
    def test_flow_curl(self):
        """Test flow curl computation."""
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        curl = flow_curl(flow)
        
        self.assertEqual(curl.shape, (self.batch_size, 1, self.height, self.width))
    
    def test_generate_flow_pyramid(self):
        """Test flow pyramid generation."""
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        pyramid = generate_flow_pyramid(flow, num_levels=3)
        
        self.assertEqual(len(pyramid), 3)
        for i, level in enumerate(pyramid):
            self.assertEqual(level.shape[0], self.batch_size)
            self.assertEqual(level.shape[1], 2)
    
    def test_flow_confidence(self):
        """Test flow confidence computation."""
        flow = torch.randn(self.batch_size, 2, self.height, self.width)
        
        confidence = flow_confidence(flow)
        
        self.assertEqual(confidence.shape, (self.batch_size, 1, self.height, self.width))


class TestOpticalFlowModel(unittest.TestCase):
    """Test optical flow model interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.height = 64
        self.width = 64
    
    def test_model_creation(self):
        """Test optical flow model creation."""
        config = OpticalFlowModelConfig(
            model_name="optical_flow",
            network_type="basic",
            input_channels=2,
            output_channels=2,
            hidden_dim=32,
            num_layers=3
        )
        
        model = OpticalFlowModel(config)
        
        self.assertIsInstance(model, OpticalFlowModel)
        self.assertEqual(model.cfg, config)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        config = OpticalFlowModelConfig(
            model_name="optical_flow",
            network_type="basic",
            input_channels=2,
            output_channels=2,
            hidden_dim=32,
            num_layers=3
        )
        
        model = OpticalFlowModel(config)
        
        # Create dummy batch
        batch = {
            "events": torch.randn(self.batch_size, 2, self.height, self.width)
        }
        
        # Test forward pass
        outputs = model.get_outputs_dict(batch)
        
        self.assertIn("flow_pred", outputs)
        self.assertEqual(outputs["flow_pred"].shape, (self.batch_size, 2, self.height, self.width))
    
    def test_model_loss_computation(self):
        """Test model loss computation."""
        config = OpticalFlowModelConfig(
            model_name="optical_flow",
            network_type="basic",
            input_channels=2,
            output_channels=2,
            hidden_dim=32,
            num_layers=3
        )
        
        model = OpticalFlowModel(config)
        
        # Create dummy data
        outputs = {
            "flow_pred": torch.randn(self.batch_size, 2, self.height, self.width)
        }
        
        batch = {
            "flow_gt": torch.randn(self.batch_size, 2, self.height, self.width)
        }
        
        loss_weight_dict = {"epe_loss": 1.0}
        
        # Test loss computation
        loss_dict, loss_values_dict = model.get_loss_dict(outputs, batch, loss_weight_dict)
        
        self.assertIn("epe_loss", loss_dict)
        self.assertIn("epe_loss", loss_values_dict)
        self.assertIsInstance(loss_dict["epe_loss"], torch.Tensor)
        self.assertIsInstance(loss_values_dict["epe_loss"], float)
    
    def test_model_visualization(self):
        """Test model visualization."""
        config = OpticalFlowModelConfig(
            model_name="optical_flow",
            network_type="basic",
            input_channels=2,
            output_channels=2,
            hidden_dim=32,
            num_layers=3
        )
        
        model = OpticalFlowModel(config)
        
        # Create dummy data
        batch = {
            "events": torch.randn(self.batch_size, 2, self.height, self.width),
            "flow_gt": torch.randn(self.batch_size, 2, self.height, self.width),
            "image1": torch.randn(self.batch_size, 3, self.height, self.width),
            "image2": torch.randn(self.batch_size, 3, self.height, self.width)
        }
        
        outputs = {
            "flow_pred": torch.randn(self.batch_size, 2, self.height, self.width)
        }
        
        # Test visualization
        visual_dict = model.get_visual_dict(batch, outputs)
        
        expected_keys = {"events", "flow_gt", "flow_pred", "image1", "image2"}
        self.assertEqual(set(visual_dict.keys()), expected_keys)
        
        # Check tensor shapes
        for key, value in visual_dict.items():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.shape[0], self.batch_size)


class TestOpticalFlowDataset(unittest.TestCase):
    """Test optical flow dataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.height = 64
        self.width = 64
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_creation(self):
        """Test optical flow dataset creation."""
        cfg = OpticalFlowDatasetConfig(
            data_root=self.temp_dir,
            split="train",
            height=self.height,
            width=self.width,
            sequence_length=5,
            augment=False
        )
        
        # Create indices file
        indices_file = os.path.join(self.temp_dir, "train_indices.txt")
        with open(indices_file, 'w') as f:
            f.write("sequence_000000\n")
            f.write("sequence_000001\n")
        
        dataset = OpticalFlowDataset(cfg)
        
        self.assertIsInstance(dataset, OpticalFlowDataset)
        self.assertEqual(len(dataset), 2)
    
    def test_dataset_item_access(self):
        """Test dataset item access."""
        cfg = OpticalFlowDatasetConfig(
            data_root=self.temp_dir,
            split="train",
            height=self.height,
            width=self.width,
            sequence_length=5,
            augment=False
        )
        
        # Create indices file
        indices_file = os.path.join(self.temp_dir, "train_indices.txt")
        with open(indices_file, 'w') as f:
            f.write("sequence_000000\n")
        
        dataset = OpticalFlowDataset(cfg)
        
        if len(dataset) > 0:
            item = dataset[0]
            
            expected_keys = {"events", "flow_gt", "image1", "image2", "index"}
            self.assertEqual(set(item.keys()), expected_keys)
            
            # Check tensor shapes (approximate)
            self.assertEqual(item["events"].shape[0], 2)  # 2 channels for pos/neg events
            self.assertEqual(item["flow_gt"].shape[0], 2)  # 2 channels for flow components
            self.assertEqual(item["image1"].shape[0], 3)  # 3 channels for RGB
            self.assertEqual(item["image2"].shape[0], 3)  # 3 channels for RGB


if __name__ == '__main__':
    unittest.main()