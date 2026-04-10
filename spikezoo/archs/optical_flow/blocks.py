"""
Building blocks for optical flow networks in SpikeZoo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResidualFlowBlock(nn.Module):
    """Residual block for optical flow networks."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """Initialize residual flow block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class DenseFlowBlock(nn.Module):
    """Dense block for optical flow networks."""
    
    def __init__(self, in_channels: int, growth_rate: int = 32, num_layers: int = 4):
        """Initialize dense flow block.
        
        Args:
            in_channels: Number of input channels
            growth_rate: Growth rate for dense connections
            num_layers: Number of layers in the block
        """
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_in_ch = in_channels + i * growth_rate
            layer = nn.Sequential(
                nn.BatchNorm2d(layer_in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(layer_in_ch, growth_rate, kernel_size=3, padding=1)
            )
            self.layers.append(layer)
        
        self.out_channels = in_channels + num_layers * growth_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        features = [x]
        
        for layer in self.layers:
            concatenated = torch.cat(features, dim=1)
            new_feature = layer(concatenated)
            features.append(new_feature)
        
        return torch.cat(features, dim=1)


class AttentionFlowBlock(nn.Module):
    """Attention-based block for optical flow networks."""
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        """Initialize attention flow block.
        
        Args:
            channels: Number of input/output channels
            reduction_ratio: Channel reduction ratio for attention
        """
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with attention applied
        """
        # Channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # Spatial attention
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights
        
        return x


class PyramidProcessingBlock(nn.Module):
    """Block for processing features at multiple scales."""
    
    def __init__(self, in_channels: int, out_channels: int, num_scales: int = 3):
        """Initialize pyramid processing block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_scales: Number of scales to process
        """
        super().__init__()
        self.num_scales = num_scales
        
        # Scale-specific processing
        self.scale_processors = nn.ModuleList()
        for i in range(num_scales):
            scale_factor = 2 ** i
            processor = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.scale_processors.append(processor)
        
        # Scale combination
        self.combiner = nn.Conv2d(out_channels * num_scales, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        processed_features = []
        
        for i, processor in enumerate(self.scale_processors):
            # Downsample if needed
            if i > 0:
                scale_factor = 2 ** i
                downsampled = F.adaptive_avg_pool2d(x, 
                                                   (x.shape[2] // scale_factor, 
                                                    x.shape[3] // scale_factor))
                processed = processor(downsampled)
                # Upsample back to original size
                processed = F.interpolate(processed, size=x.shape[2:4], 
                                        mode='bilinear', align_corners=False)
            else:
                processed = processor(x)
            
            processed_features.append(processed)
        
        # Combine features from all scales
        combined = torch.cat(processed_features, dim=1)
        output = self.combiner(combined)
        
        return output


class FlowEstimationBlock(nn.Module):
    """Block for estimating optical flow from feature pairs."""
    
    def __init__(self, feature_channels: int, flow_channels: int = 2):
        """Initialize flow estimation block.
        
        Args:
            feature_channels: Number of feature channels
            flow_channels: Number of flow channels (typically 2)
        """
        super().__init__()
        self.feature_channels = feature_channels
        self.flow_channels = flow_channels
        
        # Correlation layer (simplified)
        self.correlation = nn.Conv2d(feature_channels * 2, feature_channels, 
                                    kernel_size=1)
        
        # Flow estimation layers
        self.flow_estimator = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, flow_channels, kernel_size=1)
        )
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Estimate flow between two feature maps.
        
        Args:
            features1: First feature map
            features2: Second feature map
            
        Returns:
            Estimated flow field
        """
        # Concatenate features
        concatenated = torch.cat([features1, features2], dim=1)
        
        # Compute correlation
        correlation = self.correlation(concatenated)
        
        # Estimate flow
        flow = self.flow_estimator(correlation)
        
        return flow


class MultiScaleFlowBlock(nn.Module):
    """Block for multi-scale flow estimation."""
    
    def __init__(self, base_channels: int, num_scales: int = 3):
        """Initialize multi-scale flow block.
        
        Args:
            base_channels: Base number of channels
            num_scales: Number of scales
        """
        super().__init__()
        self.num_scales = num_scales
        
        # Feature extractors for each scale
        self.feature_extractors = nn.ModuleList()
        self.flow_estimators = nn.ModuleList()
        
        for i in range(num_scales):
            channels = base_channels * (2 ** i)
            self.feature_extractors.append(
                nn.Sequential(
                    nn.Conv2d(2, channels, kernel_size=3, padding=1),  # Assuming 2-channel input
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
            
            self.flow_estimators.append(
                FlowEstimationBlock(channels, 2)
            )
    
    def forward(self, x: torch.Tensor) -> list:
        """Estimate flow at multiple scales.
        
        Args:
            x: Input tensor
            
        Returns:
            List of flow estimates at different scales
        """
        flows = []
        
        for i in range(self.num_scales):
            # Downsample input if needed
            if i > 0:
                scale_factor = 2 ** i
                downsampled = F.adaptive_avg_pool2d(x, 
                                                   (x.shape[2] // scale_factor, 
                                                    x.shape[3] // scale_factor))
                features = self.feature_extractors[i](downsampled)
            else:
                features = self.feature_extractors[i](x)
            
            # Split features for flow estimation
            feat1, feat2 = features.chunk(2, dim=1)
            flow = self.flow_estimators[i](feat1, feat2)
            flows.append(flow)
        
        return flows


class FlowRefinementBlock(nn.Module):
    """Block for refining optical flow estimates."""
    
    def __init__(self, channels: int, num_iterations: int = 3):
        """Initialize flow refinement block.
        
        Args:
            channels: Number of channels
            num_iterations: Number of refinement iterations
        """
        super().__init__()
        self.num_iterations = num_iterations
        
        # Refinement network
        self.refiner = nn.Sequential(
            nn.Conv2d(channels + 2, channels, kernel_size=3, padding=1),  # +2 for flow channels
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1)  # Output flow
        )
    
    def forward(self, flow: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Refine flow estimate.
        
        Args:
            flow: Initial flow estimate
            features: Feature tensor
            
        Returns:
            Refined flow estimate
        """
        refined_flow = flow
        
        for _ in range(self.num_iterations):
            # Concatenate flow with features
            input_tensor = torch.cat([refined_flow, features], dim=1)
            
            # Refine
            delta_flow = self.refiner(input_tensor)
            refined_flow = refined_flow + delta_flow
        
        return refined_flow