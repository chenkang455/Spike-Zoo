"""
Reusable modules for optical flow networks in SpikeZoo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from spikezoo.archs.optical_flow.layers import WarpingLayer


class FlowPyramidGenerator(nn.Module):
    """Module for generating image pyramids for optical flow."""
    
    def __init__(self, num_levels: int = 3):
        """Initialize flow pyramid generator.
        
        Args:
            num_levels: Number of pyramid levels
        """
        super().__init__()
        self.num_levels = num_levels
    
    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Generate image pyramid.
        
        Args:
            image: Input image tensor of shape (B, C, H, W)
            
        Returns:
            List of image tensors at different scales
        """
        pyramid = [image]
        
        for i in range(1, self.num_levels):
            scale_factor = 2 ** i
            scaled = F.interpolate(image, 
                                 scale_factor=1.0/scale_factor,
                                 mode='bilinear', 
                                 align_corners=False)
            pyramid.append(scaled)
        
        return pyramid


class FlowWarper(nn.Module):
    """Module for warping images using optical flow."""
    
    def __init__(self, interpolation_mode: str = 'bilinear'):
        """Initialize flow warper.
        
        Args:
            interpolation_mode: Interpolation mode for warping
        """
        super().__init__()
        self.warping_layer = WarpingLayer(interpolation_mode)
    
    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp image using optical flow.
        
        Args:
            image: Image tensor of shape (B, C, H, W)
            flow: Flow tensor of shape (B, 2, H, W)
            
        Returns:
            Warped image tensor
        """
        return self.warping_layer(image, flow)


class MultiScaleFlowProcessor(nn.Module):
    """Module for processing optical flow at multiple scales."""
    
    def __init__(self, base_channels: int = 32, num_scales: int = 3):
        """Initialize multi-scale flow processor.
        
        Args:
            base_channels: Base number of channels
            num_scales: Number of scales
        """
        super().__init__()
        self.num_scales = num_scales
        
        # Processing modules for each scale
        self.processors = nn.ModuleList()
        for i in range(num_scales):
            channels = base_channels * (2 ** i)
            processor = nn.Sequential(
                nn.Conv2d(2, channels, kernel_size=3, padding=1),  # 2 for flow channels
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            self.processors.append(processor)
    
    def forward(self, flows: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process flows at multiple scales.
        
        Args:
            flows: List of flow tensors
            
        Returns:
            List of processed flow features
        """
        processed = []
        
        for i, (flow, processor) in enumerate(zip(flows, self.processors)):
            features = processor(flow)
            processed.append(features)
        
        return processed


class FlowFeatureExtractor(nn.Module):
    """Module for extracting features from flow fields."""
    
    def __init__(self, input_channels: int = 2, base_channels: int = 32, num_levels: int = 3):
        """Initialize flow feature extractor.
        
        Args:
            input_channels: Number of input channels (typically 2 for flow)
            base_channels: Base number of channels
            num_levels: Number of feature levels
        """
        super().__init__()
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.num_levels = num_levels
        
        # Feature extraction at multiple levels
        self.extractors = nn.ModuleList()
        in_ch = input_channels
        for i in range(num_levels):
            out_ch = base_channels * (2 ** i)
            extractor = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.extractors.append(extractor)
            in_ch = out_ch
    
    def forward(self, flow: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from flow field.
        
        Args:
            flow: Flow tensor of shape (B, 2, H, W)
            
        Returns:
            List of feature tensors at different levels
        """
        features = []
        x = flow
        
        for extractor in self.extractors:
            x = extractor(x)
            features.append(x)
        
        return features


class FlowFusionModule(nn.Module):
    """Module for fusing flow information from multiple sources."""
    
    def __init__(self, num_inputs: int, fusion_method: str = 'concat'):
        """Initialize flow fusion module.
        
        Args:
            num_inputs: Number of input flow tensors
            fusion_method: Fusion method ('concat', 'sum', 'attention')
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.fusion_method = fusion_method
        
        if fusion_method == 'attention':
            # Attention mechanism for weighted fusion
            self.attention_weights = nn.Parameter(torch.ones(num_inputs))
    
    def forward(self, flows: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple flow tensors.
        
        Args:
            flows: List of flow tensors
            
        Returns:
            Fused flow tensor
        """
        if self.fusion_method == 'concat':
            return torch.cat(flows, dim=1)
        elif self.fusion_method == 'sum':
            return sum(flows)
        elif self.fusion_method == 'attention':
            # Weighted sum with attention
            weights = F.softmax(self.attention_weights, dim=0)
            weighted_flows = [flow * weight for flow, weight in zip(flows, weights)]
            return sum(weighted_flows)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class FlowRegularizationModule(nn.Module):
    """Module for regularizing optical flow fields."""
    
    def __init__(self, regularization_type: str = 'smoothness'):
        """Initialize flow regularization module.
        
        Args:
            regularization_type: Type of regularization ('smoothness', 'divergence', 'curl')
        """
        super().__init__()
        self.regularization_type = regularization_type
    
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """Apply flow regularization.
        
        Args:
            flow: Flow tensor of shape (B, 2, H, W)
            
        Returns:
            Regularized flow tensor
        """
        if self.regularization_type == 'smoothness':
            return self._smoothness_regularization(flow)
        elif self.regularization_type == 'divergence':
            return self._divergence_regularization(flow)
        elif self.regularization_type == 'curl':
            return self._curl_regularization(flow)
        else:
            return flow
    
    def _smoothness_regularization(self, flow: torch.Tensor) -> torch.Tensor:
        """Apply smoothness regularization."""
        # Calculate flow gradients
        flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        
        # Zero padding for gradient dimensions
        pad_x = torch.zeros_like(flow_dx[:, :, :, :1])
        pad_y = torch.zeros_like(flow_dy[:, :, :1, :])
        
        flow_dx = torch.cat([flow_dx, pad_x], dim=3)
        flow_dy = torch.cat([flow_dy, pad_y], dim=2)
        
        # Apply smoothing
        smoothed_flow = flow - 0.1 * (flow_dx + flow_dy)
        return smoothed_flow
    
    def _divergence_regularization(self, flow: torch.Tensor) -> torch.Tensor:
        """Apply divergence regularization."""
        # Calculate divergence
        div = flow[:, 0:1, :, 1:] - flow[:, 0:1, :, :-1] + \
              flow[:, 1:2, 1:, :] - flow[:, 1:2, :-1, :]
        
        # Pad to match original size
        pad = torch.zeros_like(div[:, :, :1, :])
        div = torch.cat([div, pad], dim=2)
        pad = torch.zeros_like(div[:, :, :, :1])
        div = torch.cat([div, pad], dim=3)
        
        # Apply divergence correction
        corrected_flow = flow - 0.01 * div
        return corrected_flow
    
    def _curl_regularization(self, flow: torch.Tensor) -> torch.Tensor:
        """Apply curl regularization."""
        # Calculate curl
        curl = flow[:, 1:2, :, 1:] - flow[:, 1:2, :, :-1] - \
               flow[:, 0:1, 1:, :] + flow[:, 0:1, :-1, :]
        
        # Pad to match original size
        pad = torch.zeros_like(curl[:, :, :1, :])
        curl = torch.cat([curl, pad], dim=2)
        pad = torch.zeros_like(curl[:, :, :, :1])
        curl = torch.cat([curl, pad], dim=3)
        
        # Apply curl correction
        corrected_flow = flow - 0.01 * curl
        return corrected_flow


class FlowConfidenceModule(nn.Module):
    """Module for estimating confidence in flow estimates."""
    
    def __init__(self, in_channels: int, confidence_channels: int = 1):
        """Initialize flow confidence module.
        
        Args:
            in_channels: Number of input channels
            confidence_channels: Number of confidence channels
        """
        super().__init__()
        self.confidence_estimator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, confidence_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, flow: torch.Tensor, features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate flow confidence.
        
        Args:
            flow: Flow tensor
            features: Optional feature tensor
            
        Returns:
            Tuple of (flow, confidence)
        """
        if features is not None:
            x = torch.cat([flow, features], dim=1)
        else:
            x = flow
            
        confidence = self.confidence_estimator(x)
        return flow, confidence


class FlowUpsamplingModule(nn.Module):
    """Module for upsampling flow fields."""
    
    def __init__(self, upscale_factor: int = 2, mode: str = 'bilinear'):
        """Initialize flow upsampling module.
        
        Args:
            upscale_factor: Factor by which to upscale
            mode: Interpolation mode
        """
        super().__init__()
        self.upscale_factor = upscale_factor
        self.mode = mode
    
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """Upsample flow field.
        
        Args:
            flow: Flow tensor of shape (B, 2, H, W)
            
        Returns:
            Upsampled flow tensor
        """
        # Upsample flow
        upsampled = F.interpolate(flow, 
                                scale_factor=self.upscale_factor,
                                mode=self.mode, 
                                align_corners=False)
        
        # Scale flow vectors by upscale factor
        upsampled = upsampled * self.upscale_factor
        
        return upsampled