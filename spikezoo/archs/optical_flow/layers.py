"""
Custom layers for optical flow networks in SpikeZoo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FlowRefinementLayer(nn.Module):
    """Layer for refining optical flow estimates."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """Initialize flow refinement layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, flow: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            flow: Input flow tensor
            features: Optional feature tensor to concatenate
            
        Returns:
            Refined flow tensor
        """
        if features is not None:
            x = torch.cat([flow, features], dim=1)
        else:
            x = flow
            
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class WarpingLayer(nn.Module):
    """Layer for warping images/features using optical flow."""
    
    def __init__(self, interpolation_mode: str = 'bilinear'):
        """Initialize warping layer.
        
        Args:
            interpolation_mode: Interpolation mode for grid sampling
        """
        super().__init__()
        self.interpolation_mode = interpolation_mode
    
    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp image using optical flow.
        
        Args:
            image: Input image tensor of shape (B, C, H, W)
            flow: Optical flow tensor of shape (B, 2, H, W)
            
        Returns:
            Warped image tensor
        """
        # Create meshgrid
        B, C, H, W = image.shape
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float().to(image.device)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float().to(image.device)
        
        # Add flow to grid
        grid = torch.cat([xx, yy], dim=1) + flow
        
        # Normalize grid to [-1, 1]
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        
        # Transpose grid
        grid = grid.permute(0, 2, 3, 1)
        
        # Warp image
        warped = F.grid_sample(image, grid, mode=self.interpolation_mode, padding_mode='zeros')
        return warped


class FlowRegularizationLayer(nn.Module):
    """Layer for regularizing optical flow fields."""
    
    def __init__(self, smoothness_weight: float = 0.01):
        """Initialize flow regularization layer.
        
        Args:
            smoothness_weight: Weight for smoothness penalty
        """
        super().__init__()
        self.smoothness_weight = smoothness_weight
    
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """Apply flow regularization.
        
        Args:
            flow: Input flow tensor of shape (B, 2, H, W)
            
        Returns:
            Regularized flow tensor
        """
        # Calculate flow gradients
        flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        
        # Calculate smoothness penalty
        smoothness_penalty = torch.mean(flow_dx ** 2) + torch.mean(flow_dy ** 2)
        
        # Apply regularization (this is typically used in loss computation rather than forward pass)
        # For forward pass, we just return the flow unchanged
        return flow


class MultiScaleFlowFusion(nn.Module):
    """Layer for fusing optical flow at multiple scales."""
    
    def __init__(self, num_scales: int, fusion_method: str = 'upsample'):
        """Initialize multi-scale flow fusion layer.
        
        Args:
            num_scales: Number of scales to fuse
            fusion_method: Method for fusion ('upsample' or 'concat')
        """
        super().__init__()
        self.num_scales = num_scales
        self.fusion_method = fusion_method
        
        if fusion_method == 'upsample':
            self.upsamplers = nn.ModuleList([
                nn.ConvTranspose2d(2, 2, kernel_size=2**(i+1), stride=2**(i+1), padding=0)
                for i in range(num_scales - 1)
            ])
    
    def forward(self, flows: list) -> torch.Tensor:
        """Fuse flows from multiple scales.
        
        Args:
            flows: List of flow tensors from coarse to fine scale
            
        Returns:
            Fused flow tensor
        """
        if self.fusion_method == 'upsample':
            # Upsample coarser flows and add to finer flows
            fused_flow = flows[-1]  # Start with finest flow
            for i in range(len(flows) - 2, -1, -1):
                upsampled = self.upsamplers[len(flows) - 2 - i](flows[i])
                # Ensure dimensions match
                if upsampled.shape[2:] != fused_flow.shape[2:]:
                    # Crop if necessary
                    upsampled = upsampled[:, :, :fused_flow.shape[2], :fused_flow.shape[3]]
                fused_flow = fused_flow + upsampled
            return fused_flow
        else:
            # Simple concatenation approach
            return torch.cat(flows, dim=1)


class EventFeatureExtractor(nn.Module):
    """Feature extractor for event-based data."""
    
    def __init__(self, input_channels: int = 2, base_channels: int = 32, num_layers: int = 3):
        """Initialize event feature extractor.
        
        Args:
            input_channels: Number of input channels
            base_channels: Base number of channels
            num_layers: Number of layers
        """
        super().__init__()
        self.input_channels = input_channels
        self.base_channels = base_channels
        
        # Create feature extraction layers
        layers = []
        in_ch = input_channels
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            in_ch = out_ch
        
        self.features = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from event data.
        
        Args:
            x: Input tensor of shape (B, input_channels, H, W)
            
        Returns:
            Feature tensor
        """
        return self.features(x)


class FlowConfidenceLayer(nn.Module):
    """Layer for estimating confidence in optical flow estimates."""
    
    def __init__(self, in_channels: int, confidence_channels: int = 1):
        """Initialize flow confidence layer.
        
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
            flow: Input flow tensor
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