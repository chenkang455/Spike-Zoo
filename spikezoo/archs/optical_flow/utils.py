"""
Utility functions for optical flow in SpikeZoo.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from scipy.ndimage import gaussian_filter


def flow_to_image(flow: torch.Tensor, max_flow: Optional[float] = None) -> torch.Tensor:
    """Convert optical flow to RGB image for visualization.
    
    Args:
        flow: Flow tensor of shape (B, 2, H, W)
        max_flow: Maximum flow value for normalization (if None, computed from flow)
        
    Returns:
        RGB image tensor of shape (B, 3, H, W)
    """
    # Compute flow magnitude
    mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
    
    # Compute max flow if not provided
    if max_flow is None:
        max_flow = mag.max().item()
    
    # Normalize flow
    flow_norm = flow / (max_flow + 1e-8)
    
    # Convert to HSV-like representation
    hue = torch.atan2(flow_norm[:, 1], flow_norm[:, 0])
    hue = (hue + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
    
    sat = torch.clamp(mag / max_flow, 0, 1)
    val = torch.ones_like(sat)
    
    # Convert HSV to RGB (simplified)
    rgb = torch.stack([hue, sat, val], dim=1)
    
    return rgb


def flow_warp(image: torch.Tensor, flow: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
    """Warp image using optical flow.
    
    Args:
        image: Image tensor of shape (B, C, H, W)
        flow: Flow tensor of shape (B, 2, H, W)
        mode: Interpolation mode
        
    Returns:
        Warped image tensor
    """
    B, C, H, W = image.shape
    
    # Create meshgrid
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
    warped = F.grid_sample(image, grid, mode=mode, padding_mode='zeros', align_corners=True)
    return warped


def compute_flow_metrics(flow_pred: torch.Tensor, flow_gt: torch.Tensor) -> dict:
    """Compute optical flow metrics.
    
    Args:
        flow_pred: Predicted flow tensor of shape (B, 2, H, W)
        flow_gt: Ground truth flow tensor of shape (B, 2, H, W)
        
    Returns:
        Dictionary of metrics
    """
    # Endpoint error (EPE)
    epe = torch.norm(flow_pred - flow_gt, p=2, dim=1)
    epe_mean = epe.mean()
    
    # Angular error
    flow_pred_mag = torch.norm(flow_pred, p=2, dim=1)
    flow_gt_mag = torch.norm(flow_gt, p=2, dim=1)
    
    dot_product = (flow_pred[:, 0] * flow_gt[:, 0] + flow_pred[:, 1] * flow_gt[:, 1])
    cos_angle = dot_product / (flow_pred_mag * flow_gt_mag + 1e-8)
    angular_error = torch.acos(torch.clamp(cos_angle, -1, 1))
    angular_error_mean = torch.rad2deg(angular_error).mean()
    
    # Outliers (EPE > 3px or > 5% of flow magnitude)
    flow_mag = torch.norm(flow_gt, p=2, dim=1)
    outlier_mask = (epe > 3.0) & (epe > 0.05 * flow_mag)
    outlier_rate = outlier_mask.float().mean()
    
    return {
        'epe': epe_mean.item(),
        'angular_error': angular_error_mean.item(),
        'outlier_rate': outlier_rate.item()
    }


def resize_flow(flow: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """Resize optical flow field.
    
    Args:
        flow: Flow tensor of shape (B, 2, H, W)
        target_size: Target size (H, W)
        
    Returns:
        Resized flow tensor
    """
    B, _, H, W = flow.shape
    target_H, target_W = target_size
    
    # Resize flow
    resized = F.interpolate(flow, size=target_size, mode='bilinear', align_corners=False)
    
    # Scale flow vectors
    scale_x = target_W / W
    scale_y = target_H / H
    resized[:, 0] *= scale_x
    resized[:, 1] *= scale_y
    
    return resized


def flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    """Compute flow smoothness penalty.
    
    Args:
        flow: Flow tensor of shape (B, 2, H, W)
        
    Returns:
        Smoothness penalty tensor
    """
    # Calculate flow gradients
    flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    
    # Calculate smoothness penalty
    smoothness_penalty = torch.mean(flow_dx ** 2) + torch.mean(flow_dy ** 2)
    
    return smoothness_penalty


def flow_divergence(flow: torch.Tensor) -> torch.Tensor:
    """Compute flow divergence.
    
    Args:
        flow: Flow tensor of shape (B, 2, H, W)
        
    Returns:
        Divergence tensor of shape (B, 1, H, W)
    """
    # Calculate divergence
    div = flow[:, 0:1, :, 1:] - flow[:, 0:1, :, :-1] + \
          flow[:, 1:2, 1:, :] - flow[:, 1:2, :-1, :]
    
    # Pad to match original size
    pad = torch.zeros_like(div[:, :, :1, :])
    div = torch.cat([div, pad], dim=2)
    pad = torch.zeros_like(div[:, :, :, :1])
    div = torch.cat([div, pad], dim=3)
    
    return div


def flow_curl(flow: torch.Tensor) -> torch.Tensor:
    """Compute flow curl.
    
    Args:
        flow: Flow tensor of shape (B, 2, H, W)
        
    Returns:
        Curl tensor of shape (B, 1, H, W)
    """
    # Calculate curl
    curl = flow[:, 1:2, :, 1:] - flow[:, 1:2, :, :-1] - \
           flow[:, 0:1, 1:, :] + flow[:, 0:1, :-1, :]
    
    # Pad to match original size
    pad = torch.zeros_like(curl[:, :, :1, :])
    curl = torch.cat([curl, pad], dim=2)
    pad = torch.zeros_like(curl[:, :, :, :1])
    curl = torch.cat([curl, pad], dim=3)
    
    return curl


def generate_flow_pyramid(flow: torch.Tensor, num_levels: int = 3) -> List[torch.Tensor]:
    """Generate flow pyramid.
    
    Args:
        flow: Flow tensor of shape (B, 2, H, W)
        num_levels: Number of pyramid levels
        
    Returns:
        List of flow tensors at different scales
    """
    pyramid = [flow]
    
    for i in range(1, num_levels):
        scale_factor = 2 ** i
        scaled = F.interpolate(flow, 
                             scale_factor=1.0/scale_factor,
                             mode='bilinear', 
                             align_corners=False)
        # Scale flow vectors
        scaled = scaled / scale_factor
        pyramid.append(scaled)
    
    return pyramid


def flow_confidence(flow: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """Compute flow confidence based on local consistency.
    
    Args:
        flow: Flow tensor of shape (B, 2, H, W)
        window_size: Size of local window for consistency check
        
    Returns:
        Confidence tensor of shape (B, 1, H, W)
    """
    B, _, H, W = flow.shape
    pad = window_size // 2
    
    # Pad flow
    flow_padded = F.pad(flow, (pad, pad, pad, pad), mode='replicate')
    
    # Compute local variance
    local_var = torch.zeros(B, 1, H, W, device=flow.device)
    
    for i in range(window_size):
        for j in range(window_size):
            flow_patch = flow_padded[:, :, i:i+H, j:j+W]
            diff = flow_patch - flow
            local_var += diff ** 2
    
    local_var = local_var / (window_size ** 2)
    
    # Convert variance to confidence (lower variance = higher confidence)
    confidence = torch.exp(-local_var.mean(dim=1, keepdim=True))
    
    return confidence


def load_flow_from_file(filepath: str) -> torch.Tensor:
    """Load optical flow from file.
    
    Args:
        filepath: Path to flow file (.flo format)
        
    Returns:
        Flow tensor of shape (2, H, W)
    """
    # This is a simplified implementation
    # In practice, you would need to handle the .flo file format
    with open(filepath, 'rb') as f:
        # Read header
        tag = np.frombuffer(f.read(4), dtype=np.float32)[0]
        if tag != 202021.25:
            raise ValueError("Invalid .flo file")
        
        width = np.frombuffer(f.read(4), dtype=np.int32)[0]
        height = np.frombuffer(f.read(4), dtype=np.int32)[0]
        
        # Read flow data
        flow_data = np.frombuffer(f.read(), dtype=np.float32)
        flow_data = flow_data.reshape((height, width, 2))
        
        # Convert to tensor
        flow = torch.from_numpy(flow_data).permute(2, 0, 1).float()
        
    return flow


def save_flow_to_file(flow: torch.Tensor, filepath: str):
    """Save optical flow to file.
    
    Args:
        flow: Flow tensor of shape (2, H, W)
        filepath: Path to save flow file
    """
    # Convert to numpy
    flow_np = flow.permute(1, 2, 0).cpu().numpy()
    
    # Save in .flo format
    with open(filepath, 'wb') as f:
        # Write header
        f.write(np.array(202021.25, dtype=np.float32).tobytes())
        f.write(np.array(flow_np.shape[1], dtype=np.int32).tobytes())
        f.write(np.array(flow_np.shape[0], dtype=np.int32).tobytes())
        
        # Write flow data
        f.write(flow_np.astype(np.float32).tobytes())


def flow_to_point_cloud(flow: torch.Tensor, depth: torch.Tensor, 
                       intrinsics: torch.Tensor) -> torch.Tensor:
    """Convert optical flow to 3D point cloud using depth and camera intrinsics.
    
    Args:
        flow: Flow tensor of shape (B, 2, H, W)
        depth: Depth tensor of shape (B, 1, H, W)
        intrinsics: Camera intrinsics matrix of shape (B, 3, 3)
        
    Returns:
        Point cloud tensor of shape (B, 3, H, W)
    """
    B, _, H, W = flow.shape
    
    # Create pixel coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    x_coords = x_coords.float().to(flow.device)
    y_coords = y_coords.float().to(flow.device)
    
    # Stack coordinates
    coords = torch.stack([x_coords, y_coords], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    
    # Add flow to coordinates
    coords_warped = coords + flow
    
    # Convert to homogeneous coordinates
    ones = torch.ones(B, 1, H, W, device=flow.device)
    pixel_coords = torch.cat([coords_warped, ones], dim=1)
    
    # Apply inverse intrinsics
    inv_intrinsics = torch.inverse(intrinsics)
    normalized_coords = torch.bmm(inv_intrinsics, pixel_coords.view(B, 3, -1)).view(B, 3, H, W)
    
    # Scale by depth
    point_cloud = normalized_coords * depth
    
    return point_cloud