import numpy as np
import torch
# import cupy as cp

def compute_dsft_core(spike):
    H, W, T = spike.shape
    time = spike * torch.arange(T, device='cuda').reshape(1, 1, T)
    l_idx, _ = time.cummax(dim=2)
    time[time==0] = T
    r_idx, _ = torch.flip(time, [2]).cummin(dim=2)
    r_idx = torch.flip(r_idx, [2])
    r_idx = torch.cat([r_idx[:, :, 1:], torch.ones([H, W, 1], device='cuda') * T], dim=2)
    res = r_idx - l_idx
    
    res = torch.clip(res, 0)
    return res
