from skimage import metrics
import torch
import torch.hub
from lpips.lpips import LPIPS
import os
import os
import pyiqa
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

# todo with the union type
metric_pair_names = ["psnr", "ssim", "lpips", "mse"]
metric_single_names = ["niqe", "brisque", "piqe", "liqe_mix", "clipiqa"]
metric_all_names = metric_pair_names + metric_single_names

metric_single_list = {}

metric_pair_list = {
    "mse": metrics.mean_squared_error,
    "ssim": metrics.structural_similarity,
    "psnr": metrics.peak_signal_noise_ratio,
    "lpips": None,
}


def cal_metric_single(img: torch.Tensor, metric_name="niqe"):
    if metric_name not in metric_single_list.keys():
        if metric_name in pyiqa.list_models():
            iqa_metric = pyiqa.create_metric(metric_name, device=torch.device("cuda"))
            metric_single_list.update({metric_name: iqa_metric})
        else:
            raise RuntimeError(f"Metric {metric_name} not recognized by the IQA lib.")
    # image process
    if img.dim() == 3:
        img = img[None]
    elif img.dim() == 2:
        img = img[None, None]

    # resize
    if metric_name == "liqe_mix":
        short_edge = 384
        h, w = img.shape[2], img.shape[3]
        if h < w:
            new_h, new_w = short_edge, int(w * short_edge / h)
        else:
            new_h, new_w = int(h * short_edge / w), short_edge
        img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return metric_single_list[metric_name](img).item()


def cal_metric_pair(im1t: torch.Tensor, im2t: torch.Tensor, metric_name="mse"):
    """
    im1t, im2t: torch.tensors with batched imaged shape, range from (0, 1)
    """
    if metric_name not in metric_pair_list.keys():
        raise RuntimeError(f"Metric {metric_name} not recognized")
    if metric_name == "lpips" and metric_pair_list[metric_name] is None:
        metric_pair_list[metric_name] = LPIPS().cuda() if im1t.is_cuda else LPIPS().cpu()
    metric_method = metric_pair_list[metric_name]

    # convert from [0, 1] to [-1, 1]
    im1t = (im1t * 2 - 1).clamp(-1, 1)
    im2t = (im2t * 2 - 1).clamp(-1, 1)

    # [c,h,w] -> [1,c,h,w]
    if im1t.dim() == 3:
        im1t = im1t[None]
        im2t = im2t[None]
    elif im1t.dim() == 2:
        im1t = im1t[None, None]
        im2t = im2t[None, None]

    # [1,h,w,3] -> [1,3,h,w]
    if im1t.shape[-1] == 3:
        im1t = im1t.permute(0, 3, 1, 2)
        im2t = im2t.permute(0, 3, 1, 2)

    # img array: [1,h,w,3] imgt tensor: [1,3,h,w]
    im1 = im1t.permute(0, 2, 3, 1).detach().cpu().numpy()
    im2 = im2t.permute(0, 2, 3, 1).detach().cpu().numpy()
    batchsz, hei, wid, _ = im1.shape

    # batch processing
    values = []
    for i in range(batchsz):
        if metric_name in ["mse", "psnr"]:
            value = metric_method(im1[i], im2[i])
        elif metric_name in ["ssim"]:
            value, ssimmap = metric_method(im1[i], im2[i], channel_axis=-1, data_range=2, full=True)
        elif metric_name in ["lpips"]:
            value = metric_method(im1t[i : i + 1], im2t[i : i + 1])[0, 0, 0, 0]
            value = value.detach().cpu().float().item()
        values.append(value)

    return sum(values) / len(values)
