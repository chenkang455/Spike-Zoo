import torch
import torch.nn.functional as F
from utils import InputPadder


def compute_l1_loss(img_list, gt):
    l1_loss = 0.0
    for img in img_list:
        cur_size = img.shape[-2:]
        gt_resize = F.interpolate(gt, size=cur_size, mode="bilinear", align_corners=False)
        l1_loss += (img - gt_resize).abs().mean()

    return l1_loss


def compute_per_loss_single(img, gt, vgg):
    img_relu5_1 = vgg((img.repeat([1,3,1,1]) + 1.) / 2.)
    with torch.no_grad():
        gt_relu5_1 = vgg((gt.repeat([1,3,1,1]).detach() + 1.) / 2.)
    percep_loss = F.mse_loss(img_relu5_1, gt_relu5_1)
    return percep_loss
