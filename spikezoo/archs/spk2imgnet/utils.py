import math

import numpy as np
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2.0 / 9.0 / 64.0)).clamp_(
            -0.025, 0.025
        )
        nn.init.constant(m.bias.data, 0.0)


def batch_psnr(img, imclean, data_range):
    img = img.data.cpu().numpy().astype(np.float32)
    imclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = peak_signal_noise_ratio(img, imclean, data_range=data_range)
    """
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(imclean[i,:,:,:], img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])
    """
    return psnr


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))
