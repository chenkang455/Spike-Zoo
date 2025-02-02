import numpy as np
import torch
import torch.nn.functional as F
import os
import os.path as osp
import random
import cv2

def set_seeds(_seed_):
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance) or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
    # torch.use_deterministic_algorithms(True)


def make_dir(path):
    if not osp.exists(path):
        os.makedirs(path)
    return


def add_args_to_cfg(cfg, args, args_list):
    for aa in args_list:
        cfg['train'][aa] = eval('args.{:s}'.format(aa))
    return cfg


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self, precision=3):
#         self.precision = precision
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

#     def __repr__(self):
#         return '{:.{}f} ({:.{}f})'.format(self.val, self.precision, self.avg, self.precision)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3, names=None):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)
        self.names = names
        if names is not None:
            assert self.meters == len(self.names)
        else:
            self.names = [''] * self.meters

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = [0] * i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        if not isinstance(n, list):
            n = [n] * self.meters
        assert (len(val) == self.meters and len(n) == self.meters)
        for i in range(self.meters):
            self.count[i] += n[i]
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n[i]
            self.avg[i] = self.sum[i] / self.count[i]

    def __repr__(self):
        # val = ' '.join(['{} {:.{}f}'.format(n, v, self.precision) for n, v in
        #                 zip(self.names, self.val)])
        # avg = ' '.join(['{} {:.{}f}'.format(n, a, self.precision) for n, a in
        #                 zip(self.names, self.avg)])
        out = ' '.join(['{} {:.{}f} ({:.{}f})'.format(n, v, self.precision, a, self.precision) for n, v, a in
                        zip(self.names, self.val, self.avg)])
        # return '{} ({})'.format(val, avg)
        return '{}'.format(out)


def normalize_image_torch(image, percentile_lower=1, percentile_upper=99):
    b, c, h, w = image.shape
    image_reshape = image.reshape([b, c, h*w])
    mini = torch.quantile(image_reshape, 0.01, dim=2, keepdim=True).unsqueeze_(dim=3)
    maxi = torch.quantile(image_reshape, 0.99, dim=2, keepdim=True).unsqueeze_(dim=3)
    # if mini == maxi:
    #     return 0 * image + 0.5  # gray image
    return torch.clip((image - mini) / (maxi - mini + 1e-5), 0, 1)

def normalize_image_torch2(image):
    return torch.clip(image, 0, 1)

# --------------------------------------------
# Torch to Numpy 0~255
# --------------------------------------------
def torch2numpy255(im):
    im = im[0, 0].detach().cpu().numpy()
    im = (im * 255).astype(np.float64)
    return im

def torch2torch255(im):
    return im * 255.0

class InputPadder:
    """ Pads images such that dimensions are divisible by padsize """
    def __init__(self, dims, padsize=16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padsize) + 1) * padsize - self.ht) % padsize
        pad_wd = (((self.wd // padsize) + 1) * padsize - self.wd) % padsize
        #self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    


def vis_img(vis_path: str, img: torch.Tensor, vis_name: str = 'vis'):
    ww = 0
    rows = []
    for ii in range(4):
        cur_row = []
        for jj in range(img.shape[0]//4):
            cur_img = img[ww, 0].detach().cpu().numpy() * 255
            cur_img = cur_img.astype(np.uint8)
            cur_row.append(cur_img)
            ww += 1
        cur_row_cat = np.concatenate(cur_row, axis=1)
        rows.append(cur_row_cat)
    out_img = np.concatenate(rows, axis=0)
    cv2.imwrite(osp.join(vis_path, vis_name+'.png'), out_img)
    return
