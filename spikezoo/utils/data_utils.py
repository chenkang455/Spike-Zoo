import queue as Queue
import threading
import torch
from torch.utils.data import DataLoader
import math
import random

class Augmentor:
    def __init__(self, crop_size = (-1,-1)):
        self.crop_size = crop_size
         
    def augment(self, img, mode=0):
        mode = mode - mode % 2 if self.use_rot == False else mode
        if mode == 0:
            return img
        elif mode == 1:
            return torch.flip(torch.rot90(img, 1, [1, 2]), [1])  # flipud + rot90(k=1)
        elif mode == 2:
            return torch.flip(img, [1])  # flipud
        elif mode == 3:
            return torch.rot90(img, 3, [1, 2])  # rot90(k=3)
        elif mode == 4:
            return torch.flip(torch.rot90(img, 2, [1, 2]), [1])  # flipud + rot90(k=2)
        elif mode == 5:
            return torch.rot90(img, 1, [1, 2])  # rot90(k=1)
        elif mode == 6:
            return torch.rot90(img, 2, [1, 2])  # rot90(k=2)
        elif mode == 7:
            return torch.flip(torch.rot90(img, 3, [1, 2]), [1])  # flipud + rot90(k=3)
        
    def spatial_transform(self, spike, image):
        mode = random.randint(0, 7)
        spike_h = spike.shape[1]
        spike_w = spike.shape[2]
        # default mode
        if self.crop_size != (-1,-1):
            assert spike_h > self.crop_size[0] and spike_w > self.crop_size[1], f"ROI Size should be smaller than spike input size."
            y0 = random.randint(0, spike_h - self.crop_size[0])
            x0 = random.randint(0, spike_w - self.crop_size[1])
            spike = spike[:,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            image = image[:,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        # rotation set
        self.use_rot = True if image.shape[1] == image.shape[2] else False
        # aug
        spike = self.augment(spike, mode=mode)
        image = self.augment(image, mode=mode)
        return spike, image

    def __call__(self, spike, image):
        spike, image = self.spatial_transform(spike, image)
        return spike, image

