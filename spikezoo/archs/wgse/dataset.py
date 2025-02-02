import glob
import os

import numpy as np
import torch
import torch.utils.data as data
from skimage import io

import utils


class DatasetREDS(data.Dataset):
    def __init__(self, cfg, transform=None):
        super(DatasetREDS, self).__init__()
        self.cfg = cfg
        self.rootfolder = cfg['rootfolder']
        self.spikefolder = os.path.join(self.rootfolder, cfg['spikefolder'])
        self.imagefolder = os.path.join(self.rootfolder, cfg['imagefolder'])
        
        self.spike_list = os.listdir(self.spikefolder)
        self.image_list = os.listdir(self.imagefolder)
        
        self.H = int(cfg['H'])
        self.W = int(cfg['W'])
        self.C_SPIKE = int(cfg['C'])

        self.transform = transform

    def __getitem__(self, index: int):
        item_name = self.spike_list[index][:-4]
        spike_path = os.path.join(self.spikefolder, item_name+'.dat')
        image_path = os.path.join(self.imagefolder, item_name+'.png')

        path_grayframe = glob.glob(image_path)
        gray_frame = io.imread(path_grayframe[0], as_gray=False).astype(np.float32)

        gray_frame /= 255.0  # normalize
        gray_frame = np.expand_dims(gray_frame, axis=0)  # expand to [1 x H x W]
        gray_frame = torch.from_numpy(gray_frame)

        f = open(spike_path, 'rb')
        spike_seq = f.read()
        spike_seq = np.frombuffer(spike_seq, 'b')
        spikes = utils.RawToSpike(spike_seq, self.H, self.W)
        spikes = spikes.astype(np.float32)
        spikes = torch.from_numpy(spikes)
        f.close()

        if self.transform:
            gray_frame, spikes = self.transform(gray_frame, spikes)

        item = {}
        item['spikes'] = spikes
        item['image'] = gray_frame
        
        return item

    def __len__(self) -> int:
        return len(self.spike_list)
