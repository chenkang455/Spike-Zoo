import os
import os.path as osp
import random
import numpy as np
import torch
import torch.utils.data as data
from datasets.ds_utils import *
import time


class Augmentor:
    def __init__(self, crop_size):
        # spatial augmentation params
        self.crop_size = crop_size

    def augment_img(self, img, mode=0):
        '''Kai Zhang (github: https://github.com/cszn)
        W x H x C or W x H
        '''
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(np.rot90(img))
        elif mode == 2:
            return np.flipud(img)
        elif mode == 3:
            return np.rot90(img, k=3)
        elif mode == 4:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 5:
            return np.rot90(img)
        elif mode == 6:
            return np.rot90(img, k=2)
        elif mode == 7:
            return np.flipud(np.rot90(img, k=3))

    def spatial_transform(self, spk_list, img_list):
        mode = random.randint(0, 7)

        for ii, spk in enumerate(spk_list):
            spk = np.transpose(spk, [1,2,0])
            spk = self.augment_img(spk, mode=mode)
            spk_list[ii] = np.transpose(spk, [2,0,1])

        for ii, img in enumerate(img_list):
            img = np.transpose(img, [1,2,0])
            img = self.augment_img(img, mode=mode)
            img_list[ii] = np.transpose(img, [2,0,1])

        return spk_list, img_list

    def __call__(self, spk_list, img_list):
        spk_list, img_list = self.spatial_transform(spk_list, img_list)
        spk_list = [np.ascontiguousarray(spk) for spk in spk_list]
        img_list = [np.ascontiguousarray(img) for img in img_list]
        return spk_list, img_list


class sreds_train(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pair_step = self.cfg['loader']['pair_step']
        self.augmentor = Augmentor(crop_size=self.cfg['loader']['crop_size'])
        self.samples = self.collect_samples()
        print('The samples num of training data: {:d}'.format(len(self.samples)))
    
    def confirm_exist(self, path_list_list):
        for pl in path_list_list:
            for p in pl:
                if not osp.exists(p):
                    return 0
        return 1
    
    def collect_samples(self):
        spike_path = osp.join(self.cfg['data']['root'], 'crop_mini', 'spike', 'train', 'interp_{:d}_alpha_{:.2f}'.format(self.cfg['data']['interp'], self.cfg['data']['alpha']))
        image_path = osp.join(self.cfg['data']['root'], 'crop_mini', 'image', 'train', 'train_orig')
        scene_list = sorted(os.listdir(spike_path))
        samples = []

        for scene in scene_list:
            spike_dir = osp.join(spike_path, scene)
            image_dir = osp.join(image_path, scene)
            spk_path_list = sorted(os.listdir(spike_dir))
            
            spklen = len(spk_path_list)
            seq_len = self.cfg['model']['seq_len'] + 2

            for st in range(0, spklen - ((spklen - self.pair_step) % seq_len) - seq_len, self.pair_step):
                # 按照文件名称读取
                spikes_path_list = [osp.join(spike_dir, spk_path_list[ii]) for ii in range(st, st+seq_len)]
                images_path_list = [osp.join(image_dir, spk_path_list[ii][:-4]+'.png') for ii in range(st, st+seq_len)]

                if(self.confirm_exist([spikes_path_list, images_path_list])):
                    s = {}
                    s['spikes_paths'] = spikes_path_list
                    s['images_paths'] = images_path_list
                    samples.append(s)
        return samples

    def _load_sample(self, s):
        data = {}

        data['spikes'] = [np.array(dat_to_spmat(p, size=(96, 96)), dtype=np.float32) for p in s['spikes_paths']]
        data['images'] = [read_img_gray(p) for p in s['images_paths']]

        data['spikes'], data['images'] = self.augmentor(data['spikes'], data['images'])

        # print("data['spikes'][0].shape, data['images'][0].shape", data['spikes'][0].shape, data['images'][0].shape)

        return data
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data


class sreds_test(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = self.collect_samples()
        print('The samples num of testing data: {:d}'.format(len(self.samples)))
    
    def confirm_exist(self, path_list_list):
        for pl in path_list_list:
            for p in pl:
                if not osp.exists(p):
                    return 0
        return 1
    
    def collect_samples(self):
        spike_path = osp.join(self.cfg['data']['root'], 'spike', 'val', 'interp_{:d}_alpha_{:.2f}'.format(self.cfg['data']['interp'], self.cfg['data']['alpha']))
        image_path = osp.join(self.cfg['data']['root'], 'imgs', 'val', 'val_orig')
        scene_list = sorted(os.listdir(spike_path))
        samples = []

        for scene in scene_list:
            spike_dir = osp.join(spike_path, scene)
            image_dir = osp.join(image_path, scene)
            spk_path_list = sorted(os.listdir(spike_dir))
            
            spklen = len(spk_path_list)
            # seq_len = self.cfg['model']['seq_len']

            # 按照文件名称读取
            spikes_path_list = [osp.join(spike_dir, spk_path_list[ii]) for ii in range(spklen)]
            images_path_list = [osp.join(image_dir, spk_path_list[ii][:-4]+'.png') for ii in range(spklen)]

            if(self.confirm_exist([spikes_path_list, images_path_list])):
                s = {}
                s['spikes_paths'] = spikes_path_list
                s['images_paths'] = images_path_list
                samples.append(s)

        return samples

    def _load_sample(self, s):
        data = {}
        data['spikes'] = [np.array(dat_to_spmat(p, size=(720, 1280)), dtype=np.float32) for p in s['spikes_paths']]
        data['images'] = [read_img_gray(p) for p in s['images_paths']]
        return data
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data