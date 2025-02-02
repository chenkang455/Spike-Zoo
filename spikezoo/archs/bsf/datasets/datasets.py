import os
import os.path as osp
import random
import numpy as np
import torch
import torch.utils.data as data
from datasets.ds_utils import *
import h5py
from tqdm import *


class Augmentor:
    def __init__(self, crop_size):
        # spatial augmentation params
        self.crop_size = crop_size

    def augment_img(self, img, mode=0):
        '''Kai Zhang (github: https://github.com/cszn)
        W x H x C or W x H
        注:要使用此种augmentation, 则需保证crop_h = crop_w
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
        spike_h = spk_list[0].shape[1]
        spike_w = spk_list[0].shape[2]

        if spike_h > self.crop_size[0]:
            y0 = np.random.randint(0, spike_h - self.crop_size[0])
        else:
            y0 = 0
        
        if spike_w > self.crop_size[1]:
            x0 = np.random.randint(0, spike_w - self.crop_size[1])
        else:
            x0 = 0

        for ii, spk in enumerate(spk_list):
            spk = np.transpose(spk, [1,2,0])
            spk = spk[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1], :]
            spk = self.augment_img(spk, mode=mode)
            spk_list[ii] = np.transpose(spk, [2,0,1])

        for ii, img in enumerate(img_list):
            img = np.transpose(img, [1,2,0])
            img = img[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1], :]
            img = self.augment_img(img, mode=mode)
            img_list[ii] = np.transpose(img, [2,0,1])

        return spk_list, img_list

    def __call__(self, spk_list, img_list):
        spk_list, img_list = self.spatial_transform(spk_list, img_list)
        spk_list = [np.ascontiguousarray(spk) for spk in spk_list]
        img_list = [np.ascontiguousarray(img) for img in img_list]
        return spk_list, img_list



class sreds_train(torch.utils.data.Dataset):
    '''
    测试集Spike原始分辨率 148 x 256
    '''
    def __init__(self, args):
        self.args = args
        self.input_type = args.input_type
        self.eta_list = args.eta_list
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.augmentor = Augmentor(crop_size=args.train_res)
        
        self.dsft_path_name = 'dsft'
        self.spike_path_name = 'spikes'

        self.read_dsft = not args.no_dsft

        self.samples = self.collect_samples()
        print('The samples num of training data: {:d}'.format(len(self.samples)))
    
    def confirm_exist(self, path_list_list):
        for pl in path_list_list:
            for p in pl:
                if not osp.exists(p):
                    return 0
        return 1
    
    def collect_samples(self):
        samples = []
        root_path = osp.join(self.args.data_root, 'crop', 'train')
        
        for eta in self.eta_list:
            cur_eta_dir = osp.join(root_path, "eta_{:.2f}_gamma_{:d}_alpha_{:.1f}".format(eta, self.gamma, self.alpha))
            scene_list = sorted(os.listdir(cur_eta_dir))

            for scene in scene_list:
                scene_path = osp.join(cur_eta_dir, scene)
                crop_list = sorted(os.listdir(scene_path))
                for crop in crop_list:
                    crop_path = osp.join(scene_path, crop)
                    spike_dir = osp.join(crop_path, self.spike_path_name)
                    image_dir = osp.join(root_path, 'imgs', scene, crop)
                    dsft_dir = osp.join(crop_path, 'dsft')

                    ## 数据集的制作：dsft 从 09~20.h5, img从 10~19.png
                    spikes_path_list = [osp.join(spike_dir, '{:08d}.dat'.format(ii)) for ii in range(11, 28+1)]
                    dsft_path_list = [osp.join(dsft_dir, '{:08d}.h5'.format(ii)) for ii in range(11, 28+1)]
                    images00_path_list = [osp.join(image_dir, '{:08d}.png'.format(ii)) for ii in range(18, 21+1)]
                    # images05_path_list = [osp.join(image_dir, '{:08d}_05.png'.format(ii)) for ii in range(8, 11+1)]

                    if(self.confirm_exist([spikes_path_list, images00_path_list])):
                        s = {}
                        s['spikes_paths'] = spikes_path_list
                        s['dsft_paths'] = dsft_path_list
                        s['images_paths'] = images00_path_list
                        s['norm_fac'] = eta * self.alpha
                        # s['images_05_paths'] = images05_path_list
                        samples.append(s)
        return samples

    def _load_sample(self, s):
        ## 一组数据中有4个时间点可以做key-frame，抽其中一个作为一次采样
        ## images只有四个，分别是18, 19, 20, 21，直接对应于offset的1,2,3,4
        ## spikes和dsfts都比较多，所使用的key是{18, 19, 20, 21}，也即对应于spike和dsft的path list中的{7,8,9,10}index
        key_frame_offset = random.choice([0,1,2,3])
        s['spikes_paths'] = s['spikes_paths'][7+key_frame_offset-3-self.args.half_reserve : 7+key_frame_offset+3+self.args.half_reserve+1]
        s['dsft_paths'] = s['dsft_paths'][7+key_frame_offset-3-self.args.half_reserve : 7+key_frame_offset+3+self.args.half_reserve+1]

        ## 第一个Key是13.dat, imgs从10开始，应该是 key_frame_offset+3-2
        s['images_paths'] = [s['images_paths'][key_frame_offset]]

        data = {}
        if self.read_dsft:
            ## 读入Spike
            h5files = [h5py.File(p, 'r') for p in s['dsft_paths']]
            data['dsft'] = [np.array(f['dsft']).astype(np.float32) for f in h5files]
            for f in h5files:
                f.close()
        data['spikes'] = [dat_to_spmat(p, size=(256, 256)).astype(np.float32) for p in s['spikes_paths']]

        ## 读入 Image
        data['images'] = [read_img_gray(p) for p in s['images_paths']]
        data['norm_fac'] = np.array(s['norm_fac'])

        if self.read_dsft:
            data['spikes'] = data['spikes'] + data['dsft']
            data['spikes'], data['images'] = self.augmentor(data['spikes'], data['images'])
            data['spikes'], data['dsft'] = data['spikes'][:len(data['spikes'])//2], data['spikes'][len(data['spikes'])//2:]
        else:
            data['spikes'], data['images'] = self.augmentor(data['spikes'], data['images'])
        
        return data
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data


class sreds_test(torch.utils.data.Dataset):
    '''
    测试集Spike原始分辨率 540 x 960
    '''
    def __init__(self, args, eta):
        self.args = args
        self.input_type = args.input_type
        self.alpha = args.alpha
        self.eta = eta
        self.gamma = args.gamma
        self.dsft_path_name = 'dsft'
        self.spike_path_name = 'spikes'
        self.samples = self.collect_samples()
        print('The samples num of testing data: {:d}'.format(len(self.samples)))
    
    def confirm_exist(self, path_list_list):
        for pl in path_list_list:
            for p in pl:
                if not osp.exists(p):
                    print(p)
                    return 0
        return 1
    
    def collect_samples(self):
        root_path = osp.join(self.args.data_root, 'crop', 'val')

        cur_eta_dir = osp.join(root_path, "eta_{:.2f}_gamma_{:d}_alpha_{:.1f}".format(self.eta, self.gamma, self.alpha))
        scene_list = sorted(os.listdir(cur_eta_dir))
        samples = []

        for scene in scene_list:
            scene_path = osp.join(cur_eta_dir, scene)
            spike_dir = osp.join(scene_path, self.spike_path_name)
            image_dir = osp.join(root_path, 'imgs', scene)
            dsft_dir = osp.join(scene_path, 'dsft')

            ## 数据集的制作：dsft 从 09~20.h5, img从 10~19.png
            spikes_path_list = [osp.join(spike_dir, '{:08d}.dat'.format(ii)) for ii in range(11, 28+1)]
            dsft_path_list = [osp.join(dsft_dir, '{:08d}.h5'.format(ii)) for ii in range(11, 28+1)]
            images_path_list = [osp.join(image_dir, '{:08d}.png'.format(ii)) for ii in range(18, 21+1)]

            if(self.confirm_exist([spikes_path_list, images_path_list])):
            ## 在test函数里测试四组数据
            ## images只有四个，分别是18, 19, 20, 21，直接对应于offset的1,2,3,4
            ## spikes和dsfts都比较多，所使用的key是{18, 19, 20, 21}，也即对应于spike和dsft的path list中的{7,8,9,10}index
                for ii in range(4):
                # for ii in range(1):
                    s = {}
                    s['spikes_paths'] = spikes_path_list[7+ii-3-self.args.half_reserve : 7+ii+3+self.args.half_reserve+1]
                    s['dsft_paths'] = dsft_path_list[7+ii-3-self.args.half_reserve : 7+ii+3+self.args.half_reserve+1]
                    s['images_paths'] = [images_path_list[ii]]
                    s['norm_fac'] = self.alpha * self.eta
                    samples.append(s)

        return samples

    def _load_sample(self, s):
        data = {}
        h5files = [h5py.File(p, 'r') for p in s['dsft_paths']]
        data['dsft'] = [np.array(f['dsft']).astype(np.float32) for f in h5files]
        for f in h5files:
            f.close()
        data['spikes'] = [dat_to_spmat(p, size=(540, 960)).astype(np.float32) for p in s['spikes_paths']]

        data['images'] = [read_img_gray(p) for p in s['images_paths']]
        data['norm_fac'] = np.array(s['norm_fac'])
        return data
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data

class sreds_test_small(torch.utils.data.Dataset):
    '''
    测试集Spike原始分辨率 384 x 512
    '''
    def __init__(self, args, eta):
        self.args = args
        self.input_type = args.input_type
        self.alpha = args.alpha
        self.eta = eta
        self.gamma = args.gamma
        self.dsft_path_name = 'dsft'
        self.spike_path_name = 'spikes'

        self.read_dsft = not args.no_dsft
        self.samples = self.collect_samples()
        print('The samples num of testing data: {:d}'.format(len(self.samples)))
    
    def confirm_exist(self, path_list_list):
        for pl in path_list_list:
            for p in pl:
                if not osp.exists(p):
                    print(p)
                    return 0
        return 1
    
    def collect_samples(self):
        root_path = osp.join(self.args.data_root, 'crop', 'val_small')

        cur_eta_dir = osp.join(root_path, "eta_{:.2f}_gamma_{:d}_alpha_{:.1f}".format(self.eta, self.gamma, self.alpha))
        scene_list = sorted(os.listdir(cur_eta_dir))
        samples = []

        for scene in scene_list:
            scene_path = osp.join(cur_eta_dir, scene)
            spike_dir = osp.join(scene_path, self.spike_path_name)
            image_dir = osp.join(root_path, 'imgs', scene)
            dsft_dir = osp.join(scene_path, 'dsft')

            ## 数据集的制作：dsft 从 09~20.h5, img从 10~19.png
            spikes_path_list = [osp.join(spike_dir, '{:08d}.dat'.format(ii)) for ii in range(11, 28+1)]
            dsft_path_list = [osp.join(dsft_dir, '{:08d}.h5'.format(ii)) for ii in range(11, 28+1)]
            images_path_list = [osp.join(image_dir, '{:08d}.png'.format(ii)) for ii in range(18, 21+1)]

            if(self.confirm_exist([spikes_path_list, images_path_list])):
                # for ii in range(4):
                for ii in range(4):
                    s = {}
                    s['spikes_paths'] = spikes_path_list[7+ii-3-self.args.half_reserve : 7+ii+3+self.args.half_reserve+1]
                    s['dsft_paths'] = dsft_path_list[7+ii-3-self.args.half_reserve : 7+ii+3+self.args.half_reserve+1]
                    s['images_paths'] = [images_path_list[ii]]
                    s['norm_fac'] = self.alpha * self.eta
                    samples.append(s)

        return samples

    def _load_sample(self, s):
        ## 在test函数里测试四组数据
        ## spikes全取
        ## image取四个key对应的[13, 14, 15, 16]
        data = {}
        if self.read_dsft:
            h5files = [h5py.File(p, 'r') for p in s['dsft_paths']]
            data['dsft'] = [np.array(f['dsft']).astype(np.float32) for f in h5files]
            for f in h5files:
                f.close()
        data['spikes'] = [dat_to_spmat(p, size=(384, 512)).astype(np.float32) for p in s['spikes_paths']]

        data['images'] = [read_img_gray(p) for p in s['images_paths']]
        data['norm_fac'] = np.array(s['norm_fac'])
        return data
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data
