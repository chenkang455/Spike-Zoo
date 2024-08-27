import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config as cfg
from dataset import *
from utils import load_spike_numpy


def generate_file_list(tr_or_ev):
    scene_name = os.listdir(cfg.data_root[0 if tr_or_ev == "train" else 1])
    random_index = np.random.permutation(len(scene_name))
    data_random_list = []
    for i, idx in enumerate(random_index):
        data_random_list.append(scene_name[idx])
    return data_random_list


def read_img(img_name, xx, yy):
    raw = cv2.imread(img_name, -1)
    raw_full = raw
    raw_patch = raw_full[yy:yy + cfg.image_height * 2,
                   xx:xx + cfg.image_width * 2]  # 256 * 256
    raw_pack_data = pack_gbrg_raw(raw_patch)
    return raw_pack_data


def decode_data(data_name, tr_or_ev):
    frame_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    H = 250
    W = 400
    # print((W - cfg.image_width + 1) / 2)
    # xx = np.random.randint(0, (W - cfg.image_width + 1) / 2)
    # yy = np.random.randint(0, (H - cfg.image_height + 1) / 2)
    xx = 0
    yy = 0

    scene_ind = data_name[:-15]
    frame_ind = int(data_name[-15])
    lightscale = data_name[-8:-4]
    # print(lightscale)
    # iso_ind = data_name.split('/')[0]

    # noisy_level_ind = iso_list.index(int(iso_ind[3:]))
    # noisy_level = 1, 1

    gt_img_list = []
    noisy_img_list = []
    data_name = os.path.join(cfg.data_root[0 if tr_or_ev == "train" else 1], data_name)
    # data_name = os.path.join(cfg.data_root[0 if tr_or_ev == "train" else 1], scene_ind + str(frame_list[frame_ind]) + "-light0032.npz")
    seq, tag, length = load_spike_numpy(data_name)
    frames = cfg.frame_num

    if lightscale == "0032":
        wins = 32
    elif lightscale == "0256":
        wins = 16

    for i in range(0, frames):
        # data_name = os.path.join(cfg.data_root[0 if tr_or_ev == "train" else 1], scene_ind + str(frame_list[frame_ind]) + "-light0256.npz")
        # data = np.load(data_name)
        # seq = data['seq']
        # length = data['length']
        # tag = data['tag']
        # spike_arr = np.zeros([length, cfg.image_height, cfg.image_width], dtype=np.uint8)
        # for j in range(length):
        #     spike_arr[j] = seq[int(j / 8), yy:yy + cfg.image_height, xx:xx + cfg.image_width] & (j % 8)
        # noisy_img = np.zeros([H, W], dtype=np.float32)
        # if lightscale == "-light0032.npz":
        #     noisy_img = seq[:256].mean(axis=0).astype(np.float32)
        # elif lightscale == "-light0256.npz":
        noisy_img = seq[wins*i:wins*(i+1)].mean(axis=0).astype(np.float32)
        noisy_img = np.expand_dims(noisy_img, axis=0).astype(np.float32)
        # gt_img = np.zeros([H, W], dtype=np.float32)
        noisy_img_list.append(noisy_img)
    gt_img = tag[yy:yy + cfg.image_height, xx:xx + cfg.image_width] / 255.0
    gt_img = np.expand_dims(gt_img, axis=0).astype(np.float32)
    gt_img_list.append(gt_img)
    gt_raw_batch = np.concatenate(gt_img_list, axis=0)
    noisy_raw_batch = np.concatenate(noisy_img_list, axis=0)

    return torch.from_numpy(noisy_raw_batch), torch.from_numpy(gt_raw_batch)


class loadImgs(Dataset):

    def __init__(self, filelist, tr_or_ev, use_cache=False, cached_data=None, cached_label=None):
        self.filelist = filelist
        self.tr_or_ev = tr_or_ev
        self.cached_data = []
        self.cached_label = []
        self.use_cache = use_cache
        if cached_data is not None:
            self.cached_data = cached_data
            self.cached_label = cached_label
        super(loadImgs, self).__init__()

    def __getitem__(self, item):
        if not self.use_cache:
            self.data_name = self.filelist[item]
            image, label = decode_data(self.data_name, self.tr_or_ev)
            self.cached_data.append(image)
            self.cached_label.append(label)
            # print(len(self.cached_data))

        else:
            image = self.cached_data[item]
            label = self.cached_label[item]

        return image, label

    def set_use_cache(self, use_cache):
        if use_cache:
            x_img = tuple(self.cached_data)
            y_img = tuple(self.cached_label)
            # noisy_level = tuple(self.cached_noisy_level)
            # print(self.cached_data[0])
            self.cached_data = torch.stack(x_img)
            self.cached_label = torch.stack(y_img)
        else:
            self.cached_data = []
            self.cached_label = []
        self.use_cache = use_cache

    def __len__(self):
        return len(self.filelist)

# if __name__ == '__main__':
#     file_list = generate_file_list("train")
#     noisy, gt, level = decode_data(file_list[0])
