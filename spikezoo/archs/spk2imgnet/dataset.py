import os
import os.path
from typing import List

import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from functools import partial

bytes2num = partial(int.from_bytes, byteorder="little", signed=False)


def normalize(data):
    return data / 255.0


def raw_to_spike(video_seq, h, w):
    video_seq = np.array(video_seq).astype(np.uint8)
    img_size = h * w
    img_num = len(video_seq) // (img_size // 8)
    spike_matrix = np.zeros([img_num, h, w], np.uint8)
    pix_id = np.arange(0, h * w)
    pix_id = np.reshape(pix_id, (h, w))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in np.arange(img_num):
        id_start = img_id * img_size // 8
        id_end = id_start + img_size // 8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        spike_matrix[img_id, :, :] = np.flipud((result == comparator))

    return spike_matrix


def Im2Patch(img, win, stride=40):
    k = 0
    [endc, endw, endh] = img.shape
    patch = img[:, 0: endw - win + 0 + 1: stride, 0: endh - win + 0 + 1: stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, total_pat_num], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[
                    :, i: endw - win + i + 1: stride, j: endh - win + j + 1: stride
                    ]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
            k = k + 1
    return Y.reshape([endc, win, win, total_pat_num])


def read_image_and_concat_as_tensor(paths: List[str]):
    tensors = []
    for path in paths:
        img = cv2.imread(path)
        tensors.append(img.reshape([1, *img.shape]))
    return np.concatenate(tensors, axis=0)


def prepare_data(data_path, patch_size, stride, h5_name, aug_times=1):
    print("process training data")
    input_files = glob.glob(os.path.join(data_path, "input", "*.dat"))
    print(len(input_files))
    input_files.sort()
    input_h5f = h5py.File(h5_name + "_input.h5", "w")
    gt_h5f = h5py.File(h5_name + "_gt.h5", "w")
    train_num = 0
    h = 250
    w = 400
    for i in range(len(input_files)):
        input_f = open(input_files[i], "rb+")
        video_seq = input_f.read()
        video_seq = np.fromstring(video_seq, "B")
        # print(video_seq)
        spike_array = raw_to_spike(video_seq, h, w)  # c*h*w
        # print(input_files[i][:-3])
        # SpikeArray = SpikeArray[10:-10, :, :]
        # print(np.mean(SpikeArray))
        print(spike_array.shape)
        file_name = input_files[i].replace("\\", "/").split("/")[-1]
        gt = []
        for num in [7, 14, 21, 28, 35]:
            img = cv2.imread(os.path.join(data_path, "gt", file_name[:-6] + str(num) + ".png"), 0)
            gt.append(img.reshape([1, *img.shape]))
        gt = np.concatenate(gt, axis=0)
        print(input_files[i])
        print(os.path.join(data_path, "gt", file_name[:-3] + "png"))
        gt = np.float32(normalize(gt))  # size
        print(gt.shape)
        print(spike_array.shape)
        input_patches = Im2Patch(spike_array, win=patch_size, stride=stride)
        gt_patches = Im2Patch(gt, win=patch_size, stride=stride)
        assert input_patches.shape[3] == gt_patches.shape[3]
        for n in range(input_patches.shape[3]):
            inputs = input_patches[:, :, :, n].copy()
            input_h5f.create_dataset(str(train_num), data=inputs)
            gt = gt_patches[:, :, :, n].copy()
            gt_h5f.create_dataset(str(train_num), data=gt)
            train_num += 1

    input_h5f.close()
    gt_h5f.close()


class Dataset(udata.Dataset):
    def __init__(self, h5_name):
        super(Dataset, self).__init__()
        input_h5f = h5py.File(h5_name + "_input.h5", "r")
        gt_h5f = h5py.File(h5_name + "_gt.h5", "r")
        self.h5_name = h5_name
        self.keys = list(input_h5f.keys())
        # print(self.keys)
        random.shuffle(self.keys)
        input_h5f.close()
        gt_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        input_h5f = h5py.File(self.h5_name + "_input.h5", "r")
        gt_h5f = h5py.File(self.h5_name + "_gt.h5", "r")
        key = self.keys[index]
        inputs = np.array(input_h5f[key])
        gt = np.array(gt_h5f[key])
        input_h5f.close()
        gt_h5f.close()
        return torch.Tensor(inputs), torch.Tensor(gt)


if __name__ == "__main__":
    prepare_data(
        data_path="./Spk2ImgNet_train/train2/",
        patch_size=40,
        stride=40,
        h5_name="train",
    )
    # PrepareData(data_path = './SpikeDataset/val/', patch_size=40, stride=40, h5_name='val')
