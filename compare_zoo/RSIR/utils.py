import numpy as np
import cv2

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def load_spike_numpy(path):
    '''
    Load a spike sequence with it's tag from prepacked `.npz` file.\n
    The sequence is of shape (`length`, `height`, `width`) and tag of
    shape (`height`, `width`).
    '''
    data = np.load(path)
    seq, tag, length = data['seq'], data['tag'], int(data['length'])
    seq = np.array([(seq[i // 8] >> (i & 7)) & 1 for i in range(length)])
    return seq, tag, length


def RawToSpike(video_seq, h, w):
    '''
    Load a spike sequence with it's tag from raw file ended with `.dat`.
    '''
    video_seq = np.array(video_seq).astype(np.uint8)
    img_size = h*w
    img_num = len(video_seq)//(img_size//8)
    SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
    pix_id = np.arange(0, h*w)
    pix_id = np.reshape(pix_id, (h, w))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in np.arange(img_num):
        id_start = img_id*img_size//8
        id_end = id_start + img_size//8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))

    return SpikeMatrix


def cal_para():
    '''
    Calculate the noise parameters (q for Q_r, fo1 for L_d, fpn_test for R) by calibration.
    '''
    # The illuminance of three scenes being calibrated, change them with your calibration.
    lux1 = 0
    lux2 = 153
    lux3 = 335

    # load and process light data
    light_data = open("compare_zoo/RSIR/uniformlight/light.dat", 'rb+')
    light_seq = light_data.read()
    light_seq = np.fromstring(light_seq, 'B')
    light_spike = RawToSpike(light_seq, 250, 416)[:, :, :400]  # c*h*w
    # print(light_spike.shape)
    light_img = light_spike[0:20000].mean(axis=0).astype(np.float32)
    # cv2.imwrite("./uniformlight/light.png", light_img * 255)
    D_light = 1 / light_img - 1

    # load meduim  data
    meduim_data = open("compare_zoo/RSIR/uniformlight/medium.dat", 'rb+')
    meduim_seq = meduim_data.read()
    meduim_seq = np.fromstring(meduim_seq, 'B')
    meduim_spike = RawToSpike(meduim_seq, 250, 416)[:, :, :400]  # c*h*w
    meduim_img = meduim_spike[0:20000].mean(axis=0).astype(np.float32)
    # cv2.imwrite("./uniformlight/medium.png", meduim_img * 255)
    D_meduim = 1 / meduim_img - 1

    # load dark  data
    dark_data = open("compare_zoo/RSIR/uniformlight/dark.dat", 'rb+')
    dark_seq = dark_data.read()
    dark_seq = np.fromstring(dark_seq, 'B')
    dark_spike = RawToSpike(dark_seq, 250, 416)[:, :, :400]  # c*h*w
    dark_img = dark_spike[0:20000].mean(axis=0).astype(np.float32)
    # cv2.imwrite("./uniformlight/dark.png", dark_img * 255)
    D_dark = 1 / (dark_img + 1e-5) - 1

    th = D_meduim / D_dark
    fo1 = (lux2 * th - lux1) / (1 - th)

    base = D_light[0, 0]
    fpn_test = base * (lux3 + fo1[0, 0]) / (D_light * (lux3 + fo1))

    q = D_light * (lux3 + fo1)

    return q, fo1, fpn_test
