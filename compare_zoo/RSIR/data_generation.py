import os
import cv2
from os.path import splitext
import numpy as np
import torch
import tqdm
from PIL import Image
import math
import argparse
import scipy.io as scio
import matplotlib.pyplot as plt


def dump_spike_numpy(path: str, seq: np.ndarray, tag: np.ndarray):
    '''
    Store a spike sequence with it's tag to `.npz` file. (array -> .npz)
    '''
    length = seq.shape[0]
    seq = seq.astype(np.bool)
    seq = np.array([seq[i] << (i & 7) for i in range(length)])
    seq = np.array([np.sum(seq[i: min(i + 8, length)], axis=0)
                    for i in range(0, length, 8)]).astype(np.uint8)
    np.savez(path, seq=seq, tag=tag, length=np.array(length))


def load_spike_raw(path: str, width=400, height=250) -> np.ndarray:
    '''
    Load bit-compact raw spike data into an ndarray of shape (.dat -> array)
        (`frame number`, `height`, `width`).
    '''
    with open(path, 'rb') as f:
        fbytes = f.read()
    fnum = (len(fbytes) * 8) // (width * height)  # number of frames
    frames = np.frombuffer(fbytes, dtype=np.uint8)
    frames = np.array([frames & (1 << i) for i in range(8)])
    frames = frames.astype(np.bool).astype(np.uint8)
    frames = frames.transpose(1, 0).reshape(fnum, height, width)
    frames = np.flip(frames, 1)
    return frames


def RawToSpike(video_seq, h, w):
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


def TFI_gamma(seq: np.ndarray, mid: int, gamma: float) -> np.ndarray:
    '''
    Snapshot an image using interval method.
    '''
    length, height, width = seq.shape
    result = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            up, down = mid, mid - 1
            for up in range(mid, length):
                if seq[up, i, j] == 1:
                    break
            for down in range(mid - 1, -1, -1):
                if seq[down, i, j] == 1:
                    break
            result[i, j] = math.pow(255 / (up - down), 1 / gamma)
    result = (255 * result / (np.max(result)))
    return result.astype(np.uint8)


def TFP_gamma(seq: np.ndarray, start: int, length: int, gamma: float) -> np.ndarray:
    '''
    Generate an image using window method.
    '''
    result = seq[start:start + length].mean(axis=0) * 255
    result = np.power(result, 1 / gamma)
    result = (255 * result / (np.max(result)))
    return result.astype(np.uint8)


def main(WINDOW_SIZE: int, light_scale: str, video_path: str, out_path: str, resize: tuple, GT_frameno: int, fo1 : np.ndarray, fpn_test: np.ndarray, lux3: int):
    out_pattern_ivs = 'spike-video{:s}-light{:04d}.npz'
    intermediate_save = False

    total_num = 0
    video_folder = os.listdir(video_path)

    channels = WINDOW_SIZE
    if GT_frameno == None:
        GT_frameno = channels / 2 + 1

    for video_n in video_folder[:]:
        my_videos = os.path.join(video_path, video_n)
        video_list = os.listdir(my_videos)
        for video_name in video_list[:10]:

            in_name = os.path.join(my_videos, video_name)
            out_name1 = os.path.join(out_path, out_pattern_ivs)
            cap = cv2.VideoCapture(in_name)
            frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            size = [width, height]

            iter_num = 100

            cap = cv2.VideoCapture(in_name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print('width : {}\nheight: {}\nfps   : {}\nframes: {}'.format(width, height, fps, frame_num))

            threshold = 255
            # threshold = np.random.normal(loc=180, scale=np.sqrt(50))
            if resize != None:
                integrator = np.random.random((resize[1], resize[0])) * threshold
                # integrator = np.random.random((int(size[1]/down_scale), int(size[0]/down_scale))) * threshold
            win_num = 0

            xframes = []  # (window, height, width)
            yframes = None  # (height, width)
            light_intensity = light_scale / 256  # scale
            win_size = 0
            for numf in range(frame_num):
                while (True):
                    if win_size == -1:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    frame_ori = frame
                    if resize != None:
                        frame = cv2.resize(frame, resize)
                    frame = frame * light_intensity
                    for i in range(iter_num):
                        # if i % 16 == 0:
                        # add poisson noise
                        frame_poisson = np.clip(np.random.poisson(frame), 0, 255)
                        # frame_gaussian = np.clip(frame_poisson + np.random.normal(0, 20 * (k * frame_poisson + b)), 0, 255)
                        # trans to interval
                        interval = 255 / (frame_poisson + 1e-5) - 1
                        interval_fpn = interval / (fpn_test + interval * fo1 / (D_light * (lux3 + fo1)))
                        # add quantization noise
                        interval_fpn_quantization = interval_fpn + np.random.uniform(0, 1, size=interval_fpn.shape)
                        # trans to light
                        frame_fpn = 255 / ((interval_fpn_quantization) + 1)
                        integrator += frame_fpn

                        spike_frame = integrator >= threshold
                        xframes.append(spike_frame)
                        integrator -= spike_frame * 255  # ------------------way one
                        # integrator = integrator * (1 - spike_frame)  # ------way two
                        win_size += 1

                        if win_size == GT_frameno:  # 31:
                            yframes = frame_ori
                        if win_size == channels:  # end this window
                            xframes = np.array(xframes).astype(np.bool)
                            if resize != None:
                                yframes = cv2.resize(yframes, resize)
                            yframes = yframes.astype(np.uint8)
                            # print(xframes)
                            # print(yframes)

                            dump_spike_numpy(out_name1.format(video_n + '-' + video_name[0:-4], light_scale), xframes,
                                             yframes * light_intensity)
                            # dump_spike_numpy_notag(out_name2, xframes)

                            win_size = -1

                        if win_size == -1:
                            break
                    break
            total_num = total_num + 1
            print('Now processing video {}'.format(total_num))
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # init_tau, batch_size, learning_rate, T_max, log_dir, use_plif
    parser.add_argument('-window_size', type=int, default=300, help="The total number of synthesis frames")
    parser.add_argument('-iter_num', type=int, default=300, help="The number of repetitions of a single image, and should be equal to the window_size if you need to simulate a stationary scene")
    parser.add_argument('-light_scale', type=int, default=256, help="Adjust the maximum brightness linearly")
    parser.add_argument('-video_path', type=str, default="../Dataset/DAVIS/JPEGImages/480p")
    parser.add_argument('-out_path', type=str, default="./sys_data_motion")
    # parser.add_argument('-down_scale', type=int, default=1) 
    parser.add_argument('-resize', type=tuple, default=(400, 250))
    parser.add_argument('-GT_frameno', type=int, default=None, help="the number of GT frame")
    args = parser.parse_args()
    WINDOW_SIZE = args.WINDOW_SIZE
    light_scale = args.light_scale
    video_path = args.video_path
    out_path = args.out_path
    # down_scale = args.down_scale
    resize = args.resize
    GT_frameno = args.GT_frameno

    # load and process light data
    light_data = open("./uniformlight_new/light.dat", 'rb+')
    light_seq = light_data.read()
    light_seq = np.fromstring(light_seq, 'B')
    light_spike = RawToSpike(light_seq, 250, 416)[:, :, :400]  # c*h*w
    light_img = light_spike[0:20000].mean(axis=0).astype(np.float32)
    D_light = 1 / light_img - 1


    # load meduim data
    meduim_data = open("./uniformlight_new/medium.dat", 'rb+')
    meduim_seq = meduim_data.read()
    meduim_seq = np.fromstring(meduim_seq, 'B')
    meduim_spike = RawToSpike(meduim_seq, 250, 416)[:, :, :400]  # c*h*w
    meduim_img = meduim_spike[0:20000].mean(axis=0).astype(np.float32)
    D_meduim = 1 / meduim_img - 1

    # load dark data
    dark_data = open("./uniformlight_new/dark.dat", 'rb+')
    dark_seq = dark_data.read()
    dark_seq = np.fromstring(dark_seq, 'B')
    dark_spike = RawToSpike(dark_seq, 250, 416)[:, :, :400]  # c*h*w
    dark_img = dark_spike[0:20000].mean(axis=0).astype(np.float32)
    D_dark = 1 / (dark_img + 1e-5) - 1

    # cal fo1 and fpn_test when they are not given
    lux1 = 0
    lux2 = 153
    lux3 = 335

    th = D_meduim / D_dark
    fo1 = (lux2 * th - lux1) / (1 - th)

    base = D_light[0, 0]
    fpn_test = base * (lux3 + fo1[0, 0]) / (D_light * (lux3 + fo1))
    main(WINDOW_SIZE, light_scale, video_path, out_path, resize, GT_frameno, fo1, fpn_test, lux3)