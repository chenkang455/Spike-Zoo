import numpy as np
import torch
import torch.nn as nn
import os
from typing import Literal
import platform
import cv2
import imageio

_platform_check_done = False


def load_vidar_dat(filename, height, width, remove_head=False, version: Literal["python", "cpp"] = "python", out_format: Literal["array", "tensor"] = "array"):
    """Load the spike stream from the .dat file."""
    global _platform_check_done
    # Spike decode
    if version == "cpp" and platform.system().lower() == "linux":
        from .vidar_loader import load_vidar_dat_cpp

        spikes = load_vidar_dat_cpp(filename, height, width)
    else:
        # todo double check
        if version == "cpp" and platform.system().lower() != "linux" and _platform_check_done == False:
            _platform_check_done = True
            print("Cpp load version is only supported on the linux now. Auto transfer to the python version.")
        array = np.fromfile(filename, dtype=np.uint8)
        len_per_frame = height * width // 8
        framecnt = len(array) // len_per_frame
        spikes = []
        for i in range(framecnt):
            compr_frame = array[i * len_per_frame : (i + 1) * len_per_frame]
            blist = []
            for b in range(8):
                blist.append(np.right_shift(np.bitwise_and(compr_frame, np.left_shift(1, b)), b))
            frame_ = np.stack(blist).transpose()
            frame_ = np.flipud(frame_.reshape((height, width), order="C"))
            spk = frame_.copy()[None]
            spikes.append(spk)
        spikes = np.concatenate(spikes).astype(np.float32)
    spikes = spikes[:, :, :-16] if remove_head == True else spikes

    # # Output format conversion
    format_dict = {"array": lambda x: x, "tensor": torch.from_numpy}
    spikes = format_dict[out_format](spikes)
    return spikes


def save_vidar_dat(save_path, SpikeSeq, filpud=True):
    """Save the spike sequence to the .dat file."""
    if os.path.exists(save_path):
        os.remove(save_path)
    sfn, h, w = SpikeSeq.shape
    remainder = int((h * w) % 8)
    base = np.power(2, np.linspace(0, 7, 8))
    fid = open(save_path, "ab")
    for img_id in range(sfn):
        if filpud:
            spike = np.flipud(SpikeSeq[img_id, :, :])
        else:
            spike = SpikeSeq[img_id, :, :]
        if remainder == 0:
            spike = spike.flatten()
        else:
            spike = np.concatenate([spike.flatten(), np.array([0] * (8 - remainder))])
        spike = spike.reshape([int(h * w / 8), 8])
        data = spike * base
        data = np.sum(data, axis=1).astype(np.uint8)
        fid.write(data.tobytes())
    fid.close()


def merge_vidar_dat(filename, dat_files, height, width, remove_head=False):
    """Merge selected spike dat files."""
    spikes = []
    for dat_file in dat_files:
        spike = load_vidar_dat(dat_file,height, width, remove_head)
        spikes.append(spike)
    spikes = np.concatenate(spikes, axis=0)
    save_vidar_dat(filename, spikes)
    return spikes

def visual_vidar_dat(filename, spike, out_format: Literal["mp4", "gif"] = "gif", fps=15):
    """Convert the spike stream to the video."""
    _, height, width = spike.shape
    if out_format == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或 'avc1'
        mp4_video = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    elif out_format == "gif":
        frames = []

    for idx in range(len(spike)):
        spk = spike[idx]
        spk = (255 * spk).astype(np.uint8)
        spk = spk[..., None].repeat(3, axis=-1)
        if out_format == "mp4":
            mp4_video.write(spk)
        elif out_format == "gif":
            frames.append(spk)

    if out_format == "mp4":
        mp4_video.release()
    elif out_format == "gif":
        imageio.mimsave(filename, frames, "GIF", fps=fps, loop=0)


def video2spike_simulation(imgs, threshold=2.0):
    """Convert the images input to the spike stream."""
    imgs = np.array(imgs)
    T, H, W = imgs.shape
    spike = np.zeros([T, H, W], np.uint8)
    integral = np.random.random(size=([H, W])) * threshold
    for t in range(0, T):
        integral += imgs[t]
        fire = (integral - threshold) >= 0
        fire_pos = fire.nonzero()
        integral[fire_pos] -= threshold
        spike[t][fire_pos] = 1
    return spike
