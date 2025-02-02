import numpy as np
import torch
import torch.nn as nn
import os
from .vidar_loader import load_vidar_dat_cpp
from typing import Literal

def load_vidar_dat(filename, height, width,remove_head=False, version:Literal['python','cpp'] = "cpp", out_format : Literal['array','tensor']="array",):
    """Load the spike stream from the .dat file."""
    # Spike decode
    if version == "python":
        if isinstance(filename, str):
            array = np.fromfile(filename, dtype=np.uint8)
        elif isinstance(filename, (list, tuple)):
            l = []
            for name in filename:
                a = np.fromfile(name, dtype=np.uint8)
                l.append(a)
            array = np.concatenate(l)
        else:
            raise NotImplementedError
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
            spk = spk[:, :, :-16] if remove_head == True else spk
            spikes.append(spk)
        spikes = np.concatenate(spikes).astype(np.float32)
    elif version == "cpp":
        spikes = load_vidar_dat_cpp(filename, height, width)
    else:
        raise RuntimeError("Not recognized version.")

    # # Output format conversion
    format_dict = {"array": lambda x: x, "tensor": torch.from_numpy}
    spikes = format_dict[out_format](spikes)
    return spikes


def SpikeToRaw(save_path, SpikeSeq, filpud=True, delete_if_exists=True):
    """Save the spike sequence to the .dat file."""
    if delete_if_exists:
        if os.path.exists(save_path):
            os.remove(save_path)
    sfn, h, w = SpikeSeq.shape
    remainder = int((h * w) % 8)
    base = np.power(2, np.linspace(0, 7, 8))
    fid = open(save_path, 'ab')
    for img_id in range(sfn):
        if filpud:
            spike = np.flipud(SpikeSeq[img_id, :, :])
        else:
            spike = SpikeSeq[img_id, :, :]
        if remainder == 0:
            spike = spike.flatten()
        else:
            spike = np.concatenate([spike.flatten(), np.array([0]*(8-remainder))])
        spike = spike.reshape([int(h*w/8), 8])
        data = spike * base
        data = np.sum(data, axis=1).astype(np.uint8)
        fid.write(data.tobytes())
    fid.close()
    return

def video2spike_simulation(imgs, threshold=2.0):
    """Convert the images input to the spike stream."""
    imgs = np.array(imgs)
    T,H, W = imgs.shape
    spike = np.zeros([T, H, W], np.uint8)
    integral = np.random.random(size=([H,W])) * threshold
    for t in range(0, T):
        integral += imgs[t]
        fire = (integral - threshold) >= 0
        fire_pos = fire.nonzero()
        integral[fire_pos] -= threshold
        spike[t][fire_pos] = 1
    return spike


