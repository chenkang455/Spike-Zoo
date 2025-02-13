from json import load
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import DoubleNet
import cv2

def load_vidar_dat(filename, left_up=(0, 0), window=None, frame_cnt = None, **kwargs):
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
    
    height = 250
    width = 400

    if window == None:
        window = (height - left_up[0], width - left_up[0])

    len_per_frame = height * width // 8
    framecnt = frame_cnt if frame_cnt != None else len(array) // len_per_frame

    spikes = []

    for i in range(framecnt):
        compr_frame = array[i * len_per_frame: (i + 1) * len_per_frame]
        blist = []
        for b in range(8):
            blist.append(np.right_shift(np.bitwise_and(compr_frame, np.left_shift(1, b)), b))
        
        frame_ = np.stack(blist).transpose()
        frame_ = np.flipud(frame_.reshape((height, width), order='C'))

        if window is not None:
            spk = frame_[left_up[0]:left_up[0] + window[0], left_up[1]:left_up[1] + window[1]]
        else:
            spk = frame_

        spk = torch.from_numpy(spk.copy().astype(np.float32)).unsqueeze(dim=0)

        spikes.append(spk)

    return torch.cat(spikes)

if __name__ == '__main__':
    model = DoubleNet()
    model_path = "./fin3g-best-lucky.pt"
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path)) 
    model = model.cuda()
    
    spike_path = "./rotation1.dat"
    spike = load_vidar_dat(spike_path)[200:200+41].unsqueeze(0).cuda()

    res = model(spike)
    res = res[0].detach().cpu().permute(1,2,0).numpy()*255
    res_path = "./res.png"
    cv2.imwrite(res_path,res)

    print("done.")