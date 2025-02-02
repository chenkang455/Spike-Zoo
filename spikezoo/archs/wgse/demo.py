import os

import numpy as np
import cv2
import torch
import argparse

from dwtnets import Dwt1dResnetX_TCN
from utils import RawToSpike
from einops import rearrange
from pytorch_wavelets import DWT1DForward


parser = argparse.ArgumentParser(description='AAAI - WGSE - REDS')
parser.add_argument('-w', '--wvl', type=str, default='db8', help='select wavelet base function')
parser.add_argument('-j', '--jlevels', type=int, default=5)
parser.add_argument('-k', '--kernel_size', type=int, default=3)
parser.add_argument('-l', '--logpath', type=str, default='WGSE-Dwt1dNet')
parser.add_argument('-f', '--datfile', type=str, default="/home/chenkang455/chenk/myproject/SpikeCLIP/selfcode/data/recVidarReal2019/classB/train-350kmh.dat", help='path of the spike data to be tested')

args = parser.parse_args()
dataroot = args.datfile
wvlname = args.wvl
j = args.jlevels
logfolder = args.logpath
ks = args.kernel_size


def nor(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def progress_bar_time(total_time):
    hour = int(total_time) // 3600
    minu = (int(total_time) % 3600) // 60
    sec = int(total_time) % 60
    return '%d:%02d:%02d' % (hour, minu, sec)


def main():

    f = open(dataroot, 'rb')
    spike_seq = f.read()
    spike_seq = np.frombuffer(spike_seq, 'b')
    spikes = RawToSpike(spike_seq, 250, 400)
    spikes = spikes.astype(np.float32)
    spikes = torch.from_numpy(spikes)
    f.close()

    spikes = spikes[None, 130:171, :, :]

    s = spikes[:, :, 0:1, 0:1]
    dwt = DWT1DForward(wave=wvlname, J=j)
    s_r = rearrange(s, 'b t h w -> b h w t')
    s_r = rearrange(s_r, 'b h w t -> (b h w) 1 t')
    yl, yh = dwt(s_r)
    yl_size = yl.shape[-1]
    yh_size = [yhi.shape[-1] for yhi in yh]
    print(yl_size,yh_size)
    model = Dwt1dResnetX_TCN(
        wvlname=wvlname, J=j, yl_size=yl_size, yh_size=yh_size, num_residual_blocks=3, norm=None, ks=ks, store_features=True
    )
    print(model)

    saved_state_dict = torch.load('model_best.pt')
    model.load_state_dict(saved_state_dict.module.state_dict())
    
    model = model.cuda()
    model.eval()

    pred = model(spikes.cuda())        
    prediction = pred[0].permute(1,2,0).cpu().detach().numpy()
    print(os.path.join(logfolder, 'demo.png'))
    cv2.imwrite( 'demo.png', prediction * 255.0)
    

if __name__ == '__main__':
    main()

    