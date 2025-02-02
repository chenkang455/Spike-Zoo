import os
import time

import numpy as np
import cv2
import torch
import argparse

from dataset import DatasetREDS
from dwtnets import Dwt1dResnetX_TCN
from utils import calculate_psnr, calculate_ssim
from einops import rearrange
from pytorch_wavelets import DWT1DForward


parser = argparse.ArgumentParser(description='AAAI - WGSE - REDS')
parser.add_argument('-w', '--wvl', type=str, default='db8', help='select wavelet base function')
parser.add_argument('-j', '--jlevels', type=int, default=5)
parser.add_argument('-k', '--kernel_size', type=int, default=3)
parser.add_argument('-l', '--logpath', type=str, default='WGSE-Dwt1dNet')
parser.add_argument('--dataroot', type=str, default=None)

args = parser.parse_args()
dataroot = args.dataroot
wvlname = args.wvl
j = args.jlevels
logfolder = args.logpath
ks = args.kernel_size


def nor(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def save_wvl(wvl, savefolder, saveprefix):
    wvl = wvl.squeeze()
    t, h, w = wvl.shape
    wvl = nor(wvl)
    
    for i in range(t):
        wvl_t = wvl[i] * 255
        cv2.imwrite(os.path.join(savefolder, saveprefix+'_{:03d}.png'.format(i)), wvl_t)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def progress_bar_time(total_time):
    hour = int(total_time) // 3600
    minu = (int(total_time) % 3600) // 60
    sec = int(total_time) % 60
    return '%d:%02d:%02d' % (hour, minu, sec)


def main():
    cfg = {}
    cfg['rootfolder'] = os.path.join(dataroot, 'val')
    cfg['spikefolder'] = 'input'
    cfg['imagefolder'] = 'gt'
    cfg['H'] = 250
    cfg['W'] = 400
    cfg['C'] = 41
    test_set = DatasetREDS(cfg)
    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    item0 = test_set[0]
    s = item0['spikes']
    s = s[None, :, 0:1, 0:1]
    dwt = DWT1DForward(wave=wvlname, J=j)
    B, T, H, W = s.shape
    s_r = rearrange(s, 'b t h w -> b h w t')
    s_r = rearrange(s_r, 'b h w t -> (b h w) 1 t')
    yl, yh = dwt(s_r)
    yl_size = yl.shape[-1]
    yh_size = [yhi.shape[-1] for yhi in yh]
    
    model = Dwt1dResnetX_TCN(
        wvlname=wvlname, J=j, yl_size=yl_size, yh_size=yh_size, num_residual_blocks=3, norm=None, ks=ks, store_features=True
    )
    print(model)

    saved_state_dict = torch.load(logfolder + '/model_best.pt')
    model.load_state_dict(saved_state_dict.module.state_dict())
    
    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()

    model.eval()
    
    with torch.no_grad():
        sum_ssim = 0.0
        sum_psnr = 0.0
        sum_num = 0
        total_time = 0
        for i, item in enumerate(test_data_loader):
            start_time = time.time()

            spikes = item['spikes'][:, 130:171, :, :].cuda()
            image = item['image'].cuda()

            pred = model(spikes)
            
            prediction = pred[0].permute(1,2,0).cpu().numpy()
            gt = image[0].permute(1,2,0).cpu().numpy()

            sum_ssim += calculate_ssim(gt * 255.0, prediction * 255.0)
            sum_psnr += calculate_psnr(gt * 255.0, prediction * 255.0)
            sum_num += 1
            elapse_time = time.time() - start_time
            total_time += elapse_time

            print('\r[evaluating] %3.2f%% | %6d/%6d [%s<%s, %.2fs/it]' % (
                float(i + 1) / int(len(test_data_loader)) * 100, i + 1, int(len(test_data_loader)),
                progress_bar_time(total_time),
                progress_bar_time(total_time / (i + 1) * int(len(test_data_loader))),
                total_time / (i + 1)), end='')
                            
        sum_psnr /= sum_num
        sum_ssim /= sum_num

    print('')
    print('\r[Evaluation Result] PSNR: %.4f | SSIM: %.4f' % (sum_psnr, sum_ssim))


if __name__ == '__main__':
    main()

    