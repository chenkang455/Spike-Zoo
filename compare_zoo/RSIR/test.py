import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
import structure
import netloss
from utils import load_spike_numpy, RawToSpike, cal_para


device = cfg.device
model = structure.MainDenoise()
model = model.to(device)

checkpoint = torch.load(cfg.test_checkpoint)
model.load_state_dict(checkpoint['model'])
best_PSNR = checkpoint['best_psnr']
print("the best PSNR is :{}".format(best_PSNR))

Q, Nd, Nl = cal_para()
q = torch.from_numpy(Q).to(device)
nd = torch.from_numpy(Nd).to(device)
nl = torch.from_numpy(Nl).to(device)

model.eval()
test_scene = ["tuk-tuk", "train", "upside-down", "varanus-cage", "walking"]
light_scale = ["0256", "0032"]
total_psnr = 0
total_ssim = 0
cnt = 0
with torch.no_grad():
    for tag in test_scene:
        for ls in light_scale:
            h, w = 250, 400
            seq, label, length = load_spike_numpy(cfg.simulated_dir+"spike-video{}-00000-light{}.npz".format(tag, ls))
            if ls == "0256":
                wins = 16
            elif ls == "0032":
                wins = 32
            for i in range(cfg.frame_num):
                noisy_img = np.ones([1, 1, 250, w], dtype=np.float32)
                noisy_img[0, 0, :, :] = seq[wins*i:wins*(i+1)].mean(axis=0).astype(np.float32)**(1/2.2)
                fgt = np.ones([1, 1, 250, w], dtype=np.float32)
                fgt[0, 0, :, :] = (label / 255.0)**(1/2.2)
                cv2.imwrite("./result/gt_{}.png".format(tag), fgt[0, 0] * 255.0)
                cv2.imwrite("./result/noisy_{}.png".format(tag), noisy_img[0, 0] * 255.0)
                noisy_img = torch.from_numpy(noisy_img).to(device)
                fgt = torch.from_numpy(fgt).to(device)
                if i == 0:
                    input = noisy_img
                else:
                    input = torch.cat([noisy_img, ft0_fusion_data], dim=1)

                fpn_denoise, img_true, fusion_out, denoise_out, refine_out, ft_denoise_out_d0, fgt_d0 = model(input, fgt, q, nd, nl)
                fgt = F.pad(fgt, [0, 0, 3, 3], mode="reflect")
                ft0_fusion_data = fusion_out[:, :, 3:253, :]
                if i == cfg.frame_num-1:
                    psnr = netloss.PSNR().to(device)
                    ssim = netloss.SSIM().to(device)
                    cnt += 1
                    PSNR = psnr(refine_out, fgt).item()
                    SSIM = ssim(refine_out, fgt).item()
                    total_psnr += PSNR
                    total_ssim += SSIM
                    print("%10s: PSNR:%.2f SSIM:%.4f" % (tag, PSNR, SSIM))

    print("Total PSNR:")
    print(total_psnr / cnt)
    print("Total SSIM:")
    print(total_ssim / cnt)
