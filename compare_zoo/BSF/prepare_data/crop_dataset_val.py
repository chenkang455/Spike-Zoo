import os
import os.path as osp
import argparse
import cv2
import numpy as np
from io_utils import *
import h5py
from tqdm import *
from DSFT import DSFT

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="/data/rzhao/REDS120fps")
parser.add_argument("--output_path", type=str, default="/data/rzhao/REDS120fps/crop")
###### 参数
parser.add_argument("--eta", type=float, default=1.00)
parser.add_argument("--gamma", type=int, default=60)
parser.add_argument("--alpha", type=float, default=0.7)

parser.add_argument("--cu", '-c', type=str, default='0')

parser.add_argument("--crop_image", action='store_true')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cu

if __name__ == '__main__':
    imgs_path = osp.join(args.root, 'imgs', 'val')
    spks_path = osp.join(args.root, 'spikes', 'val', 
                        "eta_{:.2f}_gamma_{:d}_alpha_{:.1f}".format(args.eta, args.gamma, args.alpha))

    scene_list = sorted(os.listdir(spks_path))    
    for scene in tqdm(scene_list):
        scene_imgs_path = osp.join(imgs_path, scene)
        scene_spks_path = osp.join(spks_path, scene)

        if not args.crop_image:
            # read all the dat files
            dat_path = sorted(os.listdir(scene_spks_path))
            spks_list = []
            #### abandon 00000000.dat, corresponding to the following spike_idx_offset
            for dat_name in dat_path[1:]:
                spks_list.append(dat_to_spmat(dat_path=osp.join(scene_spks_path, dat_name), size=(720, 1280)))
            spikes = np.concatenate(spks_list, axis=0)

            # spikes -> DSFT(max_search_half_win=80)
            dsft_solver = DSFT(spike_h=720, spike_w=1280, device='cuda')
            dsft = dsft_solver.spikes2images(spikes, max_search_half_window=100)


        # crop Image
        if args.crop_image:
            imgs_list = []
            for im_idx in range(11, 28+1):
                    img = cv2.imread(osp.join(scene_imgs_path, '{:08d}.png'.format(im_idx)))
                    # 1. central crop
                    crop_img = img[32:-32, 128:-128]
                    for sub_scene_idx in range(4):
                        cur_scene = '{:s}_{:d}'.format(scene, sub_scene_idx)
                        if sub_scene_idx == 0:
                            cur_crop_img = crop_img[:384, :512]
                        elif sub_scene_idx == 1:
                            cur_crop_img = crop_img[-384:, :512]
                        elif sub_scene_idx == 2:
                            cur_crop_img = crop_img[:384, -512:]
                        elif sub_scene_idx == 3:
                            cur_crop_img = crop_img[-384:, -512:]

                        cur_save_root = osp.join(args.output_path, 'val_small', 'imgs', cur_scene)
                        os.makedirs(cur_save_root, exist_ok=True)
                        cur_save_path = osp.join(cur_save_root, '{:08d}.png'.format(im_idx))
                        if osp.exists(cur_save_path):
                            os.remove(cur_save_path)
                        cv2.imwrite(cur_save_path, cur_crop_img)
            continue


        # 裁切 spikes
        # since 00000000.dat is abandoned
        spike_idx_offset = 10
        # 1. central crop
        spikes = spikes[:, 32:-32, 128:-128]
        # 2. crop
        for spk_idx in range(11, 28+1):
            crop_spike = spikes[spk_idx*10-spike_idx_offset : spk_idx*10-spike_idx_offset+10]
            
            for sub_scene_idx in range(4):
                cur_scene = '{:s}_{:d}'.format(scene, sub_scene_idx)
                if sub_scene_idx == 0:
                    cur_crop_spike = crop_spike[:, :384, :512]
                elif sub_scene_idx == 1:
                    cur_crop_spike = crop_spike[:, -384:, :512]
                elif sub_scene_idx == 2:
                    cur_crop_spike = crop_spike[:, :384, -512:]
                elif sub_scene_idx == 3:
                    cur_crop_spike = crop_spike[:, -384:, -512:]

                cur_save_root = osp.join(args.output_path, 'val_small',
                                "eta_{:.2f}_gamma_{:d}_alpha_{:.1f}".format(args.eta, args.gamma, args.alpha), 
                                cur_scene,
                                'spikes')

                os.makedirs(cur_save_root, exist_ok=True)
                cur_save_path = osp.join(cur_save_root,'{:08d}.dat'.format(spk_idx))
                if osp.exists(cur_save_path):
                    os.remove(cur_save_path)
                SpikeToRaw(SpikeSeq=cur_crop_spike, save_path=cur_save_path)


        # crop dsft
        dsft_idx_offset = 10 + 100
        # 1. central crop
        dsft = dsft[:, 32:-32, 128:-128]
        # 2. crop
        for dsft_idx in range(11, 28+1):
            crop_dsft = dsft[dsft_idx*10-dsft_idx_offset : dsft_idx*10-dsft_idx_offset+10]

            for sub_scene_idx in range(4):
                cur_scene = '{:s}_{:d}'.format(scene, sub_scene_idx)
                if sub_scene_idx == 0:
                    cur_crop_dsft = crop_dsft[:, :384, :512]
                elif sub_scene_idx == 1:
                    cur_crop_dsft = crop_dsft[:, -384:, :512]
                elif sub_scene_idx == 2:
                    cur_crop_dsft = crop_dsft[:, :384, -512:]
                elif sub_scene_idx == 3:
                    cur_crop_dsft = crop_dsft[:, -384:, -512:]

                cur_save_root = osp.join(args.output_path, 'val_small',
                                "eta_{:.2f}_gamma_{:d}_alpha_{:.1f}".format(args.eta, args.gamma, args.alpha), 
                                cur_scene,
                                'dsft')
                os.makedirs(cur_save_root, exist_ok=True)
                cur_save_path = osp.join(cur_save_root, '{:08d}.h5'.format(dsft_idx))
                if osp.exists(cur_save_path):
                    os.remove(cur_save_path)
                f = h5py.File(cur_save_path, 'w')
                f['dsft'] = cur_crop_dsft
                f.close()

