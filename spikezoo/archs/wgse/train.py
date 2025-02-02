import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWT1DForward

from transform import Compose, RandomCrop, RandomRotationFlip
from dataset import DatasetREDS
from dwtnets import Dwt1dResnetX_TCN
from utils import calculate_psnr, calculate_ssim, mkdir

parser = argparse.ArgumentParser(description='AAAI - WGSE - REDS')
parser.add_argument('-c', '--cuda', type=str, default='1', help='select gpu card')
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-e', '--epoch', type=int, default=600)
parser.add_argument('-w', '--wvl', type=str, default='db8', help='select wavelet base function')
parser.add_argument('-j', '--jlevels', type=int, default=5)
parser.add_argument('-k', '--kernel_size', type=int, default=3)
parser.add_argument('-l', '--logpath', type=str, default='WGSE-Dwt1dNet')
parser.add_argument('-r', '--resume_from', type=str, default=None)
parser.add_argument('--dataroot', type=str, default=None)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

resume_folder = args.resume_from
batch_size = args.batch_size
learning_rate = 1e-4
train_epoch = args.epoch
dataroot = args.dataroot

opt = 'adam'
opt_param = "{\"beta1\":0.9,\"beta2\":0.99,\"weight_decay\":0}"

random_seed = True
manual_seed = 123

scheduler = "MultiStepLR"
scheduler_param = "{\"milestones\": [400, 600], \"gamma\": 0.2}"

wvlname = args.wvl
j = args.jlevels
ks = args.kernel_size

if_save_model = False
eval_freq = 1
checkpoints_folder = args.logpath + '-' + args.wvl + '-' + str(args.jlevels) + '-' + 'ks' + str(ks)


def progress_bar_time(total_time):
    hour = int(total_time) // 3600
    minu = (int(total_time) % 3600) // 60
    sec = int(total_time) % 60
    return '%d:%02d:%02d' % (hour, minu, sec)

def main():

    global batch_size, learning_rate, random_seed, manual_seed, opt, opt_param, if_save_model, checkpoints_folder

    mkdir(os.path.join('logs', checkpoints_folder))

    if random_seed:
        seed = np.random.randint(0, 10000)
    else:
        seed = manual_seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    opt_param_dict = json.loads(opt_param)
    scheduler_param_dict = json.loads(scheduler_param)

    cfg = {}
    cfg['rootfolder'] = os.path.join(dataroot, 'train')
    cfg['spikefolder'] = 'input'
    cfg['imagefolder'] = 'gt'
    cfg['H'] = 250
    cfg['W'] = 400
    cfg['C'] = 41
    train_set = DatasetREDS(cfg,
        transform=Compose(
                [
                    RandomCrop(128),
                    RandomRotationFlip(0.0, 0.5, 0.5)
                ]),
    )

    cfg = {}
    cfg['rootfolder'] = os.path.join(dataroot, 'val')
    cfg['spikefolder'] = 'input'
    cfg['imagefolder'] = 'gt'
    cfg['H'] = 250
    cfg['W'] = 400
    cfg['C'] = 41
    test_set = DatasetREDS(cfg)

    print('train_set len', train_set.__len__())
    print('test_set len', test_set.__len__())

    train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,
            drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False)

    print(train_data_loader)
    print(test_data_loader)

    item0 = train_set[0]
    s = item0['spikes']
    s = s[None, :, 0:1, 0:1]
    dwt = DWT1DForward(wave=wvlname, J=j)
    B, T, H, W = s.shape
    s_r = rearrange(s, 'b t h w -> b h w t')
    s_r = rearrange(s_r, 'b h w t -> (b h w) 1 t')
    yl, yh = dwt(s_r)
    yl_size = yl.shape[-1]
    yh_size = [yhi.shape[-1] for yhi in yh]
    
    model = Dwt1dResnetX_TCN(inc=41, wvlname=wvlname, J=j, yl_size=yl_size, yh_size=yh_size, num_residual_blocks=3, norm=None, ks=ks)

    
    if args.resume_from:
        print("loading model weights from ", resume_folder)
        saved_state_dict = torch.load(os.path.join(resume_folder, 'model_best.pt'))
        model.load_state_dict(saved_state_dict.module.state_dict())
        print("Weighted loaded.")

    model = torch.nn.DataParallel(model).cuda()

    # optimizer
    if opt.lower() == 'adam':
        assert ('beta1' in opt_param_dict.keys() and 'beta2' in opt_param_dict.keys() and 'weight_decay' in opt_param_dict.keys())
        betas = (opt_param_dict['beta1'], opt_param_dict['beta2'])
        del opt_param_dict['beta1']
        del opt_param_dict['beta2']
        opt_param_dict['betas'] = betas
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, **opt_param_dict)
    elif opt.lower() == 'sgd':
        assert ('momentum' in opt_param_dict.keys() and 'weight_decay' in opt_param_dict.keys())
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, **opt_param_dict)
    else:
        raise ValueError()

    lr_scheduler = getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **scheduler_param_dict)
    best_psnr, best_ssim = 0.0, 0.0

    for epoch in range(train_epoch+1):
        print('Epoch %d/%d ... ' % (epoch, train_epoch))

        model.train()
        total_time = 0
        f = open(os.path.join('logs', checkpoints_folder, 'log.txt'), "a")
        for i, item in enumerate(train_data_loader):

            start_time = time.time()

            spikes = item['spikes'].cuda()
            image = item['image'].cuda()
            optimizer.zero_grad()
            
            pred = model(spikes)
            
            loss = F.l1_loss(image, pred)
            loss.backward()
            optimizer.step()

            elapse_time = time.time() - start_time  
            total_time += elapse_time

            lr_list = lr_scheduler.get_last_lr()
            lr_str = ""
            for ilr in lr_list:
                lr_str += str(ilr) + ' '
            print('\r[training] %3.2f%% | %6d/%6d [%s<%s, %.2fs/it] | LOSS: %.4f | LR: %s' % (
                float(i + 1) / int(len(train_data_loader)) * 100, i + 1, int(len(train_data_loader)),
                progress_bar_time(total_time),
                progress_bar_time(total_time / (i + 1) * int(len(train_data_loader))),
                total_time / (i + 1),
                loss.item(),
                lr_str), end='')
            f.write('[training] %3.2f%% | %6d/%6d [%s<%s, %.2fs/it] | LOSS: %.4f | LR: %s\n' % (
                float(i + 1) / int(len(train_data_loader)) * 100, i + 1, int(len(train_data_loader)),
                progress_bar_time(total_time),
                progress_bar_time(total_time / (i + 1) * int(len(train_data_loader))),
                total_time / (i + 1),
                loss.item(),
                lr_str))
        
        lr_scheduler.step()

        print('')
        if epoch % eval_freq == 0:
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
                    f.write('[evaluating] %3.2f%% | %6d/%6d [%s<%s, %.2fs/it]\n' % (
                    float(i + 1) / int(len(test_data_loader)) * 100, i + 1, int(len(test_data_loader)),
                    progress_bar_time(total_time),
                    progress_bar_time(total_time / (i + 1) * int(len(test_data_loader))),
                    total_time / (i + 1)))
                        
                sum_psnr /= sum_num
                sum_ssim /= sum_num

            print('')
            print('\r[Evaluation Result] PSNR: %.3f | SSIM: %.3f' % (sum_psnr, sum_ssim))
            f.write('[Evaluation Result] PSNR: %.3f | SSIM: %.3f\n' % (sum_psnr, sum_ssim))

        if if_save_model and epoch % eval_freq == 0:
            print('saving net...')
            torch.save(model, os.path.join('logs', checkpoints_folder) + '/model_epoch%d.pt' % epoch)
            print('saved')

        if sum_psnr > best_psnr or sum_ssim > best_ssim:
            best_psnr = sum_psnr
            best_ssim = sum_ssim
            print('saving best net...')
            torch.save(model, os.path.join('logs', checkpoints_folder) + '/model_best.pt')
            print('saved')

        f.close()


if __name__ == '__main__':
    main()
