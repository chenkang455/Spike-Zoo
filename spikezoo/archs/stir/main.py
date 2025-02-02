import argparse
import os
import os.path as osp
import shutil
import time
import numpy as np
import torch
import torch.optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from thop import profile
import pprint
import datetime
import lpips
# import pyiqa
# import cpbd
import imageio
from configs.yml_parser import *
from datasets.dataset_sreds import *
from configs.utils import *
from metrics.psnr import *
from metrics.ssim import *
from metrics.losses import *
from models.Vgg19 import *
from spikingjelly.clock_driven import functional

os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from models.networks_STIR import *

parser = argparse.ArgumentParser()

parser.add_argument('--data_root', '-dr', type=str, default='/data/local_userdata/fanbin/REDS_dataset/REDS120fps')
parser.add_argument('--arch', '-a', type=str, default='STIR')
parser.add_argument('--batch_size', '-b', type=int, default=8)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--configs', '-cfg', type=str, default='./configs/STIR.yml')
parser.add_argument('--epochs', '-ep', type=int, default=100)
parser.add_argument('--epoch_size', '-es', type=int, default=1000)
parser.add_argument('--workers', '-j', type=int, default=8)
parser.add_argument('--pretrained', '-prt', type=str, default=None)
parser.add_argument('--start_epoch', '-sep', type=int, default=0)
parser.add_argument('--print_freq', '-pf', type=int, default=1)
parser.add_argument('--save_dir', '-sd', type=str, default='ckpt_outputs')
parser.add_argument('--save_name', '-sn', type=str, default='t1')
parser.add_argument('--vis_path', '-vp', type=str, default='vis_train')
parser.add_argument('--vis_name', '-vn', type=str, default='STIR_train')
parser.add_argument('--eval_path', '-evp', type=str, default='vis_eval')
parser.add_argument('--vis_freq', '-vf', type=int, default=200)
parser.add_argument('--eval', '-e', action='store_true')
parser.add_argument('--w_per', '-wper', type=float, default=0.2)
parser.add_argument('--print_details', '-pd', action='store_true')
parser.add_argument('--milestones', default=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], metavar='N', nargs='*')
parser.add_argument('--lr_scale_factor', '-lrsf', type=float, default=0.7)
parser.add_argument('--eval_interval', '-ei', type=int, default=5)
parser.add_argument('--save_interval', '-si', type=int, default=5)
parser.add_argument('--no_imwrite', action='store_true', default=False)
args = parser.parse_args()

args.milestones = [int(m) for m in args.milestones]
print('milstones', args.milestones)

cfg_parser = YAMLParser(args.configs)
cfg = cfg_parser.config

cfg['data']['root'] = args.data_root
cfg = add_args_to_cfg(cfg, args, ['batch_size', 'arch', 'learning_rate', 'configs', 'epochs', 'epoch_size', 'workers', 'pretrained', 'start_epoch', 
                        'print_freq', 'save_dir', 'save_name', 'vis_path', 'vis_name', 'eval_path', 'vis_freq', 'w_per'])

n_iter = 0


def train(cfg, train_loader, model, optimizer, epoch, train_writer):
    ######################################################################
    ## Init
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_name = ['rec_loss', 'per_loss', 'mulscl_loss', 'all_loss']
    losses = AverageMeter(precision=6, i=len(losses_name), names=losses_name)
    model.train()
    torch.cuda.synchronize()
    end = time.time()
    
    vgg19 = Vgg19(requires_grad=False).cuda()
    if torch.cuda.device_count() > 1:
        vgg19 = nn.DataParallel(vgg19, list(range(torch.cuda.device_count())))

    loss_fn_tv2 = VariationLoss(nc=2).cuda()
    downsampleX2 = nn.AvgPool2d(2, stride=2).cuda()
    loss_fn_L1 = L1Loss()
    
    ######################################################################
    ## Training Loop
    
    for ww, data in enumerate(train_loader, 0):
        
        if ww >= args.epoch_size:
            return

        spikes = [spk.cuda() for spk in data['spikes']]
        images = [img.cuda() for img in data['images']]
        torch.cuda.synchronize()
        data_time.update(time.time() - end)

        cur_spks = torch.cat(spikes, dim=1)
        
        rec_loss = 0.0
        per_loss = 0.0
        loss_L1_multiscale = 0.0
        loss_rep_est = 0.0

        seq_len = len(data['spikes']) - 3###corres 23th img GT

        for jj in range(1, 1+seq_len):
            x = cur_spks[:, jj*20-11 : jj*20+50]

            img_gt = images[jj+1]
            
            img_pred_0, Fs_lv_0, Fs_lv_1, Fs_lv_2, Fs_lv_3, Fs_lv_4, Est = model(x)
            pred_F = [Fs_lv_0]
            pred_F.append(Fs_lv_1)
            pred_F.append(Fs_lv_2)
            pred_F.append(Fs_lv_3)
            pred_F.append(Fs_lv_4)
            
            # if jj > 1+2:
            if jj >= 2:
                rec_loss += loss_fn_L1(img_pred_0, img_gt, mean=True) / (seq_len - 1)
                if cfg['train']['w_per'] > 0:
                    per_loss += cfg['train']['w_per'] * compute_per_loss_single(img_pred_0, img_gt, vgg19) / (seq_len - 1)
                else:
                    per_loss = torch.tensor([0.0]).cuda()
                
                pyr_weights = [1.0, 0.5, 0.25, 0.25, 0.25] 
                num=5  #pyramid: 3, 4, 5
                for l in range(1, num):
                    img_gt = downsampleX2(img_gt)
                    loss_L1_multiscale += pyr_weights[l] * loss_fn_L1(pred_F[l][0], img_gt, mean=True) / (num-1) / (seq_len - 1)
        all_loss = rec_loss + per_loss + loss_L1_multiscale #+ loss_rep_est
        
        # record loss
        losses.update([rec_loss.item(), per_loss.item(), loss_L1_multiscale.item(), all_loss.item()])
        train_writer.add_scalar('rec_loss', rec_loss.item(), n_iter)
        train_writer.add_scalar('per_loss', per_loss.item(), n_iter)
        train_writer.add_scalar('mulscl_loss', loss_L1_multiscale.item(), n_iter)
        train_writer.add_scalar('total_loss', all_loss.item(), n_iter)

        ## compute gradient and optimize
        all_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        functional.reset_net(model)
        
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        torch.cuda.synchronize()
        end = time.time()
        n_iter += 1

        if n_iter % cfg['train']['vis_freq'] == 0:
            vis_img(cfg['train']['vis_path'], img_pred_0, cfg['train']['vis_name'])
        
        if ww % cfg['train']['print_freq'] == 0:
            out_str = 'Epoch: [{:d}] [{:d}/{:d}],  Iter: {:d}  '.format(epoch, ww, len(train_loader), n_iter-1)
            out_str += ' '.join(map('{:s} {:.4f} ({:.6f}) '.format, losses.names, losses.val, losses.avg))
            out_str += 'lr {:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            print(out_str)
        
        torch.cuda.synchronize()
        end = time.time()
    
    return


def validation(cfg, test_loader, model, epoch, auto_save_path):
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    metrics_name = ['PSNR', 'SSIM', 'LPIPS', 'AvgTime']
    all_metrics = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)

    timestamp1 = datetime.datetime.now().strftime('%m-%d')
    timestamp2 = datetime.datetime.now().strftime('%H%M%S')

    model.eval()

    #lpips_loss = pyiqa.create_metric('lpips').cuda()
    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()
    
    padder = InputPadder(dims=(720, 1280))

    for ww, data in enumerate(test_loader, 0):
        torch.cuda.synchronize()
        st1 = time.time()
        spikes = torch.cat([spk.cuda() for spk in data['spikes']], dim=1)
        images = data['images']
        torch.cuda.synchronize()
        data_time.update(time.time() - st1)

        seq_metrics = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)

        seq_len = len(data['spikes']) - 3###corres 23th img GT

        pred_gif=[]
        gt_gif=[]

        for jj in range(1, 1+seq_len):
            x = spikes[:, jj*20-11 : jj*20+50]
            x = padder.pad(x)[0]
            
            gt = images[jj+1].cuda()

            with torch.no_grad():
                torch.cuda.synchronize()
                st = time.time()
                
                out = model(x)
                torch.cuda.synchronize()
                mtime = time.time() - st
            rec = padder.unpad(out)
            
            cur_rec = torch2numpy255(rec)
            cur_gt = torch2numpy255(gt)

            if not args.no_imwrite and args.eval:
                save_path = osp.join(args.eval_path, timestamp1)
                make_dir(save_path)
                cur_vis_path = osp.join(save_path, '{:03d}_{:03d}.png'.format(ww, jj))
                cv2.imwrite(cur_vis_path, cur_rec.astype(np.uint8))

                pred_gif.append(cur_rec.astype(np.uint8))
                gt_gif.append(cur_gt.astype(np.uint8))

            cur_psnr = calculate_psnr(cur_rec, cur_gt)
            cur_ssim = calculate_ssim(cur_rec, cur_gt)
            with torch.no_grad():
                cur_lpips = loss_fn_vgg(rec, gt)

            cur_metrics_list = [cur_psnr, cur_ssim, cur_lpips.item(), mtime]
            if args.eval:
                print("[Seq%d, %d-th image]: PSNR:%.4f SSIM:%.4f LPIPS:%.4f Time:%.4f" % (ww, jj+2, cur_psnr, cur_ssim, cur_lpips.item(), mtime))

            all_metrics.update(cur_metrics_list)
            seq_metrics.update(cur_metrics_list)
        
        functional.reset_net(model)
            
        if args.print_details:
            print('\n')
            ostr = 'Data{:02d}  '.format(ww) + ' '.join(map('{:s} {:.4f} '.format, seq_metrics.names, seq_metrics.avg))
            print(ostr)
            print()
    
    ostr = 'All  ' + ' '.join(map('{:s} {:.4f} '.format, all_metrics.names, all_metrics.avg))
    print(ostr)

    if args.eval:
        print('\n')
    else:
        print('Test current epoch\n')
        f_metric_avg=open(os.path.join(auto_save_path, 'ckpt_'+args.save_name+'_metric_avg.txt'), 'a+')#Save the files next to the last line
        f_metric_avg.write('%s  ' % (str(epoch).zfill(2)))
        f_metric_avg.write(ostr)
        f_metric_avg.write('\n')
        f_metric_avg.close()

    return
    

def main():
    ##########################################################################################################
    # Set random seeds
    set_seeds(cfg['seed'])

    # Create save path and logs
    timestamp1 = datetime.datetime.now().strftime('%m-%d')
    timestamp2 = datetime.datetime.now().strftime('%H%M%S')
    if args.save_name == None:
        save_folder_name = 'b{:d}_{:s}'.format(args.batch_size, timestamp2)
    else:
        save_folder_name = 'b{:d}_{:s}_{:s}'.format(args.batch_size, args.save_name, timestamp2)

    save_path = osp.join(args.save_dir, timestamp1, save_folder_name)
    print('save path: ', save_path)
    if args.eval:
        print('\n')
    else:
        make_dir(save_path)
        #auto save test results during training
        f_metric_avg=open(os.path.join(save_path, 'ckpt_'+args.save_name+'_metric_avg.txt'), 'w')
        f_metric_avg.close()

    make_dir(args.vis_path)
    make_dir(args.eval_path)
    
    train_writer = SummaryWriter(save_path)

    if args.eval:
        shutil.rmtree(save_path)
        print('remove path: ', save_path)

    cfg_str = pprint.pformat(cfg)
    print('=> configurations: ')
    print(cfg_str)
    
    ##########################################################################################################
    ## Create model
    model = eval(args.arch)()

    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print('=> using pretrained model {:s}'.format(args.pretrained))
        model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        model.load_state_dict(network_data)
    else:
        network_data = None
        print('=> train from scratch')
        model.init_weights()
        print('=> model params: {:.6f}M'.format(model.num_parameters()/1e6))
        model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()

    cudnn.benchmark = True

    ##########################################################################################################
    ## Create Optimizer
    cfgopt = cfg['optimizer']
    cfgmdl = cfg['model']
    assert(cfgopt['solver'] in ['Adam', 'SGD'])
    print('=> settings {:s} solver'.format(cfgopt['solver']))
    
    param_groups = [{'params': model.parameters(), 'weight_decay': cfgmdl['flow_weight_decay']}]
    if cfgopt['solver'] == 'Adam':
        optimizer = torch.optim.Adam(param_groups, args.learning_rate, betas=(cfgopt['momentum'], cfgopt['beta']))
    elif cfgopt['solver'] == 'SGD':
        optimizer = torch.optim.SGD(param_groups, args.learning_rate, momentum=cfgopt['momentum'])

    ##########################################################################################################
    ## Dataset
    train_set = sreds_train(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        drop_last=False,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['workers'],
        # pin_memory=True
    )

    test_set = sreds_test(cfg)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        drop_last=False,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['train']['workers']
    )

    ##########################################################################################################
    ## Train or Evaluate
    if args.eval:
        validation(cfg=cfg, test_loader=test_loader, model=model, epoch=0, auto_save_path=save_path)
    else:
        epoch = cfg['train']['start_epoch']
        while(True):
            train(
                cfg=cfg,
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_writer=train_writer
            )
            epoch += 1

            # scheduler can be added here
            if epoch in args.milestones:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * args.lr_scale_factor

            # save model
            if epoch % args.save_interval == 0:
                model_save_name = '{:s}_epoch{:03d}.pth'.format(cfg['model']['arch'], epoch)
                torch.save(model.state_dict(), osp.join(save_path, model_save_name))
            
            if epoch % args.eval_interval == 0:
                validation(cfg=cfg, test_loader=test_loader, model=model, epoch=epoch, auto_save_path=save_path)

            if epoch >= cfg['train']['epochs']:
                break

if __name__ == '__main__':
    main()
