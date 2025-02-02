import argparse
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import pprint
import datetime
from configs.yml_parser import *
from datasets.dataset_sreds import *
from models.networks import *
from utils import *
from metrics.psnr import *
from metrics.ssim import *
import lpips
from losses import *
from models.Vgg19 import *
from spikingjelly.clock_driven import functional


parser = argparse.ArgumentParser()
parser.add_argument('--data-root', '-dr', type=str, default='/home/data/rzhao/REDS_dataset/REDS120fps')
parser.add_argument('--arch', '-a', type=str, default='SSIR')
parser.add_argument('--batch-size', '-b', type=int, default=8)
parser.add_argument('--learning-rate', '-lr', type=float, default=4e-4)
parser.add_argument('--configs', '-cfg', type=str, default='./configs/SSIR.yml')
parser.add_argument('--epochs', '-ep', type=int, default=100)
parser.add_argument('--epoch-size', '-es', type=int, default=1000)
parser.add_argument('--workers', '-j', type=int, default=8)
parser.add_argument('--pretrained', '-prt', type=str, default=None)
parser.add_argument('--start-epoch', '-sep', type=int, default=0)
parser.add_argument('--print-freq', '-pf', type=int, default=200)
parser.add_argument('--save-dir', '-sd', type=str, default='outputs')
parser.add_argument('--save-name', '-sn', type=str, default=None)
parser.add_argument('--vis-path', '-vp', type=str, default='vis')
parser.add_argument('--vis-name', '-vn', type=str, default='SSIR')
parser.add_argument('--eval_path', '-evp', type=str, default='eval_vis')
parser.add_argument('--vis-freq', '-vf', type=int, default=20)
parser.add_argument('--eval', '-e', action='store_true')
parser.add_argument('--w_per', '-wper', type=float, default=0.2)
parser.add_argument('--print_details', '-pd', action='store_true')
parser.add_argument('--milestones', default=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], metavar='N', nargs='*')
parser.add_argument('--lr-scale-factor', '-lrsf', type=float, default=0.7)
parser.add_argument('--eval-interval', '-ei', type=int, default=5)
parser.add_argument('--save-interval', '-si', type=int, default=5)
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
    losses_name = ['rec_loss', 'per_loss', 'all_loss']
    losses = AverageMeter(precision=6, i=len(losses_name), names=losses_name)
    model.train()
    end = time.time()
    
    vgg19 = Vgg19(requires_grad=False).cuda()
    if torch.cuda.device_count() > 1:
        vgg19 = nn.DataParallel(vgg19, list(range(torch.cuda.device_count())))


    ######################################################################
    ## Training Loop
    
    for ww, data in enumerate(train_loader, 0):
        
        if ww >= args.epoch_size:
            return

        spikes = [spk.cuda() for spk in data['spikes']]
        images = [img.cuda() for img in data['images']]
        data_time.update(time.time() - end)

        cur_spks = torch.cat(spikes, dim=1)

        rec_loss = 0.0
        per_loss = 0.0
        
        for jj in range(1, 1+cfg['model']['seq_len']):
            x = cur_spks[:, jj*20-20 : jj*20+21]

            gt = images[jj]

            out_list = model(x)
            rec_list = [torch.clip(out, 0, 1) for out in out_list]

            # if jj > 1+2:
            if jj >= 2:
                rec_loss += compute_l1_loss(rec_list, gt) / (cfg['model']['seq_len'] - 2)
                if cfg['train']['w_per'] > 0:
                    per_loss += cfg['train']['w_per'] * compute_per_loss_single(rec_list[-1], gt, vgg19) / (cfg['model']['seq_len'] - 2)
                else:
                    per_loss = torch.tensor([0.0]).cuda()
                
        all_loss = rec_loss + per_loss
        
        # record loss
        losses.update([rec_loss.item(), per_loss.item(), all_loss.item()])
        train_writer.add_scalar('rec_loss', rec_loss.item(), n_iter)
        train_writer.add_scalar('per_loss', per_loss.item(), n_iter)
        train_writer.add_scalar('total_loss', all_loss.item(), n_iter)

        ## compute gradient and optimize
        all_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        functional.reset_net(model)
        
        batch_time.update(time.time() - end)
        end = time.time()
        n_iter += 1

        if n_iter % cfg['train']['vis_freq'] == 0:
            vis_img(cfg['train']['vis_path'], rec_list[-1], cfg['train']['vis_name'])
        
        if ww % cfg['train']['print_freq'] == 0:
            out_str = 'Epoch: [{:d}] [{:d}/{:d}],  Iter: {:d}  '.format(epoch, ww, len(train_loader), n_iter-1)
            out_str += 'Time: {},  Data: {}  '.format(batch_time, data_time)
            out_str += ' '.join(map('{:s} {:.4f} ({:.6f}) '.format, losses.names, losses.val, losses.avg))
            out_str += 'lr {:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            print(out_str)
        
        end = time.time()
    
    return


def validation(cfg, test_loader, model):
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    metrics_name = ['PSNR', 'SSIM', 'LPIPS', 'AvgTime']
    all_metrics = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)

    model.eval()

    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()

    for ww, data in enumerate(test_loader, 0):
        st1 = time.time()
        spikes = torch.cat([spk.cuda() for spk in data['spikes']], dim=1)
        images = data['images']
        data_time.update(time.time() - st1)

        seq_metrics = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)

        seq_len = len(data['spikes']) - 2

        rec = []
        for jj in range(1, 1+seq_len):
            x = spikes[:, jj*20-20 : jj*20+21]
            
            gt = images[jj].cuda()

            with torch.no_grad():
                st = time.time()
                out = model(x)
                mtime = time.time() - st
            rec = torch.clip(out, 0, 1)

            cur_rec = torch2numpy255(rec)    
            cur_gt = torch2numpy255(gt)

            if not args.no_imwrite:
                cur_vis_path = osp.join(args.eval_path, '{:03d}_{:03d}.png'.format(ww, jj))
                cv2.imwrite(cur_vis_path, cur_rec.astype(np.uint8))

            cur_psnr = calculate_psnr(cur_rec, cur_gt)
            cur_ssim = calculate_ssim(cur_rec, cur_gt)
            with torch.no_grad():
                cur_lpips = loss_fn_vgg(rec, gt)

            cur_metrics_list = [cur_psnr, cur_ssim, cur_lpips.item(), mtime]

            all_metrics.update(cur_metrics_list)
            seq_metrics.update(cur_metrics_list)
        
        functional.reset_net(model)
            
        if args.print_details:
            ostr = 'Data{:02d}  '.format(ww) + ' '.join(map('{:s} {:.4f} '.format, seq_metrics.names, seq_metrics.avg))
            print(ostr)
            print()

    
    ostr = 'All  ' + ' '.join(map('{:s} {:.4f} '.format, all_metrics.names, all_metrics.avg))
    print(ostr)

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
        save_folder_name = 'b{:d}_{:s}_{:s}'.format(args.batch_size, timestamp2, args.save_name)

    save_path = osp.join(args.save_dir, timestamp1, save_folder_name)
    print('save path: ', save_path)
    make_dir(save_path)
    make_dir(args.vis_path)
    make_dir(args.eval_path)    

    train_writer = SummaryWriter(save_path)

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
        validation(cfg=cfg, test_loader=test_loader, model=model)
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
            
            # if epoch % 5 == 0:
            if epoch % args.eval_interval == 0:
                validation(cfg=cfg, test_loader=test_loader, model=model)

            if epoch >= cfg['train']['epochs']:
                break

if __name__ == '__main__':
    main()
