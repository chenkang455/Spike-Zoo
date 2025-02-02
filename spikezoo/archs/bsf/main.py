import argparse
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import datetime
from datasets import datasets
from models.get_model import get_model
from utils import *
from metrics.psnr import *
from metrics.ssim import *
import lpips
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pprint
from models.bsf.dsft_convert import convert_dsft4

parser = argparse.ArgumentParser()
############################ Dataset Root ############################
parser.add_argument('--dataset_storage', type=str, default='ram')   ## ram or disk
parser.add_argument('--data-root', type=str, default='/dev/shm/rzhao/REDS120fps')
parser.add_argument('--half_reserve', type=int, default=2, help=' DSFT half reserve + 3ref + 4key + 3ref + DSFT half reserve')
############################ Training Params ############################
parser.add_argument('--arch', '-a', type=str, default='MEPF')
parser.add_argument('--batch-size', '-bs', type=int, default=8)
parser.add_argument('--learning-rate', '-lr', type=float, default=2e-4)
parser.add_argument('--train-res', '-tr', type=int, default=[128, 128], metavar='N', nargs='*')
parser.add_argument('--input-type', type=str, default='raw_spike', choices=['dsft', 'raw_spike'])
parser.add_argument('--epochs', '-ep', type=int, default=100)
parser.add_argument('--workers', '-j', type=int, default=8)
parser.add_argument('--pretrained', '-prt', type=str, default=None)
parser.add_argument('--start-epoch', '-sep', type=int, default=0)
parser.add_argument('--print-freq', '-pf', type=int, default=100)
parser.add_argument('--save-dir', '-sd', type=str, default='outputs')
parser.add_argument('--save-name', '-sn', type=str, default=None)
parser.add_argument('--vis-path', '-vp', type=str, default='vis')
parser.add_argument('--vis-name', '-vn', type=str, default='model1')
parser.add_argument('--eval-path', '-evp', type=str, default='eval_vis/model1')
parser.add_argument('--vis-freq', '-vf', type=int, default=20)
parser.add_argument('--eval', '-e', action='store_true')
parser.add_argument('--print_details', '-pd', action='store_true')
parser.add_argument('--milestones', default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], metavar='N', nargs='*')
parser.add_argument('--lr-scale-factor', '-lrsf', type=float, default=0.7)
parser.add_argument('--eval-interval', '-ei', type=int, default=5)
parser.add_argument('--no_imwrite', action='store_true', default=False)
parser.add_argument('--compile_model', '-cmpmd', action='store_true')
parser.add_argument('--seed', type=int, default=2728)
############################ Params about Dataset ############################
parser.add_argument('--alpha', type=float, default=0.7)
parser.add_argument('--eta_list', default=[1.00, 0.75, 0.50], type=float, metavar='N', nargs='*')
parser.add_argument('--gamma', type=int, default=60)
############################ About Optimizer ############################
parser.add_argument('--solver', type=str, default='Adam')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--beta', type=float, default=0.999)
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--test_eval', action='store_true')
parser.add_argument('--logs_file_name', type=str, default='bsf')

parser.add_argument('--loss_type', type=str, default='l1')
parser.add_argument('--dsft_convertor_type', type=int, default=4)

parser.add_argument('--no_dsft', action='store_true')
args = parser.parse_args()



##########################################################################################################
## configs
writer_root = 'logs/{:s}/'.format(args.logs_file_name)
os.makedirs(writer_root, exist_ok=True)
writer_path = writer_root + args.arch + '.txt'
writer = open(writer_path, 'a')

for k, v in vars(args).items():
    vv = pprint.pformat(v)
    ostr = '{:s} : {:s}'.format(k, vv)
    writer.write(ostr + '\n')

args.milestones = [int(m) for m in args.milestones]
ostr = 'milsones '
for mmm in args.milestones:
    ostr += '{:d} '.format(mmm)
writer.write(ostr + '\n')

n_iter = 0


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = compare_psnr(Img, Iclean, data_range=data_range)
    '''
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])
    '''
    return PSNR


def train(args, train_loader, model, optimizer, epoch, train_writer):
    ######################################################################
    ## Init
    global n_iter
    batch_time = AverageMeter(precision=3)
    data_time = AverageMeter(precision=3)
    
    losses = AverageMeter(precision=6, names=['Loss'])
    batch_psnr = AverageMeter(precision=4)

    model.train()
    
    end = time.time()
    
    ######################################################################
    ## Training Loop
    for ww, data in enumerate(train_loader, 0):
        spikes = [spk.float().cuda() for spk in data['spikes']]
        spks = torch.cat(spikes, dim=1)
        central_idx = 10*args.half_reserve + 30
        spks = spks[:, central_idx-30:central_idx+31]

        if not args.no_dsft:
            dsfts = [d.float().cuda() for d in data['dsft']]
            dsfts = torch.cat(dsfts, dim=1)
            central_idx = 10*args.half_reserve + 30
            dsfts = dsfts[:, central_idx-30:central_idx+31]

        images = [img.cuda() for img in data['images']]
        norm_fac = data['norm_fac'].unsqueeze_(dim=1).unsqueeze_(dim=1).unsqueeze_(dim=1).cuda().float()
        
        data_time.update(time.time() - end)

        if not args.no_dsft:
            dsft_dict = convert_dsft4(spike=spks, dsft=dsfts)

            input_dict = {
                'dsft_dict': dsft_dict,
                'spikes': spks,
            }

        gt = images[0]

        rec = model(input_dict=input_dict)

        rec = rec / norm_fac

        if args.loss_type == 'l1':
            loss = (rec - gt).abs().mean()
        elif args.loss_type == 'charbonnier':
            loss = torch.sqrt((rec - gt)**2 + 1e-6).mean()

        # record loss
        losses.update(loss)
        cur_batch_psnr = batch_PSNR(img=rec, imclean=gt, data_range=1.0)
        batch_psnr.update(cur_batch_psnr)

        if ww % 10 == 0:
            train_writer.add_scalar('loss', loss.item(), n_iter)
            train_writer.add_scalar('batch_psnr', cur_batch_psnr, n_iter)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        batch_time.update(time.time() - end)
        n_iter += 1

        if n_iter % args.vis_freq == 0:
            vis_img(args.vis_path, torch.clip(rec, 0, 1), args.arch)
        
        ostr = 'Epoch: [{:03d}] [{:04d}/{:04d}],  Iter: {:6d}  '.format(epoch+1, ww, len(train_loader), n_iter-1)
        ostr += 'Time: {},  Data: {}  '.format(batch_time, data_time)
        ostr += ' '.join(map('{:s} {:.4f} ({:.6f}) '.format, losses.names, losses.val, losses.avg))
        ostr += 'batch_PSNR {} '.format(batch_psnr)
        ostr += 'lr {:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
        if ww % args.print_freq == 0:
            writer.write(ostr + '\n')
        end = time.time()
    
    return


def validation(args, test_loader_list, model, lpips_function_dict):
    model.eval()
    
    for eta, test_loader in zip(args.eta_list, test_loader_list):
        cur_eval_root = osp.join(args.eval_path, args.arch, 'eta_{:.2f}'.format(eta))
        os.makedirs(cur_eval_root, exist_ok=True)

        global n_iter
        batch_time = AverageMeter()
        data_time = AverageMeter()
        metrics_name = ['PSNR', 'SSIM', 'LPIPS-A', 'LPIPS-V', 'AvgTime']
        metrics = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)
        
        for ww, data in enumerate(test_loader, 0):
            st1 = time.time()
            spks = torch.cat([spk.float().cuda() for spk in data['spikes']], dim=1)
            central_idx = 10*args.half_reserve + 30
            spks = spks[:, central_idx-30:central_idx+31]

            if not args.no_dsft:
                dsfts = torch.cat([d.float().cuda() for d in data['dsft']], dim=1)
                central_idx = 10*args.half_reserve + 30
                dsfts = dsfts[:, central_idx-30:central_idx+31]

            images = data['images']
            norm_fac = data['norm_fac'].unsqueeze_(dim=1).unsqueeze_(dim=1).unsqueeze_(dim=1).cuda().float()
            
            data_time.update(time.time() - st1)

            if not args.no_dsft:
                dsft_dict = convert_dsft4(spike=spks, dsft=dsfts)
                    
                input_dict = {
                    'dsft_dict': dsft_dict,
                    'spikes': spks,
                }

            with torch.no_grad():
                st = time.time()
                rec = model(input_dict=input_dict)
                mtime = time.time() - st

            rec = rec / norm_fac
            rec = torch.clip(rec, 0, 1)
            rec_np = torch2numpy255(rec)
            img_np = torch2numpy255(images[0])

            if not args.no_imwrite:
                cur_vis_path = osp.join(cur_eval_root, '{:03d}.png'.format(ww))
                cv2.imwrite(cur_vis_path, rec_np.astype(np.uint8))

            cur_psnr = calculate_psnr(rec_np, img_np)
            cur_ssim = calculate_ssim(rec_np, img_np)
            with torch.no_grad():
                cur_lpips_alex = lpips_function_dict['alex'](rec, images[0].cuda())
                cur_lpips_vgg  = lpips_function_dict['vgg'](rec, images[0].cuda())

            cur_metrics_list = [cur_psnr, cur_ssim, cur_lpips_alex.item(), cur_lpips_vgg.item() , mtime]
            metrics.update(cur_metrics_list)

        torch.cuda.empty_cache()
        ostr = 'Eta {:.2f}  ALL  '.format(eta) + '  '.join(map('{:s} {:.4f}'.format, metrics.names, metrics.avg))
        writer.write(ostr + '\n')

    return
    

def main():
    ##########################################################################################################
    # Set random seeds
    set_seeds(args.seed)

    # Create save path and logs
    timestamp1 = datetime.datetime.now().strftime('%m-%d')
    timestamp2 = datetime.datetime.now().strftime('%H%M%S')

    save_folder_name = 'a_{:s}_b{:d}_{:s}'.format(args.arch, args.batch_size, timestamp2)

    save_path = osp.join(args.save_dir, timestamp1, save_folder_name)
    make_dir(save_path)
    ostr = '=>Save path: ' + save_path
    writer.write(ostr + '\n')
    # print('=>Save path: ', save_path)
    train_writer = SummaryWriter(save_path)
    
    make_dir(args.vis_path)
    make_dir(args.eval_path)    

    model = None
    optimizer = None

    ##########################################################################################################
    ## Create model
    print(args.arch)
    model = get_model(args)

    if args.compile_model and (torch.__version__ >= '2.0.0'):
        ostr = 'Start compile the model'
        writer.write(ostr + '\n')
        st = time.time()
        torch.compile(model)
        ostr = 'Finish compiling the model  Time {:.2f}s'.format(time.time() - st)
        writer.write(ostr + '\n')

    if args.pretrained != None:
        network_data = torch.load(args.pretrained)
        ostr = '=> using pretrained model {:s}'.format(args.pretrained)
        writer.write(ostr + '\n')
        ostr = '=> model params: {:.6f}M'.format(model.num_parameters()/1e6)
        writer.write(ostr + '\n')
        model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        model.load_state_dict(network_data)
    else:
        network_data = None
        ostr = '=> train from scratch'
        writer.write(ostr + '\n')
        model.init_weights()
        ostr = '=> model params: {:.6f}M'.format(model.num_parameters()/1e6)
        writer.write(ostr + '\n')
        model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()

    cudnn.benchmark = True

    ##########################################################################################################
    ## Create Optimizer
    assert(args.solver in ['Adam', 'SGD'])
    ostr = '=> settings {:s} solver'.format(args.solver)
    writer.write(ostr + '\n')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


    ##########################################################################################################
    ## Dataset
    train_set = datasets.sreds_train(args)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        drop_last=False,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    test_loader_list = []
    for eta in args.eta_list:
        test_set = datasets.sreds_test_small(args, eta=eta)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            drop_last=False,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
        )
        test_loader_list.append(test_loader)

    ##########################################################################################################
    ## For LPIPS
    loss_fn_alex = lpips.LPIPS(net='alex').cuda() # closer to "traditional" perceptual loss, when used for optimization
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization
    lpips_function_dict = {'alex': loss_fn_alex, 'vgg': loss_fn_vgg}

    ##########################################################################################################
    ## Train or Evaluate
    if args.test_eval:
        validation(
            args=args, 
            test_loader_list=test_loader_list, 
            model=model,
            lpips_function_dict=lpips_function_dict,
            )
        return

    epoch = args.start_epoch
    while(True):
        train(
            args=args,
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_writer=train_writer,
        )
        epoch += 1

        # scheduler can be added here
        if epoch in args.milestones:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * args.lr_scale_factor

        # save model
        if epoch % 5 == 0:
            model_save_name = '{:s}_epoch{:03d}.pth'.format(args.arch, epoch)
            torch.save(model.state_dict(), osp.join(save_path, model_save_name))
        
        # if epoch % 5 == 0:
        if epoch % args.eval_interval == 0:
            validation(
                args=args, 
                test_loader_list=test_loader_list, 
                model=model,
                lpips_function_dict=lpips_function_dict,
                )


        if epoch >= args.epochs:
            break


if __name__ == '__main__':
    main()
