import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
from utils import *
# supported methods 
from compare_zoo.Spk2ImgNet.nets import SpikeNet
from compare_zoo.WGSE.dwtnets import Dwt1dResnetX_TCN
from compare_zoo.SSML.model import DoubleNet
from compare_zoo.TFP.tfp_model import TFP
from compare_zoo.TFI.tfi_model import TFI
from compare_zoo.TFSTP.tfstp_model import TFSTP
from compare_zoo.SPCS_Net.model import SPCS_Net
from compare_zoo.SpikeFormer.Model.SpikeFormer import SpikeFormer
from compare_zoo.RSIR.structure import MainDenoise
from compare_zoo.RSIR.utils import cal_para
from compare_zoo.BSF.models.bsf.bsf import BSF
from compare_zoo.LRN.model import LRN

# supported datasets 
from dataset import SpikeData_REDS,SpikeData_Real,SpikeData_UHSR

from tqdm import tqdm
from metrics import compute_img_metric,compute_img_metric_single
from thop import profile
import torch.nn.functional as F
import time
import argparse

# network loading
def network_construct(method_name):
    global spkrecon_net
    # CNN-based
    if method_name == 'Spk2ImgNet':
        spkrecon_net = SpikeNet(
            in_channels=13, features=64, out_channels=1, win_r=6, win_step=7
        )
        load_path = "compare_zoo/Spk2ImgNet/model_061.pth"
        load_net = True
    elif method_name == 'WGSE':
        yl_size,yh_size = 15 ,[28, 21, 18, 16, 15]
        spkrecon_net = Dwt1dResnetX_TCN(
            wvlname="db8", J=5, yl_size=yl_size, yh_size=yh_size, num_residual_blocks=3, norm=None, ks=3, store_features=True
        )
        load_path = "compare_zoo/WGSE/model_best.pt"
        load_net = True
    elif method_name.startswith("SPCS"):
        spkrecon_net = SPCS_Net()
        load_path = f"models/{method_name}.pth"
        load_net = True
    elif method_name == "SSML":
        spkrecon_net = DoubleNet()
        load_path = "compare_zoo/SSML/fin3g-best-lucky.pt"
        load_net = True
    elif method_name == 'SpikeFormer':
        spkrecon_net = SpikeFormer(
        inputDim=65,
        dims=(32, 64, 160, 256),  
        heads=(1, 2, 5, 8),
        ff_expansion=(8, 8, 4, 4),  
        reduction_ratio=(8, 4, 2, 1),
        num_layers=2, 
        decoder_dim=256,
        out_channel=1)  
        load_path = "compare_zoo/SpikeFormer/best.pth"
        load_net = True
    elif method_name == 'RSIR':
        spkrecon_net = MainDenoise()
        load_path = "compare_zoo/RSIR/model_best.pth"
        load_net = True
    elif method_name == 'BSF':
        spkrecon_net = BSF()
        load_path = "compare_zoo/BSF/ckpt/bsf.pth"
        load_net = True
    elif method_name == 'LRN':
        spkrecon_net = LRN()
        # load_path = "compare_zoo/LRN/caltech_cls_xxx_25.pth"
        load_path = "compare_zoo/LRN/caltech_cls_25.pth"
        # load_path = "compare_zoo/LRN/caltech_cls_notfi_25.pth"
        load_net = True
    # Explained
    elif method_name == "TFP":
        spkrecon_net = TFP()
        load_net = False
    elif method_name == "TFI":
        spkrecon_net = TFI()
        load_net = False
    elif method_name == "TFSTP":
        if img_size == '250_400':
            spkrecon_net = TFSTP(spike_h = 250, spike_w = 400)
        elif img_size == '224_224':
            spkrecon_net = TFSTP(spike_h = 224, spike_w = 224)
        load_net = False
        
    # load the network 
    spkrecon_net = spkrecon_net.cuda()
    if load_net == True:
        load_network(load_path,spkrecon_net)
    
# spike length check
def spike_check(spike_raw,method_name,spike_idx):
    if method_name in ['Spk2ImgNet','WGSE','SSML']:
        if len(spike_raw[0]) < 41:
            raise ValueError(f"Network requires spike length >= 41.")
        elif spike_idx - 20 < 0 or spike_idx + 21 >= len(spike_raw[0]):
            raise ValueError(f"Spike idx wrong.")
    elif method_name == 'SpikeFormer':
        if len(spike_raw[0]) < 65:
            raise ValueError(f"Network requires spike length >= 65.")
        elif spike_idx - 32 < 0 or spike_idx + 33 >= len(spike_raw[0]):
            raise ValueError(f"Spike idx wrong.")
    elif method_name == 'BSF':
        if len(spike_raw[0]) < 61:
            raise ValueError(f"Network requires spike length >= 61.")
        elif spike_idx - 30 < 0 or spike_idx + 31 >= len(spike_raw[0]):
            raise ValueError(f"Spike idx wrong.")
    elif method_name == 'LRN':
        if len(spike_raw[0]) < 200:
            raise ValueError(f"Network requires spike length >= 200.")
        elif spike_idx - 100 < 0 or spike_idx + 99 >= len(spike_raw[0]):
            raise ValueError(f"Spike idx wrong.")

# spike pre-process
def spike_process(spike_raw,method_name,spike_idx = -1):
    # input
    if spike_idx == -1:
        spike_idx = len(spike_raw[0]) // 2

    # check input
    spike_check(spike_raw,method_name,spike_idx)

    # spike length roi
    if method_name in ['Spk2ImgNet','WGSE','SSML']:
        spike = spike_raw[:,spike_idx - 20:spike_idx + 21]
    elif method_name == 'SpikeFormer':
        spike = spike_raw[:,spike_idx - 32:spike_idx + 33]
    elif method_name == 'BSF':
        spike = spike_raw[:,spike_idx - 30:spike_idx + 31]
    elif method_name == 'RSIR':
        spike = spike_raw[:,spike_idx - 80:spike_idx + 80]
    elif method_name in ['TFP','TFI','TFSTP','LRN']:
        spike  = spike_raw[:,spike_idx - 100:spike_idx + 100]
        
    # spike size roi
    if img_size == '250_400': 
        if method_name == 'Spk2ImgNet':
            spike = torch.cat([spike,spike[:,:,-2:]],dim = 2)
        elif method_name == 'SpikeFormer':
            spike  = F.pad(spike, pad=(0, 0, 3, 3, 0, 0), mode='constant', value=0)
    elif img_size == '224_224':
        if method_name == 'RSIR':
            spike  = F.pad(spike, pad=(88, 88, 13, 13, 0, 0), mode='constant', value=0)
    
    # input spike process
    if method_name == 'SpikeFormer':
        spike = 2 * spike - 1
    elif method_name == 'LRN':
        spike = torch.sum(spike.reshape(-1,50,4,spike.shape[2],spike.shape[3]), axis=2)
        
    # cuda
    if method_name != 'TFSTP':
        spike = spike.cuda()
    return spike

# network output
def network_output(spike,method_name,spike_idx = -1):
    # spike process
    spike = spike_process(spike,method_name,spike_idx)
    
    # output
    if method_name.startswith('SPCS'):
        iter_num = int(method_name.split('.')[0].split('_')[-1])
        recon_img = spkrecon_net(spike,iter_num = iter_num)
    elif method_name in ['TFP','TFI']:
        recon_img = spkrecon_net(spike,max_search_half_window = spike.shape[1] // 2 - 1)
    elif method_name == 'RSIR':
        Q, Nd, Nl = cal_para()
        q = torch.from_numpy(Q).cuda()
        nd = torch.from_numpy(Nd).cuda()
        nl = torch.from_numpy(Nl).cuda()
        for i in range(10):
            noisy_img = torch.ones([1, 1, spike.shape[2], spike.shape[3]], dtype=torch.float32).cuda()
            noisy_img[0, 0, :, :] = spike[:,16*i:16*(i+1)].mean(dim=1).float()
            if i == 0:
                input = noisy_img
            else:
                input = torch.cat([noisy_img, fusion_out], dim=1)
            fpn_denoise, img_true, fusion_out, denoise_out, refine_out, ft_denoise_out_d0, fgt_d0 = spkrecon_net(input, noisy_img, q, nd, nl)
            recon_img = fusion_out
    else:
        recon_img = spkrecon_net(spike)
        
    # post-process
    if img_size == '250_400': 
        if method_name == 'Spk2ImgNet':
            if dataset_cls == 'REDS':
                recon_img =  torch.clamp(recon_img / 0.6, 0, 1)
            recon_img = recon_img[:,:,:250,:]
        elif method_name == 'SpikeFormer':
            recon_img = recon_img[:,:,3:-3,:]
    elif img_size == '224_224':
        if method_name == 'RSIR':
            recon_img = recon_img[:,:,125 - 112:125 + 112,200 - 112:200 + 112]
        
    # normalize  
    if method_name in ['TFP','TFI']:
        pass
    elif method_name == 'TFSTP':
        recon_img = recon_img.cuda().float()
        # recon_img = recon_img.clip(0,0.15)
    return recon_img


# establish metric
def metric_construct():
    global metrics
    metrics = {}
    for method_name in method_list:
        metrics[method_name] = {}  
        for metric_name in metric_list:
            metrics[method_name][metric_name] = AverageMeter()

# update metric dictionary
def metric_update(method_name,recon_img,sharp):
    for key in metric_list:
        if key in metric_pair:
            metrics[method_name][key].update(compute_img_metric(recon_img,sharp,key))
        elif key in metric_single:
            metrics[method_name][key].update(compute_img_metric_single(recon_img,key))
        else:
            ValueError("Key not found.")

# calculate metric
def metric_calculate(method_name):
    logger.info(f"Method {method_name} estimating metrics...")
    network_construct(method_name)
    for batch_idx, (spike,sharp) in enumerate(tqdm(dataloader)):
        # Test 5 imgs for each real-world dataset
        if dataset_cls == 'Real':
            spk_length = spike.shape[1]
            for spk_idx in np.linspace(100,spk_length-100,5):
                spk_idx = int(spk_idx)
                recon_img = network_output(spike,method_name,spk_idx)
                metric_update(method_name,recon_img,sharp)
        elif dataset_cls in ['UHSR','REDS']:
            recon_img = network_output(spike,method_name)
            metric_update(method_name,recon_img,sharp)

# logger.info metric
def metric_print():
    for method_name in method_list:
        re_msg = method_name + '-----'
        for metric_name in metric_list:
            re_msg += metric_name + ": " + "{:.4f}".format(metrics[method_name][metric_name].avg) + "  "
        logger.info(re_msg)
        
# output img
def img_reconstruct(spike,method_name,spike_idx = -1,nor = False):
    logger.info(f"Method {method_name} reconstructing img...")
    network_construct(method_name)
    recon_img = network_output(spike,method_name,spike_idx)
    recon_img = recon_img.detach().cpu()
    recon_img = recon_img[0,0].numpy()
    if nor == True:
        recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min()) 
    if img_size == '224_224':
        recon_img = recon_img[30:-30,10:-10]
    recon_img = recon_img * 255
    cv2.imwrite(f'imgs/{spike_name}/{method_name}.png',recon_img)
    return recon_img

# parameters, latency and flops calculation
def params_calculate(method_name):
    global img_size
    logger.info(f"Method {method_name} estimating parameters and flops...")
    network_construct(method_name)
    # input
    spike = torch.zeros((1,200,250,400)).cuda()
    img_size = '250_400'
    spike = spike_process(spike,method_name)
    # output
    total = sum(p.numel() for p in spkrecon_net.parameters())
    if method_name.startswith('SPCS'):
        iter_num = int(method_name.split('.')[0].split('_')[-1])
        flops, _ = profile((spkrecon_net), inputs=(spike,iter_num))
    elif method_name == 'RSIR':
        Q, Nd, Nl = cal_para()
        q = torch.from_numpy(Q).cuda()
        nd = torch.from_numpy(Nd).cuda()
        nl = torch.from_numpy(Nl).cuda()
        noisy_img = torch.ones([1, 1, spike.shape[2], spike.shape[3]], dtype=torch.float32).cuda()
        flops, _ = profile((spkrecon_net), inputs=(spike,noisy_img, q, nd, nl))
    else:
        flops, _ = profile((spkrecon_net), inputs=(spike,))
    # test_time
    start_time = time.time()
    for _ in range(100):
        if method_name.startswith('SPCS'):
            iter_num = int(method_name.split('.')[0].split('_')[-1])
            spkrecon_net(spike,iter_num)
        elif method_name == 'RSIR':
            spkrecon_net(spike,noisy_img, q, nd, nl)
        else:
            spkrecon_net(spike)
    latency = (time.time() - start_time) / 100
    re_msg = (
        "Total params: %.4fM" % (total / 1e6),
        "FLOPs=" + str(flops / 1e9) + '{}'.format("G"),
        "Latency: {:.6f} seconds".format(latency)
    )    
    logger.info(re_msg)

# data load
def dataset_load(cls = 'spike'):
    global spike,dataloader,spike_name,spike_length,img_size
    # spike input 
    if cls == 'spike' or opt.test_imgs == True:
        spike_path = opt.spike_path
        # standard spike input
        if spike_path.endswith('.dat'):
            spike = load_vidar_dat(spike_path,height=250,width=400)
            img_size = '250_400'
        # uhsr spike input
        elif spike_path.endswith('.npz'):
            spike = np.load(spike_path)['spk'].astype(np.float32)[:,13:237,13:237]
            print(np.load(spike_path)['label_name'])
            img_size = '224_224'
        spike = torch.tensor(spike)[None]
        spike_length = spike.shape[1]
        spike_name,_ = os.path.splitext(os.path.basename(spike_path))
    elif cls == 'REDS':
        dataset = SpikeData_REDS(opt.reds_folder,'REDS','test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
        img_size = '250_400'
    elif cls == 'Real':
        dataset = SpikeData_Real(opt.real_folder)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
        img_size = '250_400'
    elif cls == 'UHSR':
        dataset = SpikeData_UHSR(opt.UHSR_folder,'test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
        img_size = '224_224'

        
# main
def main():
    global dataloader,logger,metric_list,method_list,metric_pair,metric_single,dataset_cls 
    dataset_cls = opt.cls
    # dataset preparation
    dataset_load(cls = dataset_cls)
    logger = setup_logging(opt.save_name)
    # metric and method definition
    method_list = ['Spk2ImgNet','WGSE','SSML','SpikeFormer','TFP','TFI','TFSTP','RSIR']
    method_totest = opt.methods.split(',')
    if set(method_totest).issubset(method_list) == False:
        raise ValueError(f"Have unsupported methods!")
    else:
        method_list = method_totest
    # lower better: lpips ; higher better: psnr,ssim
    metric_pair = ['psnr','ssim','lpips']
    # lower better: niqe,brisque,piqe ; higher better: liqe_max,clipiqa
    metric_single = ['niqe','brisque','liqe_mix','clipiqa','piqe']
    # metric to test
    metric_totest = opt.metrics.split(',')
    if set(metric_totest).issubset(set(metric_pair + metric_single)) == False:
        raise ValueError(f"Have unsupported metrics!")
    elif dataset_cls in ['UHSR','Real'] and set(metric_totest).issubset(set(metric_single)) == False:
        raise ValueError(f"Dataset doesn't support reference metrics!")
    else:
        metric_list = metric_totest
    # parameter setting
    test_metric = opt.test_metric 
    test_params = opt.test_params
    test_imgs = opt.test_imgs
    
    if sum([test_metric,test_params,test_imgs]) >= 2:
        raise ValueError(f"Varaiables conflict!")

    # test -- metrics
    if test_metric == True:    
        metric_construct()
        # test -- metric
        for method_name in method_list:
            metric_calculate(method_name)
        metric_print()

    # test -- params
    elif test_params  == True:
        for method_name in method_list:
            params_calculate(method_name)
            
    # save -- imgs
    elif test_imgs == True:
        spike_idx = -1
        logger.info(spike_name)
        os.makedirs(f'imgs/{spike_name}',exist_ok = True)
        spike_plot = spike[0,0].int()
        spike_ll = spike.shape[1]
        # 16 is adjustable
        for i in range(spike_ll // 2 - spike_ll // 16, spike_ll // 2 + spike_ll // 16):
            spike_plot = torch.bitwise_or(spike_plot, spike[0,i].int())
        if img_size == '224_224':
            cv2.imwrite(f'imgs/{spike_name}/spike.png',spike_plot[30:-30,10:-10].detach().cpu().numpy() * 255)
        else:
            cv2.imwrite(f'imgs/{spike_name}/spike.png',spike_plot.detach().cpu().numpy() * 255)
        for method_name in method_list:
            img_reconstruct(spike,method_name,spike_idx,nor = True)
    
    # todo folder --test metrcis
    # if test_folder_metric == True:
    #     folder = 'imgs/train-350kmh'
    #     metric_list = ['niqe','brisque','liqe_mix','clipiqa']
    #     metric_construct()
    #     for img_path in os.listdir(folder):
    #         img = cv2.imread(os.path.join(folder,img_path))[:,:,0] / 255
    #         img = torch.tensor(img)[None,None].cuda().float()
    #         method_name = img_path.split('.')[0]
    #         if method_name not in method_list:
    #             continue
    #         metric_update(method_name,img,None)
    #     metric_print()
        
if __name__ == "__main__":
    global opt
    parser = argparse.ArgumentParser()
    parser.add_argument('--reds_folder', type=str,default='Data/REDS')
    parser.add_argument('--real_folder', type=str,default='Data/recVidarReal2019')
    parser.add_argument('--UHSR_folder', type=str,default='Data/U-CALTECH')
    parser.add_argument('--spike_path', type=str,default='Data/uhsr.npz') # uhsr.npz,data.dat,car-100kmh.dat
    parser.add_argument('--test_metric', action='store_true',default = False)
    parser.add_argument('--test_params', action='store_true',default = False)
    parser.add_argument('--test_imgs', action='store_true',default = False)
    parser.add_argument('--cls', default = 'spike',help = 'REDS, Real, UHSR and spike.')
    parser.add_argument('--methods', default = 'WGSE',help = 'Methods to test.')
    parser.add_argument('--metrics', default = 'piqe,niqe,brisque',help = 'Metrics to test.')
    parser.add_argument('--save_name', default = 'logs/result.log',help = 'Result path to save.')
    opt = parser.parse_args()
    main()
