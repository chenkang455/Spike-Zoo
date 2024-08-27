# visualize spike 
# uhsr
CUDA_VISIBLE_DEVICES=1 python compare.py \
--test_imgs \
--methods Spk2ImgNet,WGSE,SSML,TFP,TFI,TFSTP,RSIR,SpikeFormer \
--cls spike \
--spike_path Data/uhsr.npz\
# car
CUDA_VISIBLE_DEVICES=2 python compare.py \
--test_imgs \
--methods Spk2ImgNet,WGSE,SSML,TFP,TFI,TFSTP,RSIR,SpikeFormer \
--cls spike \
--spike_path Data/car-100kmh.dat\

# test parameters flops and latency
CUDA_VISIBLE_DEVICES=0 python compare.py \
--test_params \
--save_name logs/params.log \
--methods Spk2ImgNet,WGSE,SSML,TFP,TFI,TFSTP,RSIR,SpikeFormer

# test metric on the REDS dataset
CUDA_VISIBLE_DEVICES=0 python compare.py \
--test_metric \
--save_name logs/reds_metric.log \
--methods Spk2ImgNet,WGSE,SSML,TFP,TFI,TFSTP,RSIR,SpikeFormer \
--cls REDS \
--metrics psnr,ssim,lpips,niqe,brisque,liqe_mix,clipiqa

# test metric on the real-spike dataset
CUDA_VISIBLE_DEVICES=0 python compare.py \
--test_metric \
--save_name logs/real_metric.log \
--methods Spk2ImgNet,WGSE,SSML,TFP,TFI,TFSTP,RSIR,SpikeFormer \
--cls Real \
--metrics niqe,brisque,liqe_mix,clipiqa

# test metric on the U-CALTECH
CUDA_VISIBLE_DEVICES=0  python compare.py \
--test_metric \
--UHSR_folder Data/U-CALTECH \
--save_name logs/caltech.log \
--methods RSIR,Spk2ImgNet,WGSE,SSML,TFP,TFI,TFSTP,LRN,SpikeFormer \
--metrics piqe,niqe,brisque

# test metric on the U-CIFAR
CUDA_VISIBLE_DEVICES=0  python compare.py \
--test_metric \
--UHSR_folder Data/U-CIFAR \
--save_name logs/cifar.log \
--methods RSIR,Spk2ImgNet,WGSE,SSML,TFP,TFI,TFSTP,LRN,SpikeFormer \
--metrics piqe,niqe,brisque

