## [CVPR 2024] Boosting Spike Camera Image Reconstruction from a Perspective of Dealing with Spike Fluctuations

<h4 align="center"> Rui Zhao<sup>1,2</sup>, Ruiqin Xiong<sup>1,2</sup>, Jing Zhao<sup>1,2</sup>, Jian Zhang<sup>3</sup>, Xiaopeng Fan<sup>4</sup>, Zhaofei Yu<sup>1,2</sup>, Tiejun Huang<sup>1,2</sup> </h4>
<h4 align="center">1. School of Computer Science, Peking University<br>
2. National Key Laboratory for Multimedia Information Processing, Peking University<br>
3. School of Electronic and Computer Engineering,  Peking University<br>
4. School of Computer Science and Technology, Harbin Institute of Technology
</h4><br>

This repository contains the official source code for our paper:

Boosting Spike Camera Image Reconstruction from a Perspective of Dealing with Spike Fluctuations

CVPR 2024

## Environment

You can choose cudatoolkit version to match your server. The code is tested on PyTorch 2.0.1+cu120.

```bash
conda create -n bsf python==3.10.9
conda activate bsf
# You can choose the PyTorch version you like, we recommand version >= 1.10.1
# For example
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

## Prepare the Data

##### 1. Download the dataset (Approximate 50GB)

[Link of the dataset (BaiduNetDisk)](https://pan.baidu.com/s/1zBp-ed1KtmhAab5Z_62ttw)  (Password: 2728) 

##### 2. Deploy the dataset for training faster (Approximate <u>another</u> 125GB)

firstly modify the data root and output root in `./prepare_data/crop_dataset_train.py` and `./prepare_data/crop_dataset_val.py`

```shell
cd prepare_data &&
bash crop_train.sh $your_gpu_id &&
bash crop_val.sh $your_gpu_id
```

## Evaluate

```shell
CUDA_VISIBLE_DEVICES=$1 python3 -W ignore main.py \
--alpha 0.7 \
--vis-path vis/bsf \
-evp eval_vis/bsf \
--logs_file_name bsf \
--compile_model \
--test_eval \
--arch bsf \
--pretrained ckpt/bsf.pth
```

## Train

```shell
CUDA_VISIBLE_DEVICES=$1 python3 -W ignore main.py \
-bs 8 \
-j 8 \
-lr 1e-4 \
--epochs 61 \
--train-res 96 96 \
--lr-scale-factor 0.5 \
--milestones 10 20 30 40 50 60 70 80 90 100 \
--alpha 0.7 \
--vis-path vis/bsf \
-evp eval_vis/bsf \
--logs_file_name bsf \
--compile_model \
--weight_decay 0.0 \
--eval-interval 10 \
--half_reserve 0 \
--arch bsf
```

## Citations

If you find this code useful in your research, please consider citing our paper:

```
@inproceedings{zhao2024boosting,
  title={Boosting Spike Camera Image Reconstruction from a Perspective of Dealing with Spike Fluctuations},
  author={Zhao, Rui and Xiong, Ruiqin and Zhao, Jing and Zhang, Jian and Fan, Xiaopeng and Yu, Zhaofei, and Huang, Tiejun},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
