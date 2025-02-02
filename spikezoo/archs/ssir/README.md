## [TCSVT 2023] Spike Camera Image Reconstruction Using Deep Spiking Neural Networks

<h4 align="center"> Rui Zhao<sup>1</sup>, Ruiqin Xiong<sup>1</sup>, Jian Zhang<sup>2</sup>, Zhaofei Yu<sup>1</sup>, Shuyuan Zhu<sup>3</sup>, Lei Ma <sup>1</sup>, Tiejun Huang<sup>1</sup> </h4>
<h4 align="center">1. National Engineering Research Center of Visual Technology, School of Computer Science, Peking University<br>
2. School of Electronic and Computer Engineering, Peking University Shenzhen Graduate School<br>
3.  School of Information and Communication Engineering, UESTC</h4><br>

This repository contains the official source code for our paper:

Spike Camera Image Reconstruction Using Deep Spiking Neural Networks

TCSVT 2023

[Paper](https://ieeexplore.ieee.org/document/10288531)



## Environment

You can choose cudatoolkit version to match your server. The code is tested on PyTorch 2.0.1+cuda12.0.

```shell
conda create -n ssir python==3.10
conda activate ssir
# You can choose the PyTorch version you like, we recommand version >= 1.10.1
# For example
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```

## Prepare the Data

#### 1. Download and deploy the SREDS dataset

[BaiduNetDisk](https://pan.baidu.com/s/1clA43FcxjOibL1zGTaU82g) (Password: 2728)

`train.tar` corresponds to the training data, and `test.tar` corresponds to the testing data.

Move the above two `.tar` file to the `data root` directory and extract to the current directory

```
file directory:
train:
your_data_root/crop_mini/spike/...
your_data_root/crop_mini/image/...
test:
your_data_root/spike/...
your_data_root/imgs/...
```

#### 2. Set the path of RSSF dataset in your serve

In the line25 of `main.py` or set that in command line when running main.py

## Evaluate
```shell
cd shells
bash eval_SREDS.sh
```

## Train
```shell
cd shells
bash train_SSIR.sh
```
We recommended to redirect the output logs by adding
`>> SSIR.txt 2>&1` 
to the last of the above command for management.


## Citation

If you find this code useful in your research, please consider citing our paper.

```
@article{zhao2023spike,
  title={Spike Camera Image Reconstruction Using Deep Spiking Neural Networks},
  author={Zhao, Rui and Xiong, Ruiqin and Zhang, Jian and Yu, Zhaofei and Zhu, Shuyuan and Ma, Lei and Huang, Tiejun},
  journal={IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)},
  year={2023},
}
```

If you have any questions, please contact:  
ruizhao@stu.pku.edu.cn

 