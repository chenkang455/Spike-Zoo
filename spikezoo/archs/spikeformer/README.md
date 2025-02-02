# SpikeFormer [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
Pytorch Implementation of "SpikeFormer: Image Reconstruction from the Sequence of Spike Camera Based on Transformer"[[Paper]](https://dl.acm.org/doi/abs/10.1145/3512388.3512399)

## Prerequisites
* Create a conda environment by running `conda env create -f requirements.yml`

## Dataset Structure
* To train the SpikeFormer, please organize file structure of the dataset as follows:
```
Dataset  
├── test  
│   └── c.npz  
├── train  
│   └── a.npz  
└── valid  
    └── b.npz  
```

## Pretrained Model
* Download the pretrained model [here](https://pan.baidu.com/s/1aeW15vQh0GXgRJtfStBHDg) using password: nwh5.
* Put the model to the path ./CheckPoints/

## Training
* Run `python train.py` to train SpikeFormer on training set.

## Validation
* Run `python evaluate.py` to evaluate the performance of trained model on testing set.

## Reconstruct Images from Real Spike Data
* Run `python recon_real_data.py`.
