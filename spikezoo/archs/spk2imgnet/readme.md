## [CVPR 2021] Spk2ImgNet: Learning to Reconstruct Dynamic Scene from Continuous Spike Stream


<h4 align="center"> Jing Zhao, Ruiqin Xiong, Hangfan Liu, Jian Zhang, Tiejun Huang </h4>

This repository contains the official source code for our paper:

Spk2ImgNet: Learning to Reconstruct Dynamic Scene from Continuous Spike Stream.  CVPR 2021 

Paper:  
[Spk2ImgNet-CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Spk2ImgNet_Learning_To_Reconstruct_Dynamic_Scene_From_Continuous_Spike_Stream_CVPR_2021_paper.pdf) 

* [Spk2ImgNet](#Learning-to-Reconstruct-Dynamic-Scene-from-Continuous-Spike-Stream.)
  * [Environments](#Environments)
  * [Download the pretrained models](#Download-the-pretrained-models)
  * [Evaluate](#Evaluate)
  * [Train](#Train)
  * [Citation](#Citations)


## Environments

You will have to choose cudatoolkit version to match your compute environment. The code is tested on PyTorch 1.10.2+cu113 and spatial-correlation-sampler 0.3.0 but other versions might also work. 

```bash
conda create -n steflow python==3.9
conda activate steflow
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip3 install matplotlib opencv-python h5py
```

We don't ensure that all the PyTorch versions can work well.

## Prepare the Data

### Download the pretrained models

The pretrained model can be downloaded in the Google Drive link below

[Link for pretrained model](https://drive.google.com/file/d/1vBTJxlctk4otQKsyRq7lsFYGU4WGRNjt/view?usp=sharing)

You can download the pretrained models to ```./ckpt```

### Download the training data

The training data can be downloaded in the Google Drive link below

[Link for training data](https://drive.google.com/file/d/1ozR2-fNmU10gA_TCYUfJN-ahV6e_8Ke7/view?usp=sharing)

## Evaluate 

You can set the data path in the .py files or through argparser (--data)

```bash
python3 main_steflow_dt1.py \
--test_data 'Spk2ImgNet_test2' \
--model_name 'model_061.pth'

```


## Train


All the command line arguments for hyperparameter tuning can be found in the `train.py` file.
You can set the data path in the .py files or through argparser (--data)

```bash
python3 train.py
```

## Citations

If you find this code useful in your research, please consider citing our paper: 

```
@inproceedings{zhao2021spike,
  title={Spk2ImgNet: Learning to Reconstruct Dynamic Scene from Continuous Spike Stream},
  author={Zhao, Jing and Xiong, Ruiqin and Liu, Hangfan and Zhang, Jian and Huang, Tiejun},
  booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```



