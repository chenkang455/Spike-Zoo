<p align="center">
    <img src="imgs/spike-zoo.png" width="300"/>
<p>

<h5 align="center">

[![GitHub repo stars](https://img.shields.io/github/stars/chenkang455/Spike-Zoo?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/chenkang455/Spike-Zoo/stargazers) [![GitHub Issues](https://img.shields.io/github/issues/chenkang455/Spike-Zoo?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/chenkang455/Spike-Zoo/issues) <a href="https://badge.fury.io/py/spikezoo"><img src="https://badge.fury.io/py/spikezoo.svg" alt="PyPI version"></a>  <a href='https://spike-zoo.readthedocs.io/zh-cn/latest/index.html'><img src='https://readthedocs.com/projects/plenoptix-nerfstudio/badge/?version=latest' alt='Documentation Status' /></a>[![License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/chenkang455/Spike-Zoo)
<p>



<!-- <h2 align="center"> 
  <a href="">⚡Spike-Zoo: 
  </a>
</h2> -->

## 📖 About
⚡Spike-Zoo is the go-to library for state-of-the-art pretrained **spike-to-image** models designed to reconstruct images from spike streams. Whether you're looking for a simple inference solution or aiming to train your own spike-to-image models, ⚡Spike-Zoo is a modular toolbox that supports both, with key features including:

- Fast inference with pre-trained models.
- Training support for custom-designed spike-to-image models.
- Specialized functions for processing spike data.


## 🚩 Updates/Changelog
* **25-02-02:** Release the `Spike-Zoo v0.2` code, which supports more methods, provide more usages like training your method from scratch.
* **24-07-19:** Release the `Spike-Zoo v0.1` code for base evaluation of SOTA methods.

## 🍾 Quick Start
### 1. Installation
For users focused on **utilizing pretrained models for spike-to-image conversion**, we recommend installing SpikeZoo using one of the following methods:

* Install the last stable version `0.2.3` from PyPI:
```
pip install spikezoo
```
*  Install the latest developing version `0.2.3.6` from the source code **(recommended)**:
```
git clone https://github.com/chenkang455/Spike-Zoo
cd Spike-Zoo
python setup.py install
```

For users interested in **training their own spike-to-image model based on our framework**, we recommend cloning the repository and modifying the related code directly.
```
git clone https://github.com/chenkang455/Spike-Zoo
cd Spike-Zoo
python setup.py develop
```

### 2. Inference 
Reconstructing images from the spike is super easy with Spike-Zoo. Try the following code of the single model:
``` python
from spikezoo.pipeline import Pipeline, PipelineConfig
import spikezoo as sz
pipeline = Pipeline(
    cfg=PipelineConfig(save_folder="results",version="v023"),
    model_cfg=sz.METHOD.BASE,
    dataset_cfg=sz.DATASET.BASE 
)
pipeline.infer_from_dataset(idx = 0)
```


### 3. Training
We provide a user-friendly code for training our provided `BASE` model (modified from the `SpikeCLIP`) for the classic `REDS` dataset introduced in `Spk2ImgNet`:
``` python
from spikezoo.pipeline import TrainPipelineConfig, TrainPipeline
from spikezoo.datasets.reds_base_dataset import REDS_BASEConfig
from spikezoo.models.base_model import BaseModelConfig
pipeline = TrainPipeline(
    cfg=TrainPipelineConfig(save_folder="results", epochs = 10),
    dataset_cfg=REDS_BASEConfig(root_dir = "spikezoo/data/reds_base"),
    model_cfg=BaseModelConfig(),
)
pipeline.train()
``` 
We finish the training with one 4090 GPU in `2 minutes`, achieving `32.8dB` in PSNR and `0.92` in SSIM.

> 🌟 We encourage users to develop their models with simple modifications to our framework.

## 📚 How to navigate the documentation

| **Link** | **Description** |
| --- | --- |
| [Quick Start](https://spike-zoo.readthedocs.io/zh-cn/latest/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B.html) | Learn how to quickly get started with the Spike-Zoo repository for inference and training. |
| [Dataset](https://spike-zoo.readthedocs.io/zh-cn/latest/%E6%95%B0%E6%8D%AE%E9%9B%86.html) | Learn the parameter configuration of datasets and how to construct them. |
| [Model](https://spike-zoo.readthedocs.io/zh-cn/latest/%E6%A8%A1%E5%9E%8B.html) | Learn the parameter configuration of models and how to construct them. |
| [Pipeline](https://spike-zoo.readthedocs.io/zh-cn/latest/%E5%A4%84%E7%90%86%E7%AE%A1%E7%BA%BF.html) | Learn how to configure and construct the processing pipeline for models. |
| [Released Version](https://spike-zoo.readthedocs.io/zh-cn/latest/%E5%8F%91%E8%A1%8C%E7%89%88%E6%9C%AC%E4%BB%8B%E7%BB%8D.html) | Introduces the differences between different release versions of pre-trained weights. |
| [Examples](https://spike-zoo.readthedocs.io/zh-cn/latest/%E4%BD%BF%E7%94%A8%E4%BE%8B%E5%AD%90.html) | Complete code examples for using Spike-Zoo. |
| [Supports](https://spike-zoo.readthedocs.io/zh-cn/latest/%E6%94%AF%E6%8C%81%E8%8C%83%E5%9B%B4.html) | Learn about the datasets and models supported by Spike-Zoo. |


## 📅 TODO
- [x] Support the overall pipeline for spike simulation. 
- [x] Provide the tutorials.
- [ ] Support more training settings.
- [ ] Support more spike-based image reconstruction methods and datasets. 

## ✨‍ Acknowledgment
Our code is built on the open-source projects of [SpikeCV](https://spikecv.github.io/), [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch), [BasicSR](https://github.com/XPixelGroup/BasicSR) and [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio).We appreciate the effort of the contributors to these repositories. Thanks for [@zhiwen_huang](https://github.com/hzw-abc), [@ruizhao26](https://github.com/ruizhao26), [@shiyan_chen](https://github.com/hnmizuho) and [@Leozhangjiyuan](https://github.com/Leozhangjiyuan) for their help in building this project.

## 📑 Citation
If you find our codes helpful to your research, please consider to use the following citation:
```
@misc{spikezoo,
  title={{Spike-Zoo}: A Toolbox for Spike-to-Image Reconstruction},
  author={Kang Chen and Zhiyuan Ye and Tiejun Huang and Zhaofei Yu},
  year={2025},
  howpublished = {\url{https://github.com/chenkang455/Spike-Zoo}},
}
```
