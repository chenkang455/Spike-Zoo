<p align="center">
    <img src="imgs/spike-zoo.png" width="350"/>
<p>
<h5 align="center">

[![GitHub repo stars](https://img.shields.io/github/stars/chenkang455/Spike-Zoo?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/chenkang455/Spike-Zoo/stargazers) [![GitHub Issues](https://img.shields.io/github/issues/chenkang455/Spike-Zoo?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/chenkang455/Spike-Zoo/issues) <a href="https://badge.fury.io/py/spikezoo"><img src="https://badge.fury.io/py/spikezoo.svg" alt="PyPI version"></a> [![License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/chenkang455/Spike-Zoo)
<p>

<!-- <h2 align="center"> 
  <a href="">⚡Spike-Zoo: A Toolbox for Spike-to-Image Reconstruction
  </a>
</h2> -->

## 📖 About
⚡Spike-Zoo is the go-to library for state-of-the-art pretrained **spike-to-image** models designed to reconstruct images from spike streams. Whether you're looking for a simple inference solution or aiming to train your own spike-to-image models, ⚡Spike-Zoo is a modular toolbox that supports both, with key features including:

- Fast inference with pre-trained models.
- Training support for custom-designed spike-to-image models.
- Specialized functions for processing spike data.

> 📚Tutorials: https://spike-zoo.readthedocs.io/zh-cn/latest/# 

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
*  Install the latest developing version `0.2.3` from the source code :
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
```
You can also run multiple models at once by changing the pipeline (version parameter corresponds to our released different versions in [Releases](https://github.com/chenkang455/Spike-Zoo/releases)):
``` python
import spikezoo as sz
from spikezoo.pipeline import EnsemblePipeline, EnsemblePipelineConfig
pipeline = EnsemblePipeline(
    cfg=EnsemblePipelineConfig(save_folder="results",version="v023"),
    model_cfg_list=[
        sz.METHOD.BASE,sz.METHOD.TFP,sz.METHOD.TFI,sz.METHOD.SPK2IMGNET,sz.METHOD.WGSE,
        sz.METHOD.SSML,sz.METHOD.BSF,sz.METHOD.STIR,sz.METHOD.SPIKECLIP,sz.METHOD.SSIR],
    dataset_cfg=sz.DATASET.BASE,
)
```
Having established our pipelines, we provide following functions to enjoy these spike-to-image models.

* I. Obtain the restoration metric and save the recovered image from the given spike:
``` python
# 1. spike-to-image from the given dataset
pipeline.infer_from_dataset(idx = 0)

# 2. spike-to-image from the given .dat file
pipeline.infer_from_file(file_path = 'data/scissor.dat',width = 400,height=250)

# 3. spike-to-image from the given spike
spike = sz.load_vidar_dat("data/scissor.dat",width = 400,height = 250)
pipeline.infer_from_spk(spike)
```


* II. Save all images from the given dataset.
``` python
pipeline.save_imgs_from_dataset()
```

* III. Calculate the metrics for the specified dataset.
``` python
pipeline.cal_metrics()
```

* IV. Calculate the parameters (params,flops,latency) based on the established pipeline.
``` python
pipeline.cal_params()
```

For detailed usage, welcome check [test_single.ipynb](examples/test/test_single.ipynb) and [test_ensemble.ipynb](examples/test/test_ensemble.ipynb).

### 3. Training
We provide a user-friendly code for training our provided `base` model (modified from the `SpikeCLIP`) for the classic `REDS` dataset introduced in `Spk2ImgNet`:
``` python
from spikezoo.pipeline import TrainPipelineConfig, TrainPipeline
from spikezoo.datasets.reds_base_dataset import REDS_BASEConfig
from spikezoo.models.base_model import BaseModelConfig
pipeline = TrainPipeline(
    cfg=TrainPipelineConfig(save_folder="results", epochs = 10),
    dataset_cfg=REDS_BASEConfig(root_dir = "spikezoo/data/REDS_BASE"),
    model_cfg=BaseModelConfig(),
)
pipeline.train()
``` 
We finish the training with one 4090 GPU in `2 minutes`, achieving `32.8dB` in PSNR and `0.92` in SSIM.

> 🌟 We encourage users to develop their models with simple modifications to our framework, and the tutorial will be released soon. 

We retrain all supported methods except `SPIKECLIP` on this REDS dataset (training scripts are placed on [examples/train_reds_base](examples/train_reds_base) and evaluation script is placed on [test_REDS_base.py](examples/test/test_REDS_base.py)), with our reported metrics as follows:

| Method               | PSNR  | SSIM   | LPIPS   | NIQE    | BRISQUE  | PIQE  | Params (M) | FLOPs (G) | Latency (ms) |
|----------------------|:-------:|:--------:|:---------:|:---------:|:----------:|:-------:|:------------:|:-----------:|:--------------:|
| `tfi`                | 16.503 | 0.454  | 0.382   | 7.289   | 43.17    | 49.12 | 0.00       | 0.00      | 3.60         |
| `tfp`                | 24.287 | 0.644  | 0.274   | 8.197   | 48.48    | 38.38 | 0.00       | 0.00      | 0.03         |
| `spikeclip`          | 21.873 | 0.578  | 0.333   | 7.802   | 42.08    | 54.01 | 0.19       | 23.69     | 1.27         |
| `ssir`               | 26.544 | 0.718  | 0.325   | 4.769   | 28.45    | 21.59 | 0.38       | 25.92     | 4.52         |
| `ssml`               | 33.697 | 0.943  | 0.088   | 4.669   | 32.48    | 37.30 | 2.38       | 386.02    | 244.18       |
| `base`               | 36.589 | 0.965  | 0.034   | 4.393   | 26.16    | 38.43 | 0.18       | 18.04     | 0.40         |
| `stir`               | 37.914 | 0.973  | 0.027   | 4.236   | 25.10    | 39.18 | 5.08       | 43.31     | 21.07        |
| `wgse`               | 39.036 | 0.978  | 0.023   | 4.231   | 25.76    | 44.11 | 3.81       | 415.26    | 73.62        |
| `spk2imgnet`         | 39.154 | 0.978  | 0.022   | 4.243   | 25.20    | 43.09 | 3.90       | 1000.50   | 123.38       |
| `bsf`                | 39.576 | 0.979  | 0.019   | 4.139   | 24.93    | 43.03 | 2.47       | 705.23    | 401.50       |

### 4. Model Usage
We also provide a direct interface for users interested in taking the spike-to-image model as a part of their work:

```python 
import spikezoo as sz
from spikezoo.models.base_model import BaseModel, BaseModelConfig
# input data
spike = sz.load_vidar_dat("data/data.dat", width=400, height=250, out_format="tensor")
spike = spike[None].cuda()
print(f"Input spike shape: {spike.shape}")
# net
net = BaseModel(BaseModelConfig(model_params={"inDim": 41}))
net.build_network(mode = "debug")
# process
recon_img = net(spike)
print(recon_img.shape,recon_img.max(),recon_img.min())
```
For detailed usage, welcome check [test_model.ipynb](examples/test/test_model.ipynb).

### 5. Spike Utility
#### I. Faster spike loading interface
We provide a faster `load_vidar_dat` function implemented with `cpp` (by [@zeal-ye](https://github.com/zeal-ye)):
``` python
import spikezoo as sz
spike = sz.load_vidar_dat("data/scissor.dat",width = 400,height = 250,version='cpp')
```
🚀 Results on [test_load_dat.py](examples/test_load_dat.py) show that the `cpp` version is more than 10 times faster than the `python` version.

#### II. Spike simulation pipeline.
We provide our overall spike simulation pipeline in [scripts](scripts/), try to modify the config in `run.sh` and run the command to start the simulation process:
``` bash
bash run.sh
```

#### III. Spike-related functions.
For other spike-related functions, welcome check [spike_utils.py](spikezoo/utils/spike_utils.py)

## 📅 TODO
- [x] Support the overall pipeline for spike simulation. 
- [ ] Provide the tutorials.
- [ ] Support more training settings.
- [ ] Support more spike-based image reconstruction methods and datasets. 

## 🤗 Supports
Run the following code to find our supported models, datasets and metrics:
``` python
import spikezoo as sz
print(sz.METHODS)
print(sz.DATASETS)
print(sz.METRICS)
```
**Supported Models:**
|  Models   | Source  
|  ----  | ----  | 
| `tfp`,`tfi` | Spike camera and its coding methods | 
| `spk2imgnet`  | Spk2ImgNet: Learning to Reconstruct Dynamic Scene from Continuous Spike Stream | 
| `wgse`  | Learning Temporal-Ordered Representation for Spike Streams Based on Discrete Wavelet Transforms | 
| `ssml`  | Self-Supervised Mutual Learning for Dynamic Scene Reconstruction of Spiking Camera | 
| `ssir`  | Spike Camera Image Reconstruction Using Deep Spiking Neural Networks |
| `bsf`  | Boosting Spike Camera Image Reconstruction from a Perspective of Dealing with Spike Fluctuations |
| `stir`  | Spatio-Temporal Interactive Learning for Efficient Image Reconstruction of Spiking Cameras |
| `base`,`spikeclip`  | Rethinking High-speed Image Reconstruction Framework with Spike Camera |

**Supported Datasets:**
|  Datasets   | Source  
|  ----  | ----  | 
| `reds_base`  | Spk2ImgNet: Learning to Reconstruct Dynamic Scene from Continuous Spike Stream | 
| `uhsr`  | Recognizing Ultra-High-Speed Moving Objects with Bio-Inspired Spike Camera | 
| `realworld`  | `recVidarReal2019`,`momVidarReal2021` in [SpikeCV](https://github.com/Zyj061/SpikeCV) | 
| `szdata`  | SpikeReveal: Unlocking Temporal Sequences from Real Blurry Inputs with Spike Streams | 


## ✨‍ Acknowledgment
Our code is built on the open-source projects of [SpikeCV](https://spikecv.github.io/), [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch), [BasicSR](https://github.com/XPixelGroup/BasicSR) and [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio).We appreciate the effort of the contributors to these repositories. Thanks for [@ruizhao26](https://github.com/ruizhao26), [@shiyan_chen](https://github.com/hnmizuho) and [@Leozhangjiyuan](https://github.com/Leozhangjiyuan) for their help in building this project.

## 📑 Citation
If you find our codes helpful to your research, please consider to use the following citation:
```
@misc{spikezoo,
  title={{Spike-Zoo}: Spike-Zoo: A Toolbox for Spike-to-Image Reconstruction},
  author={Kang Chen and Zhiyuan Ye and Tiejun Huang and Zhaofei Yu},
  year={2025},
  howpublished = "[Online]. Available: \url{https://github.com/chenkang455/Spike-Zoo}"
}
```