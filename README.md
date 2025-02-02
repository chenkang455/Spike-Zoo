<h2 align="center"> 
  <a href="">Spike-Zoo: A Toolbox for Spike-to-Image Reconstruction
  </a>
</h2>

## 📖 About
⚡Spike-Zoo is the go-to library for state-of-the-art pretrained **spike-to-image** models for reconstructing the image from the given spike stream. Whether you're looking for a **simple inference** solution or **training** your own spike-to-image models, ⚡Spike-Zoo is a modular toolbox that supports both. 

If Spike-Zoo helps your research or work, please help to ⭐ this repo or recommend it to your friends. Thanks😊

## 🚩 Updates/Changelog
* **25-02-02:** Release the `Spike-Zoo v0.2` code, which supports more methods, provide more usages.
* **24-08-26:** Update the `SpikeFormer` and `RSIR` methods, the `UHSR` dataset and the `piqe` non-reference metric.

* **24-07-19:** Release the `Spike-Zoo v0.1` base code.

## 🍾 Quick Start
### 1. Installation
For users focused on **utilizing pretrained models for spike-to-image conversion**, we recommend installing SpikeZoo using one of the following methods:

* Install the last stable version from PyPI:
```
pip install spikezoo
```
*  Install the latest developing version from the source code:
```
git clone https://github.com/chenkang455/Spike-Zoo
cd Spike-Zoo
python setup.py install
```

For users interested in **training their own spike-to-image model based on our framework**, we recommend cloning the repository and modifying the related code directly.

### 2. Inference 
Reconstructing images from the spike input is super easy with Spike-Zoo. Try the following code of the single model:
``` python
from spikezoo.pipeline import Pipeline, PipelineConfig
pipeline = Pipeline(
    cfg = PipelineConfig(save_folder="results"),
    model_cfg="spk2imgnet",
    dataset_cfg="base"
)
```
You can also run multiple models at once by changing the pipeline:
``` python
from spikezoo.pipeline import EnsemblePipeline, EnsemblePipelineConfig
pipeline = EnsemblePipeline(
    cfg = EnsemblePipelineConfig(save_folder="results"),
    model_cfg_list=['tfp','tfi', 'spk2imgnet', 'wgse', 'ssml', 'bsf', 'stir',  'spikeclip','spikeformer'],
    dataset_cfg="base"
)
```
* Having established the pipeline, run the following code to obtain the metric and save the reconstructed image from the given spike:
``` python
# 1. spike-to-image from the given dataset
pipeline.spk2img_from_dataset(idx = 0)

# 2. spike-to-image from the given .dat file
pipeline.spk2img_from_file(file_path = 'data/scissor.dat',width = 400,height=250)

# 3. spike-to-image from the given spike
import spikezoo as sz
spike = sz.load_vidar_dat("data/scissor.dat",width = 400,height = 250,version='cpp')
pipeline.spk2img_from_spk(spike)
```
For detailed usage, welcome check [test_single.ipynb](examples/test_single.ipynb) and [test_multi.ipynb](examples/test_multi.ipynb) 😊😊😊.

* Save all images of the given dataset.
``` python
pipeline.save_imgs_from_dataset()
```

* Calculate the metrics for the specified dataset.
``` python
pipeline.cal_metrics()
```

* Calculate the parameters (params,flops,latency) based on the established pipeline.
``` python
pipeline.cal_params()
```

### 3. Training
We provide a user-friendly code for training our provided base model (modified from the `SpikeCLIP`) for the classic `REDS` dataset introduced in `Spk2ImgNet`:
``` python
from spikezoo.pipeline import TrainPipelineConfig, TrainPipeline
from spikezoo.datasets.reds_small_dataset import REDS_Small_Config
pipeline = TrainPipeline(
    cfg=TrainPipelineConfig(save_folder="results", epochs = 10),
    dataset_cfg=REDS_Small_Config(root_dir = "path/REDS_Small"),
    model_cfg="base",
)
pipeline.train()
``` 
We finish the training with one 4090 GPU in `2 minutes`, achieving `34.7dB` in PSNR and `0.94` in SSIM.

> 🌟 We encourage users to develop their models using our framework, with the tutorial being released soon.

### 4. Others
We provide a faster `load_vidar_dat` function implemented with `cpp` (by @zeal-ye):
``` bash
import spikezoo as sz
spike = sz.load_vidar_dat("data/scissor.dat",width = 400,height = 250,version='cpp')
```
🚀 Results on [examples/test_load_dat.py](examples/test_load_dat.py) show that the `cpp` version is more than 10 times faster than the `python` version.


## 📅 TODO
- [ ] Provide the tutorials.
- [ ] Support more training settings.
- [ ] Support more spike-based image reconstruction methods and datasets. 
- [ ] Support the overall pipeline for spike simulation. 

## ✨‍ Acknowledgment
Our code is built on the open-source projects of [SpikeCV](https://spikecv.github.io/), [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch), [BasicSR](https://github.com/XPixelGroup/BasicSR) and [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio).We appreciate the effort of the contributors to these repositories. Thanks for @ruizhao26 and @Leozhangjiyuan for their help in building this project.

## 📑 Citation
If you find our codes helpful to your research, please consider to use the following citation:
```
@misc{spikezoo,
  title={{Spike-Zoo}: Spike-Zoo: A Toolbox for Spike-to-Image Reconstruction},
  author={Kang Chen and Zhiyuan Ye},
  year={2025},
  howpublished = "[Online]. Available: \url{https://github.com/chenkang455/Spike-Zoo}"
}
```