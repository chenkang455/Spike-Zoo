<h2 align="center"> 
  <a href="">Spike-Zoo: A Toolbox for Spike-to-Image Reconstruction
  </a>
</h2>

## 📖 About
⚡ Spike-Zoo is the go-to library for state-of-the-art pretrained **spike-to-image** models designed to reconstruct images from spike streams. Whether you're looking for a simple inference solution or aiming to train your own spike-to-image models, ⚡Spike-Zoo is a modular toolbox that supports both, with key features including:

- Fast inference with pre-trained models.
- Training support for custom-designed spike-to-image models.
- Specialized functions for processing spike data.



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
``` python
import spikezoo as sz
spike = sz.load_vidar_dat("data/scissor.dat",width = 400,height = 250,version='cpp')
```
🚀 Results on [examples/test_load_dat.py](examples/test_load_dat.py) show that the `cpp` version is more than 10 times faster than the `python` version.

## 📅 TODO
- [ ] Provide the tutorials.
- [ ] Support more training settings.
- [ ] Support more spike-based image reconstruction methods and datasets. 
- [ ] Support the overall pipeline for spike simulation. 

## 🤗 Supports
Run the following code to find our supported models, datasets and metrics:
``` python
import spikezoo as sz
print(sz.get_models())
print(sz.get_datasets())
print(sz.get_metrics())
```
**Supported Models:**
|  Models   | Source  
|  ----  | ----  | 
| `tfp`,`tfi` | Spike camera and its coding methods | 
| `spk2imgnet`  | Spk2ImgNet: Learning to Reconstruct Dynamic Scene from Continuous Spike Stream | 
| `wgse`  | Learning Temporal-Ordered Representation for Spike Streams Based on Discrete Wavelet Transforms | 
| `ssml`  | Self-Supervised Mutual Learning for Dynamic Scene Reconstruction of Spiking Camera | 
| `spikeformer`  | SpikeFormer: Image Reconstruction from the Sequence of Spike Camera Based on Transformer |
| `ssir`  | Spike Camera Image Reconstruction Using Deep Spiking Neural Networks |
| `bsf`  | Boosting Spike Camera Image Reconstruction from a Perspective of Dealing with Spike Fluctuations |
| `stir`  | Spatio-Temporal Interactive Learning for Efficient Image Reconstruction of Spiking Cameras |
| `spikeclip`  | Rethinking High-speed Image Reconstruction Framework with Spike Camera |

**Supported Datasets:**
|  Datasets   | Source  
|  ----  | ----  | 
| `reds_small`  | Spk2ImgNet: Learning to Reconstruct Dynamic Scene from Continuous Spike Stream | 
| `uhsr`  | Recognizing Ultra-High-Speed Moving Objects with Bio-Inspired Spike Camera | 
| `realworld`  | `recVidarReal2019`,`momVidarReal2021` in [SpikeCV](https://github.com/Zyj061/SpikeCV) | 
| `szdata`  | SpikeReveal: Unlocking Temporal Sequences from Real Blurry Inputs with Spike Streams | 


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