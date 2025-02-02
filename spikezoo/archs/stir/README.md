<!---
# Spatio-Temporal Interactive Learning for Efficient Image Reconstruction of Spiking Cameras

This repository contains the source code for the paper: [Spatio-Temporal Interactive Learning for Efficient Image Reconstruction of Spiking Cameras (NeurIPS 2024)](https://openreview.net/pdf?id=S4ZqnMywcM). 
The spiking camera is an emerging neuromorphic vision sensor that records high-speed motion scenes by asynchronously firing continuous binary spike streams. Prevailing image reconstruction methods, generating intermediate frames from these spike streams, often rely on complex step-by-step network architectures that overlook the intrinsic collaboration of spatio-temporal complementary information. In this paper, we propose an efficient spatio-temporal interactive reconstruction network to jointly perform inter-frame feature alignment and intra-frame feature filtering in a coarse-to-fine manner. Specifically, it starts by extracting hierarchical features from a concise hybrid spike representation, then refines the motion fields and target frames scale-by-scale, ultimately obtaining a full-resolution output. Meanwhile, we introduce a symmetric interactive attention block and a multi-motion field estimation block to further enhance the interaction capability of the overall network. Experiments on synthetic and real-captured data show that our approach exhibits excellent performance while maintaining low model complexity.

<img src="picture/performance-speed.png" width="75%"/>
<img src="picture/overview.png" width="80%"/>
<img src="picture/results_visual.png" width="82%"/>
-->
## Installation
You can choose cudatoolkit version to match your server. The code is tested with PyTorch 1.9.1 with CUDA 11.4.

```shell
conda create -n stir python==3.8.12
conda activate stir
# You can choose the PyTorch version you like, for example
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.0.2
```

Install the dependent packages:
```
pip install -r requirements.txt
```

Install core package
```
cd ./package_core
python setup.py install
```

In our implementation, we borrowed the code framework of [SSIR](https://github.com/ruizhao26/SSIR):

## Prepare the Data

#### 1. Download and deploy the SREDS dataset to your local computer from [SSIR](https://github.com/ruizhao26/SSIR).

#### 2. Set the path of the SREDS dataset in your serve

Set that in `--data_root` when running train_STIR.sh or eval_SREDS.sh

## Evaluate
```
sh eval_SREDS.sh
```

## Train
```
sh train_STIR.sh
```
<!---
## Citations
If you find our approach useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```
@article{fan2024spatio,
	title={Spatio-Temporal Interactive Learning for Efficient Image Reconstruction of Spiking Cameras},
	author={Fan, Bin and Yin, Jiaoyang and Dai, Yuchao and Xu, Chao and Huang, Tiejun and Shi, Boxin},
	journal={Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)},
        volume={},
	year={2024}
}
```
-->
## Statement
This project is for research purpose only, please contact us for the licence of commercial use. For any other questions or discussion please contact: binfan@mail.nwpu.edu.cn
