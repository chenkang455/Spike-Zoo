# WGSE-SpikeCamera

Codes and Datasets of **"Learning Temporal-ordered Representation for Spike Streams Based on Discrete Wavelet Transforms"**.

Jiyuan Zhang*, Shanshan Jia*, Zhaofei Yu $\dagger$ and Tiejun Huang

 Thirty-Seventh AAAI Conference on Artificial Intelligence (**AAAI 2023**) .
 __________________________________________________
 ## Code for WGSE

* **Key Requirements of the Code Environment**

  * Pytorch > 1.7.0
  * pywavelets
  * pytorch-wavelets
  * scikit-image
  * einops

* **Train**

To train the network for WGSE, use the file `train.py`

the parameter list is:

```python
parser = argparse.ArgumentParser(description='AAAI - WGSE - REDS')
parser.add_argument('-c', '--cuda', type=str, default='1', help='select gpu card')
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-e', '--epoch', type=int, default=600)
parser.add_argument('-w', '--wvl', type=str, default='db8', help='select wavelet base function')
parser.add_argument('-j', '--jlevels', type=int, default=5)
parser.add_argument('-k', '--kernel_size', type=int, default=3)
parser.add_argument('-l', '--logpath', type=str, default='WGSE-Dwt1dNet')
parser.add_argument('-r', '--resume_from', type=str, default=None)
parser.add_argument('--dataroot', type=str, default=None)
```

`-c` is the CUDA device index on your computer, `-b` is the batchsize, `-e` is the number of epoch of training, `-w` is the wavelet function, `-j` is the decompostion level, `-k` is the kernel size in the WGSE, `-r` is the folder where saving the weights that you want to load and resume training, `--dataroot` is the root path of the dataset and `-l` is the folder path where you want to save the log files and weight file of the model.

In our implementation, the training script should be:

``` bash
python train.py -c 0 -b 16 -e 600 --dataroot "rootpath/of/dataset"  -l "folder/for/save/logfiles"
```

* **Test**

To test the network for WGSE, use the file `demo.py`

the parameter list is:

```python
parser = argparse.ArgumentParser(description='AAAI - WGSE - REDS')
parser.add_argument('-w', '--wvl', type=str, default='db8', help='select wavelet base function')
parser.add_argument('-j', '--jlevels', type=int, default=5)
parser.add_argument('-k', '--kernel_size', type=int, default=3)
parser.add_argument('-l', '--logpath', type=str, default='WGSE-Dwt1dNet')
parser.add_argument('-f', '--datfile', type=str, default=None, help='path of the spike data to be tested')
```

`-w` is the wavelet function, `-j` is the decompostion level, `-k` is the kernel size in the WGSE, `-l` is the folder path where you save the weight file, `-f` is the `.dat` data path of spikes that you want to test.

## The Dataset for WGSE
In this work, we propose a synthetic dataset "Spike-Cityscapes" for semantic segmentation based on the spike streams generated from [Cityscapes](https://www.cityscapes-dataset.com/). The spike data is available and you can download them at [https://pan.baidu.com/s/1lB4qpfZwaVN6WDFo5MRR-w](https://pan.baidu.com/s/1lB4qpfZwaVN6WDFo5MRR-w) with the password **svpg**.
