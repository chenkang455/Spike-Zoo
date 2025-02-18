Spike-Zoo 文档
=======================
⚡ `Spike-Zoo <https://github.com/chenkang455/Spike-Zoo>`__ 是一个用于脉冲相机图像重构网络的推理和训练框架，主要功能包括：

1. 使用预训练权重，快速进行多个重构模型的 **推理**。
2. **训练** 自定义设计的脉冲图像重构网络。
3. 提供多种处理脉冲数据的 **函数**。

安装
-----------

* 从 ``pypi`` 安装最新的稳定版本：

.. code-block:: console

   pip install spikezoo

* 从 ``github`` 安装最新的开发版本：

.. code-block:: console

   git clone https://github.com/chenkang455/Spike-Zoo
   cd Spike-Zoo
   python setup.py install

* 使用 ``develop`` 模式安装，便于调试和训练自定义网络：

.. code-block:: console

   git clone https://github.com/chenkang455/Spike-Zoo
   cd Spike-Zoo
   python setup.py develop

.. note::

   如有任何问题或需求，欢迎提出 ``issue``。也欢迎通过 ``Pull Request`` 将自定义方法加入到 Spike-Zoo 中。

使用教程
-----------

.. toctree::
   :maxdepth: 2
   
   快速开始
   数据集
   模型
   处理管线
   发行版本介绍
   脉冲仿真
   使用例子
   支持范围




致谢
-----------

我们的代码基于以下开源项目的构建。感谢这些项目的贡献者：

- **SpikeCV**: https://spikecv.github.io/
- **IQA-PyTorch**: https://github.com/chaofengc/IQA-PyTorch
- **BasicSR**: https://github.com/XPixelGroup/BasicSR
- **NeRFStudio**: https://github.com/nerfstudio-project/nerfstudio

同时，感谢 `rui zhao <https://github.com/ruizhao26>`_ 、`jiyuan zhang <https://github.com/Leozhangjiyuan>`_ 以及 `shiyan chen <https://github.com/hnmizuho>`_
对我们项目的帮助。

引用
-----------

.. code-block:: console

   @misc{spikezoo,
   title={{Spike-Zoo}: Spike-Zoo: A Toolbox for Spike-to-Image Reconstruction},
   author={Kang Chen and Zhiyuan Ye and Tiejun Huang and Zhaofei Yu},
   year={2025},
   howpublished = "[Online]. Available: \url{https://github.com/chenkang455/Spike-Zoo}"
   }

.. _APIs:
.. toctree::
   :maxdepth: 2
   :caption: APIs

   spikezoo.utils
   spikezoo.pipeline