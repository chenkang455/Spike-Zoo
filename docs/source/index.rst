Spike-Zoo 介绍
=======================
⚡ `Spike-Zoo <https://github.com/chenkang455/Spike-Zoo>`__ 是一个针对脉冲相机 (Spike Camera) 图像重构网络进行推理和训练的框架。
支持：
1. 使用预训练权重进行多个重构模型的快速 **推理**。
2. **训练** 自定义设计的脉冲图像重构网络。
3. 提供处理脉冲数据的各类 **函数**。

安装
-----------

* 从 ``pypi`` 直接安装最新的稳定版本：

.. code-block:: console

   pip install spikezoo


* 从 ``github`` 安装最新开发版本：

.. code-block:: console

   git clone https://github.com/chenkang455/Spike-Zoo
   cd Spike-Zoo
   python setup.py install

* 基于 ``develop`` 模式安装，方便调试训练自定义网络。

.. code-block:: console

   git clone https://github.com/chenkang455/Spike-Zoo
   cd Spike-Zoo
   python setup.py setup

.. note::

   有任何BUG、需求或新的脉冲重构方法，欢迎提出 ``issue`` 或 ``Pull Request``。 

使用教程
-----------

.. toctree::
   :maxdepth: 2
   
   快速开始
   数据集
   模型
   处理管线
   函数库
   脉冲仿真
   使用例子
   

致谢
-----------

我们的代码基于以下开源项目的构建。感谢这些项目的贡献者：

- **SpikeCV**: https://spikecv.github.io/
- **IQA-PyTorch**: https://github.com/chaofengc/IQA-PyTorch
- **BasicSR**: https://github.com/XPixelGroup/BasicSR
- **NeRFStudio**: https://github.com/nerfstudio-project/nerfstudio

同时，感谢 `rui zhao <https://github.com/ruizhao26>`_ 、`jiyuan zhang <https://github.com/Leozhangjiyuan>`_ 以及 `shiyan chen <https://github.com/hnmizuho>`_
对我们项目的帮助。

