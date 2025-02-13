Spike-Zoo 介绍
=======================
⚡ `Spike-Zoo <https://github.com/chenkang455/Spike-Zoo>`__ 是一个旨在辅助开发脉冲相机图像重构网络的框架。
主要特点包括：

* 使用预训练权重进行多个重构模型的快速推理。
* 支持训练自定义设计的脉冲图像重构网络。
* 提供处理脉冲数据的各类函数。

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

* 如果想基于Spike-Zoo开发自己的脉冲重构网络,建议通过如下方式安装,以方便对网络以及训练脚本进行实时更改

.. code-block:: console

   git clone https://github.com/chenkang455/Spike-Zoo
   cd Spike-Zoo
   python setup.py setup

