Spike-Zoo 介绍
=======================
.. image:: https://github.com/chenkang455/Spike-Zoo/raw/main/imgs/spike-zoo.png
   :width: 200px
   :align: center


⚡ `Spike-Zoo <https://github.com/chenkang455/Spike-Zoo>`__ 是一个针对脉冲相机 (Spike Camera) 图像重构网络进行推理和训练的框架。
支持：

1. 使用预训练权重进行多个重构模型的快速推理。
2. 训练自定义设计的脉冲图像重构网络。
3. 提供处理脉冲数据的各类函数。

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

* 基于 ``develop`` 模式安装,方便调试训练自定义网络。

.. code-block:: console

   git clone https://github.com/chenkang455/Spike-Zoo
   cd Spike-Zoo
   python setup.py setup

