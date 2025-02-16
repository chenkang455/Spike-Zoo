spikezoo.utils 
-------------------

.. py:function:: load_vidar_dat(filename, height, width,remove_head=False,version: Literal["python", "cpp"] = "cpp", out_format: Literal["array", "tensor"] = "array")

  读取脉冲数据函数。

  :param filename: 脉冲数据路径
  :type filename: string
  :param height: 读取脉冲高度
  :type height: int
  :param width: 读取脉冲宽度
  :type width: int
  :param remove_head: 是否去除帧头
  :type remove_head: bool
  :param version: 选择cpp版还是python版实现，但目前只支持在linux系统。
  :type version: str
  :param out_format: 选择输出np.array还是torch.tensor
  :type out_format: str
  :returns: 脉冲序列 (尺寸为 K * H * W)
  :rtype: array or tensor

  根据以下代码实测表明，``cpp`` 版比 ``python`` 版快10倍左右。

  .. code-block:: python

    from spikezoo.utils.spike_utils import load_vidar_dat
    import time
    print("cpp")
    start = time.time()
    for _ in range(10):
        result = load_vidar_dat("data/data.dat",version="cpp",width = 400,height = 250,out_format="array")
    print(time.time() - start)

    print("python")
    start = time.time()
    for _ in range(10):
        result = load_vidar_dat("data/data.dat",version="python",width = 400,height = 250,out_format="array")
    print(time.time() - start)

