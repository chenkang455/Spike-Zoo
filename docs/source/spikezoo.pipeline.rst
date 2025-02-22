
.. _api_pipeline:

spikezoo.pipeline 
-------------------

.. py:function:: __init__(cfg, model_cfg, dataset_cfg)

    ``pipeline`` 初始化函数，需要分别指定 ``pipeline``, ``model`` 和 ``dataset`` 的配置参数

    :param cfg: 管线实例化参数
    :param model: 支持输入模型实例化参数或模型名称
    :param dataset: 支持输入数据集实例化参数或数据集名称

    .. code-block:: python

    from spikezoo.pipeline import Pipeline, PipelineConfig
    import spikezoo as sz
    pipeline = Pipeline(
        cfg=PipelineConfig(save_folder="results",version="v023"),
        model_cfg=sz.METHOD.BASE,
        dataset_cfg=sz.DATASET.BASE
    )

.. py:function:: infer_from_dataset(self, idx=0)

    直接从 ``pipeline`` 中给定的数据集进行推理，包含指标测试和性能指标计算两部分，结果保存在输出路径的 ``infer_from_dataset`` 文件夹下。

    :param idx: 管线实例化参数
    :type idx: int

    .. code-block:: python

    pipeline.infer_from_dataset(idx=0)

.. py:function:: infer_from_file(self, file_path, height=-1, width=-1, rate=1, img_path=None, remove_head=False)

    从给定输入脉冲文件进行推理，包含指标测试和性能指标计算两部分，结果保存在输出路径的 ``infer_from_file`` 文件夹下。

    :param file_path: 输入脉冲文件路径
    :type file_path: str
    :param height: 输入脉冲的高度
    :type height: int
    :param width: 输入脉冲的宽度
    :type width: int
    :param rate: 脉冲转化系数，对重构图像亮度进行矫正， ``img = img / rate``。
    :type rate: float
    :param img_path: 输入清晰图像的路径，用于脉冲重构图像的指标（如果不提供只会计算非参考指标）
    :type img_path: str
    :param remove_head: 是否移除帧头（针对真实拍摄数据）
    :type remove_head: bool

    .. code-block:: python

    # without gt (for calculating psnr,ssim and lpips)
    pipeline.infer_from_file(file_path='data/data.dat',width=400,height=250)
    # with gt (only niqe and brisque)
    pipeline.infer_from_file(file_path = 'data/data.dat',width = 400,height=250,img_path= "data/data.png",rate = 0.6)
    # for real-captured data with frame index
    pipeline.infer_from_file(file_path='realworld.dat',width=416,height=250,remove_head=True)

.. py:function:: infer_from_spk(self, spike, rate=1, img=None)

    利用给定输入脉冲进行推理，包含指标测试和性能指标计算两部分，结果保存在输出路径的 ``infer_from_spk`` 文件夹下。

    :param spike: 输入脉冲
    :type spike: array or tensor
    :param rate: 脉冲转化系数，对重构图像亮度进行矫正， ``img = img / rate``。
    :type rate: float
    :param img: 输入清晰图像，用于脉冲重构图像的指标（如果不提供只会计算非参考指标）
    :type img: array

    .. code-block:: python

    import spikezoo as sz
    import cv2
    spike = sz.load_vidar_dat("data/data.dat",width=400,height=250)
    img = cv2.imread("data/data.png")
    # without gt (for calculating psnr,ssim,lpips,niqe and brisque)
    pipeline.infer_from_spk(spike)

    # with gt (only niqe and brisque)
    pipeline.infer_from_spk(spike,img,rate = 0.6)

