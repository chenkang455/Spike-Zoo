import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from pipeline import YourPipeline, YourPipelineConfig
from model import YourModel, YourModelConfig
from dataset import YourDataset, YourDatasetConfig
from arch import YourNet
import spikezoo as sz
# The results might be relatively low since the BASE Dataset only contains several spike-image, try download REDS_BASE and get higher value.
pipeline = YourPipeline(
    YourPipelineConfig(),
    YourModelConfig(
        model_cls_local=YourModel,
        arch_cls_local=YourNet,
        model_params={"inDim": 41},
    ),
    YourDatasetConfig(dataset_cls_local=YourDataset),
)
pipeline.train()
