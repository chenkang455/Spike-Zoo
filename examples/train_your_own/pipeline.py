from spikezoo.pipeline import TrainPipeline, TrainPipelineConfig
from dataclasses import dataclass

@dataclass
class YourPipelineConfig(TrainPipelineConfig):
    epochs: int = 10


class YourPipeline(TrainPipeline):
    def __init__(self, cfg,model_cfg,dataset_cfg):
        super(YourPipeline, self).__init__(cfg,model_cfg,dataset_cfg)

