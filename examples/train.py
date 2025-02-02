from spikezoo.pipeline import TrainPipelineConfig, TrainPipeline
from spikezoo.datasets.reds_small_dataset import REDS_Small_Config
pipeline = TrainPipeline(
    cfg=TrainPipelineConfig(
        save_folder="results", 
        bs_train = 4,
        num_workers = 4,
        lr = 1e-3,
        epochs = 10,
        exp_name="Base"),
    dataset_cfg=REDS_Small_Config(root_dir = "path/REDS_Small"),
    model_cfg="base",
)
pipeline.train()

