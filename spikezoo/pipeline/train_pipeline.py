import torch
from dataclasses import dataclass
import os
from spikezoo.utils.img_utils import tensor2npy
import cv2
from pathlib import Path
from typing import Literal
from tqdm import tqdm
from spikezoo.models import build_model_cfg, build_model_name, BaseModel, BaseModelConfig
from spikezoo.datasets import build_dataset_cfg, build_dataset_name, BaseDataset, BaseDatasetConfig, build_dataloader
from typing import  Union
from spikezoo.pipeline.base_pipeline import Pipeline, PipelineConfig


@dataclass
class TrainPipelineConfig(PipelineConfig):
    bs_train: int = 4
    epochs: int = 100
    lr: float = 1e-3
    num_workers: int = 4
    pin_memory: bool = False
    steps_per_save_imgs = 10
    steps_per_cal_metrics = 10
    _mode: Literal["single_mode", "multi_mode", "train_mode"] = "train_mode"


class TrainPipeline(Pipeline):
    def __init__(
        self,
        cfg: TrainPipelineConfig,
        model_cfg: Union[str, BaseModelConfig],
        dataset_cfg: Union[str, BaseDatasetConfig],
    ):
        self.cfg = cfg
        self._setup_model_data(model_cfg, dataset_cfg)
        self._setup_pipeline()
        self.model.setup_training(cfg)

    def _setup_model_data(self, model_cfg, dataset_cfg):
        """Model and Data setup."""
        # model
        self.model: BaseModel = build_model_name(model_cfg) if isinstance(model_cfg, str) else build_model_cfg(model_cfg)
        self.model = self.model.train()
        torch.set_grad_enabled(True)
        # data
        if isinstance(dataset_cfg, str):
            self.train_dataset: BaseDataset = build_dataset_name(dataset_cfg, split="train")
            self.dataset: BaseDataset = build_dataset_name(dataset_cfg, split="test")
        else:
            self.train_dataset: BaseDataset = build_dataset_cfg(dataset_cfg, split="train")
            self.dataset: BaseDataset = build_dataset_cfg(dataset_cfg, split="test")
        self.train_dataloader = build_dataloader(self.train_dataset, self.cfg)
        self.dataloader = build_dataloader(self.dataset)
        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def save_network(self, epoch):
        """Save the network."""
        save_folder = self.save_folder / Path("ckpt")
        os.makedirs(save_folder, exist_ok=True)
        self.model.save_network(save_folder / f"{epoch:06d}.pth")

    def save_visual(self, epoch):
        """Save the visual results."""
        self.logger.info("Saving visual results...")
        save_folder = self.save_folder / Path("imgs") / Path(f"{epoch:06d}")
        os.makedirs(save_folder, exist_ok=True)
        for batch_idx, batch in enumerate(tqdm(self.dataloader)):
            if batch_idx % (len(self.dataloader) // 4) != 0:
                continue 
            batch = self.model.feed_to_device(batch)
            outputs = self.model.get_outputs_dict(batch)
            visual_dict = self.model.get_visual_dict(batch, outputs)
            # save
            for key, img in visual_dict.items():
                cv2.imwrite(str(save_folder / Path(f"{batch_idx:06d}_{key}.png")), tensor2npy(img))

    def train(self):
        """Training code."""
        self.logger.info("Start Training!")
        for epoch in range(self.cfg.epochs):
            # training
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
                batch = self.model.feed_to_device(batch)
                outputs = self.model.get_outputs_dict(batch)
                loss_dict, loss_values_dict = self.model.get_loss_dict(outputs, batch)
                self.model.optimize_parameters(loss_dict)
            self.model.update_learning_rate()
            self.logger.info(f"EPOCH {epoch}/{self.cfg.epochs}: Train Loss: {loss_values_dict}")
            #  save visual results & evaluate metrics
            if epoch % self.cfg.steps_per_save_imgs == 0 or epoch == self.cfg.epochs - 1:
                self.save_visual(epoch)
            if epoch % self.cfg.steps_per_cal_metrics == 0 or epoch == self.cfg.epochs - 1:
                self.cal_metrics()
