import torch
from dataclasses import dataclass, field
import os
from spikezoo.utils.img_utils import tensor2npy
import cv2
from pathlib import Path
from typing import Literal, Dict
from tqdm import tqdm
from spikezoo.models import build_model_cfg, build_model_name, BaseModel, BaseModelConfig
from spikezoo.datasets import build_dataset_cfg, build_dataset_name, BaseDataset, BaseDatasetConfig, build_dataloader
from typing import Union, List, Optional
from spikezoo.pipeline.base_pipeline import Pipeline, PipelineConfig
import torch.nn as nn
import torch.optim as optimizer
import torch.optim.lr_scheduler as lr_scheduler
import functools
from spikezoo.utils.optimizer_utils import OptimizerConfig, AdamOptimizerConfig
from spikezoo.utils.scheduler_utils import SchedulerConfig, MultiStepSchedulerConfig, CosineAnnealingLRConfig
from torch.utils.tensorboard import SummaryWriter
import subprocess
import webbrowser
import time
import re
from spikezoo.utils.other_utils import set_random_seed
from spikingjelly.clock_driven import functional


@dataclass
class TrainPipelineConfig(PipelineConfig):
    # parameters setting
    "Training epochs."
    epochs: int = 10
    "Steps per to save images."
    steps_per_save_imgs: int = 10
    "Steps per to save model weights."
    steps_per_save_ckpt: int = 10
    "Steps per to calculate the metrics."
    steps_per_cal_metrics: int = 10
    "Step for gradient accumulation. (for snn methods)"
    steps_grad_accumulation: int = 4
    "Pipeline mode."
    _mode: Literal["single_mode", "multi_mode", "train_mode"] = "train_mode"
    "Use tensorboard or not"
    use_tensorboard: bool = True
    "Random seed."
    seed: int = 521
    # dataloader setting
    "Batch size for the train dataloader."
    bs_train: int = 8
    "Num_workers for the train dataloader."
    nw_train: int = 4

    # train setting - optimizer & scheduler & loss_dict
    "Optimizer config."
    optimizer_cfg: OptimizerConfig = AdamOptimizerConfig(lr=1e-3)
    "Scheduler config."
    scheduler_cfg: Optional[SchedulerConfig] = None
    "Loss dict {loss_name,weight}."
    loss_weight_dict: Dict[Literal["l1", "l2"], float] = field(default_factory=lambda: {"l1": 1})


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
        self._setup_training()

    def _setup_pipeline(self):
        super()._setup_pipeline()
        set_random_seed(self.cfg.seed)
        if self.cfg.use_tensorboard:
            self.writer = SummaryWriter(self.save_folder / Path(""))
            subprocess.Popen(["tensorboard", f"--logdir={self.save_folder}"])

    def _setup_model_data(self, model_cfg, dataset_cfg):
        """Model and Data setup."""
        # model
        self.model: BaseModel = build_model_name(model_cfg) if isinstance(model_cfg, str) else build_model_cfg(model_cfg)
        self.model.build_network(mode = "train",version="local")
        torch.set_grad_enabled(True)
        # data
        if isinstance(dataset_cfg, str):
            self.train_dataset: BaseDataset = build_dataset_name(dataset_cfg)
            self.dataset: BaseDataset = build_dataset_name(dataset_cfg)
        else:
            self.train_dataset: BaseDataset = build_dataset_cfg(dataset_cfg)
            self.dataset: BaseDataset = build_dataset_cfg(dataset_cfg)
        self.train_dataset.build_source("train")
        self.dataset.build_source("test")
        self.train_dataloader = build_dataloader(self.train_dataset, self.cfg)
        self.dataloader = build_dataloader(self.dataset,self.cfg)
        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _setup_training(self):
        """Setup training optimizer."""
        self.optimizer = self.cfg.optimizer_cfg.setup(self.model.net.parameters())
        self.scheduler = self.cfg.scheduler_cfg.setup(self.optimizer) if self.cfg.scheduler_cfg != None else None
        self.cnt_grad = 0

    def save_network(self, epoch):
        """Save the network."""
        save_folder = self.save_folder / Path("ckpt")
        os.makedirs(save_folder, exist_ok=True)
        self.model.save_network(save_folder / f"{epoch:06d}.pth")

    def train(self):
        """Training code."""
        self.logger.info("Start Training!")
        for epoch in range(self.cfg.epochs):
            # training
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
                batch = self.model.feed_to_device(batch)
                outputs = self.model.get_outputs_dict(batch)
                loss_dict, loss_values_dict = self.model.get_loss_dict(outputs, batch, self.cfg.loss_weight_dict)
                self.optimize_parameters(loss_dict, batch_idx == len(self.train_dataloader) - 1)
            self.update_learning_rate()
            self.write_log_train(epoch, loss_values_dict)

            #  save visual results & save ckpt & evaluate metrics
            with torch.no_grad():
                if epoch % self.cfg.steps_per_save_imgs == 0 or epoch == self.cfg.epochs - 1:
                    self.save_visual(epoch)
                if epoch % self.cfg.steps_per_save_ckpt == 0 or epoch == self.cfg.epochs - 1:
                    self.save_network(epoch)
                if epoch % self.cfg.steps_per_cal_metrics == 0 or epoch == self.cfg.epochs - 1:
                    metrics_dict = self.cal_metrics()
                    self.write_log_test(epoch, metrics_dict)

    def save_visual(self, epoch):
        """Save the visual results."""
        self.logger.info("Saving visual results...")
        save_folder = self.save_folder / Path("imgs") / Path(f"{epoch:06d}")
        os.makedirs(save_folder, exist_ok=True)
        for batch_idx, batch in enumerate(tqdm(self.dataloader)):
            if batch_idx % (len(self.dataloader) // 3) != 0:
                continue
            batch = self.model.feed_to_device(batch)
            outputs = self.model.get_outputs_dict(batch)
            visual_dict = self.model.get_visual_dict(batch, outputs)
            self._state_reset(self.model)
            # save
            for key, img in visual_dict.items():
                img = self._post_process_img(img, model_name=self.model.cfg.model_name)
                cv2.imwrite(str(save_folder / Path(f"{batch_idx:06d}_{key}.png")), tensor2npy(img))
                if self.cfg.use_tensorboard == True and batch_idx == 0:
                    self.writer.add_image(f"imgs/{key}", img[0].detach().cpu(), epoch)

    def optimize_parameters(self, loss_dict, final_flag):
        """Optimize the parameters from the loss_dict."""
        loss = functools.reduce(torch.add, loss_dict.values())
        step_grad = self.cfg.steps_grad_accumulation
        # for snn methods
        if self.model.cfg.model_name == "ssir":
            if self.cnt_grad % step_grad != step_grad - 1 and final_flag == False:
                loss.backward(retain_graph=True)
                self.cnt_grad += 1
            else:
                loss.backward(retain_graph=False)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._state_reset(self.model)
                self.cnt_grad = 0
        else:
            # for cnn methods
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_learning_rate(self):
        """Update the learning rate."""
        self.scheduler.step() if self.cfg.scheduler_cfg != None else None

    def write_log_train(self, epoch, loss_values_dict):
        """Write the train log information."""
        self.logger.info(f"EPOCH {epoch}/{self.cfg.epochs}: Train Loss: {loss_values_dict}")
        if self.cfg.use_tensorboard:
            for name, val in loss_values_dict.items():
                self.writer.add_scalar(f"Loss/{name}", val, epoch)
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar(f"Loss/lr", lr, epoch)

    def write_log_test(self, epoch, metrics_dict):
        """Write the test log information."""
        if self.cfg.use_tensorboard:
            for name in metrics_dict.keys():
                self.writer.add_scalar(f"Test/{name}", metrics_dict[name].avg, epoch)
