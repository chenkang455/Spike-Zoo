import torch
from dataclasses import dataclass, field
import os
from spikezoo.utils.img_utils import tensor2npy, AverageMeter
from spikezoo.utils.spike_utils import load_vidar_dat
from spikezoo.metrics import cal_metric_pair, cal_metric_single
import numpy as np
import cv2
from pathlib import Path
from enum import Enum, auto
from typing import Literal
from spikezoo.metrics import metric_pair_names, metric_single_names, metric_all_names
from thop import profile
import time
from datetime import datetime
from spikezoo.utils import setup_logging, save_config
from tqdm import tqdm
from spikezoo.models import build_model_cfg, build_model_name, BaseModel, BaseModelConfig
from spikezoo.datasets import build_dataset_cfg, build_dataset_name, BaseDataset, BaseDatasetConfig, build_dataloader
from typing import Optional, Union, List
from spikezoo.pipeline.base_pipeline import Pipeline, PipelineConfig
import spikezoo as sz


@dataclass
class EnsemblePipelineConfig(PipelineConfig):
    _mode: Literal["single_mode", "multi_mode", "train_mode"] = "multi_mode"


class EnsemblePipeline(Pipeline):
    def __init__(
        self,
        cfg: PipelineConfig,
        model_cfg_list: Union[List[sz.METHOD], List[BaseModelConfig]],
        dataset_cfg: Union[sz.DATASET, BaseDatasetConfig],
    ):
        self.cfg = cfg
        self._setup_model_data(model_cfg_list, dataset_cfg)
        self._setup_pipeline()

    def _setup_model_data(self, model_cfg_list, dataset_cfg):
        """Model and Data setup."""
        # model
        self.model_list: List[BaseModel] = (
            [build_model_name(name) for name in model_cfg_list] if isinstance(model_cfg_list[0], str) else [build_model_cfg(cfg) for cfg in model_cfg_list]
        )
        self.model_list = [model.build_network(mode="eval", version=self.cfg.version) for model in self.model_list]
        torch.set_grad_enabled(False)
        # data
        self.dataset: BaseDataset = build_dataset_name(dataset_cfg) if isinstance(dataset_cfg, str) else build_dataset_cfg(dataset_cfg)
        self.dataset.build_source(split = "test")
        self.dataloader = build_dataloader(self.dataset,self.cfg)
        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def infer(self, spike, img, save_folder, rate):
        for model in self.model_list:
            self._infer_model(model, spike, img, save_folder, rate)

    def cal_params(self):
        for model in self.model_list:
            self._cal_prams_model(model)

    def cal_metrics(self):
        for model in self.model_list:
            self._cal_metrics_model(model)
