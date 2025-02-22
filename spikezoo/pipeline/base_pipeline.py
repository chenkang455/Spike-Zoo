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
import shutil
from spikingjelly.clock_driven import functional
import spikezoo as sz


@dataclass
class PipelineConfig:
    "Loading weights from local or version on the url."
    version: Literal["local", "v010", "v023"] = "local"
    "Save folder for the code running result."
    save_folder: str = ""
    "Saved experiment name."
    exp_name: str = ""
    "Evaluate metrics or not."
    save_metric: bool = True
    "Metric names for evaluation."
    metric_names: List[str] = field(default_factory=lambda: ["psnr", "ssim", "niqe", "brisque"])
    "Save recoverd images or not."
    save_img: bool = True
    "Normalizing recoverd images and gt or not."
    img_norm: bool = False
    "Batch size for the test dataloader."
    bs_test: int = 1
    "Num_workers for the test dataloader."
    nw_test: int = 0
    "Pin_memory true or false for the dataloader."
    pin_memory: bool = False
    "Different modes for the pipeline."
    _mode: Literal["single_mode", "multi_mode", "train_mode"] = "single_mode"


class Pipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        model_cfg: Union[sz.METHOD, BaseModelConfig],
        dataset_cfg: Union[sz.DATASET, BaseDatasetConfig],
    ):
        self.cfg = cfg
        self._setup_model_data(model_cfg, dataset_cfg)
        self._setup_pipeline()

    def _setup_model_data(self, model_cfg, dataset_cfg):
        """Model and Data setup."""
        print("Model and dataset is setting up...")
        # model [1] build the model. [2] build the network.
        self.model: BaseModel = build_model_name(model_cfg) if isinstance(model_cfg, str) else build_model_cfg(model_cfg)
        self.model.build_network(mode="eval", version=self.cfg.version)
        torch.set_grad_enabled(False)
        # dataset
        self.dataset: BaseDataset = build_dataset_name(dataset_cfg) if isinstance(dataset_cfg, str) else build_dataset_cfg(dataset_cfg)
        self.dataset.build_source(split="test")
        self.dataloader = build_dataloader(self.dataset,self.cfg)
        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _setup_pipeline(self):
        """Pipeline setup."""
        print("Pipeline is setting up...")
        # save folder
        self.thistime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:23]
        self.save_folder = Path(f"results") if len(self.cfg.save_folder) == 0 else self.cfg.save_folder
        mode_name = "train" if self.cfg._mode == "train_mode" else "detect"
        self.save_folder = (
            self.save_folder / Path(f"{mode_name}/{self.thistime}")
            if len(self.cfg.exp_name) == 0
            else self.save_folder / Path(f"{mode_name}/{self.cfg.exp_name}")
        )
        # remove and establish folder
        save_folder = str(self.save_folder)
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)
        save_folder = Path(save_folder)
        # logger result
        self.logger = setup_logging(save_folder / Path("result.log"))
        self.logger.info(f"Info logs are saved on the {save_folder}/result.log")
        # pipeline config
        save_config(self.cfg, save_folder / Path("cfg_pipeline.log"))
        # model config
        if self.cfg._mode in ["single_mode", "train_mode"]:
            save_config(self.model.cfg, save_folder / Path("cfg_model.log"))
        elif self.cfg._mode == "multi_mode":
            for model in self.model_list:
                save_config(model.cfg, save_folder / Path("cfg_model.log"), mode="a")
        # dataset config
        save_config(self.dataset.cfg, save_folder / Path("cfg_dataset.log"))

    def infer_from_dataset(self, idx=0):
        """Function I---Save the recoverd image and calculate the metric from the given dataset."""
        # save folder
        self.logger.info("*********************** infer_from_dataset ***********************")
        save_folder = self.save_folder / Path(f"infer_from_dataset/{self.dataset.cfg.dataset_name}_dataset/{self.dataset.split}/{idx:06d}")
        os.makedirs(str(save_folder), exist_ok=True)

        # data process
        # todo
        batch = self.dataset[idx]
        spike, img, rate = batch["spike"], batch["gt_img"], batch["rate"]
        spike = spike[None].to(self.device)
        if self.dataset.cfg.with_img == True:
            img = img[None].to(self.device)
        else:
            img = None
        return self.infer(spike, img, save_folder, rate)

    def infer_from_file(self, file_path, height=-1, width=-1, rate=1, img_path=None, remove_head=False):
        """Function II---Save the recoverd image and calculate the metric from the given input file."""
        # save folder
        self.logger.info("*********************** infer_from_file ***********************")
        save_folder = self.save_folder / Path(f"infer_from_file/{os.path.basename(file_path)}")
        os.makedirs(str(save_folder), exist_ok=True)

        # load spike from .dat
        if file_path.endswith(".dat"):
            spike = load_vidar_dat(file_path, height, width, remove_head)
        # load spike from .npz from UHSR
        elif file_path.endswith("npz"):
            spike = np.load(file_path)["spk"].astype(np.float32)[:, 13:237, 13:237]
        else:
            raise RuntimeError("Not recognized spike input file.")
        # load img from .png/.jpg image file
        if img_path is not None:
            img = cv2.imread(img_path)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img / 255).astype(np.float32)
            img = torch.from_numpy(img)[None, None].to(self.device)
        else:
            img = img_path
        spike = torch.from_numpy(spike)[None].to(self.device)
        return self.infer(spike, img, save_folder, rate)

    def infer_from_spk(self, spike, rate=1, img=None):
        """Function III---Save the recoverd image and calculate the metric from the given spike stream."""
        # save folder
        self.logger.info("*********************** infer_from_spk ***********************")
        save_folder = self.save_folder / Path(f"infer_from_spk")
        os.makedirs(str(save_folder), exist_ok=True)

        # spike process
        if isinstance(spike, np.ndarray):
            spike = torch.from_numpy(spike)
        spike = spike.to(self.device)
        # [c,h,w] -> [1,c,w,h]
        if spike.dim() == 3:
            spike = spike[None]
        spike = spike.float()
        # img process
        if img is not None:
            if isinstance(img, np.ndarray):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                img = (img / 255).astype(np.float32)
                img = torch.from_numpy(img)[None, None].to(self.device)
            else:
                raise RuntimeError("Not recognized image input type.")
        return self.infer(spike, img, save_folder, rate)

    # TODO: To be overridden
    def infer(self, spike, img, save_folder, rate):
        """Function IV---Spike-to-image conversion interface, input data format: spike [bs,c,h,w] (0-1), img [bs,1,h,w] (0-1)"""
        return self._infer_model(self.model, spike, img, save_folder, rate)

    def save_imgs_from_dataset(self):
        """Function V---Save all images from the given dataset."""
        base_setting = self.cfg.save_metric
        self.cfg.save_metric = False
        for idx in range(len(self.dataset)):
            self.infer_from_dataset(idx=idx)
        self.cfg.save_metric = base_setting

    # TODO: To be overridden
    def cal_params(self):
        """Function VI---Calculate the parameters/flops/latency of the given method."""
        self._cal_prams_model(self.model)

    # TODO: To be overridden
    def cal_metrics(self):
        """Function VII---Calculate the metric of the given method."""
        return self._cal_metrics_model(self.model)

    def _infer_model(self, model, spike, img, save_folder, rate):
        """Spike-to-image from the given model."""
        # spike2image conversion
        model_name = model.cfg.model_name
        recon_img = model.spk2img(spike)
        recon_img_copy = recon_img.clone()
        # normalization
        recon_img, img = self._post_process_img(recon_img, model_name, rate), self._post_process_img(img, None, 1)
        self._state_reset(model)
        # metric
        if self.cfg.save_metric == True:
            self.logger.info(f"----------------------Method: {model_name.upper()}----------------------")
            # paired metric
            for metric_name in self.cfg.metric_names:
                if img is not None and metric_name in metric_pair_names:
                    self.logger.info(f"{metric_name.upper()}: {cal_metric_pair(recon_img,img,metric_name)}")
                elif metric_name in metric_single_names:
                    self.logger.info(f"{metric_name.upper()}: {cal_metric_single(recon_img,metric_name)}")
                else:
                    self.logger.info(f"{metric_name.upper()} not calculated since no ground truth provided.")

        # visual
        if self.cfg.save_img == True:
            recon_img = tensor2npy(recon_img[0, 0])
            cv2.imwrite(f"{save_folder}/{model.cfg.model_name}.png", recon_img)
            if img is not None:
                img = tensor2npy(img[0, 0])
                cv2.imwrite(f"{save_folder}/sharp_img.png", img)
            self.logger.info(f"Images are saved on the {save_folder}")
        return recon_img_copy

    def _post_process_img(self, recon_img, model_name, rate=1):
        """Post process the reconstructed image."""
        # With no GT
        if recon_img == None:
            return None
        # spikeclip is normalized automatically
        if model_name in ["spikeclip"] or self.cfg.img_norm == True:
            recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
        else:
            recon_img = recon_img / rate
        recon_img = recon_img.clip(0, 1)
        return recon_img

    def _cal_metrics_model(self, model: BaseModel):
        """Calculate the metrics for the given model."""
        # metric state reset (since get_outputs_dict from the training state is utilized)
        model_state = model.net.training
        model.net.training = True
        # metrics construct
        model_name = model.cfg.model_name
        metrics_dict = {}
        for metric_name in self.cfg.metric_names:
            if (self.dataset.cfg.with_img == True) or (metric_name in metric_single_names):
                metrics_dict[metric_name] = AverageMeter()

        # metrics calculate
        for batch_idx, batch in enumerate(tqdm(self.dataloader)):
            batch = model.feed_to_device(batch)
            outputs = model.get_outputs_dict(batch)
            recon_img, img = model.get_paired_imgs(batch, outputs)
            recon_img, img = self._post_process_img(recon_img, model_name), self._post_process_img(img, "gt")
            for metric_name in metrics_dict.keys():
                if metric_name in metric_pair_names:
                    metrics_dict[metric_name].update(cal_metric_pair(recon_img, img, metric_name))
                elif metric_name in metric_single_names:
                    metrics_dict[metric_name].update(cal_metric_single(recon_img, metric_name))
        self._state_reset(model)
        model.net.training = model_state
        # metrics self.logger.info
        self.logger.info(f"----------------------Method: {model_name.upper()}----------------------")
        for metric_name in metrics_dict.keys():
            self.logger.info(f"{metric_name.upper()}: {metrics_dict[metric_name].avg}")
        return metrics_dict

    def _cal_prams_model(self, model):
        """Calculate the parameters for the given model."""
        network = model.net
        model_name = model.cfg.model_name.upper()
        # params
        params = sum(p.numel() for p in network.parameters())
        # latency
        spike = torch.zeros((1, 200, 250, 400)).cuda()
        start_time = time.time()
        for _ in range(100):
            model.spk2img(spike)
        latency = (time.time() - start_time) / 100
        # flop # todo thop bug for BSF
        flops, _ = profile((model), inputs=(spike,))
        re_msg = (
            "Total params: %.4fM" % (params / 1e6),
            "FLOPs:" + str(flops / 1e9) + "{}".format("G"),
            "Latency: {:.6f} seconds".format(latency),
        )
        self.logger.info(f"----------------------Method: {model_name}----------------------")
        self.logger.info(re_msg)

    def _state_reset(self, model):
        """State reset for the snn-based method."""
        if model.cfg.model_name == "ssir":
            functional.reset_net(model.net)
