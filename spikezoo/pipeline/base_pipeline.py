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


@dataclass
class PipelineConfig:
    "Evaluate metrics or not."
    save_metric: bool = True
    "Save recoverd images or not."
    save_img: bool = True
    "Normalizing recoverd images or not."
    save_img_norm: bool = False
    "Normalizing gt or not."
    gt_img_norm: bool = False
    "Save folder for the code running result."
    save_folder: str = ""
    "Saved experiment name."
    exp_name: str = ""
    "Metric names for evaluation."
    metric_names: List[str] = field(default_factory=lambda: ["psnr", "ssim"])
    "Different modes for the pipeline."
    _mode: Literal["single_mode", "multi_mode", "train_mode"] = "single_mode"


class Pipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        model_cfg: Union[str, BaseModelConfig],
        dataset_cfg: Union[str, BaseDatasetConfig],
    ):
        self.cfg = cfg
        self._setup_model_data(model_cfg, dataset_cfg)
        self._setup_pipeline()

    def _setup_model_data(self, model_cfg, dataset_cfg):
        """Model and Data setup."""
        # model
        self.model: BaseModel = build_model_name(model_cfg) if isinstance(model_cfg, str) else build_model_cfg(model_cfg)
        self.model = self.model.eval()
        torch.set_grad_enabled(False)
        # dataset
        self.dataset: BaseDataset = build_dataset_name(dataset_cfg) if isinstance(dataset_cfg, str) else build_dataset_cfg(dataset_cfg)
        self.dataloader = build_dataloader(self.dataset)
        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _setup_pipeline(self):
        """Pipeline setup."""
        # save folder
        self.thistime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:23]
        self.save_folder = Path(f"results") if len(self.cfg.save_folder) == 0 else self.cfg.save_folder
        mode_name = "train" if self.cfg._mode == "train_mode" else "detect"
        self.save_folder = (
            self.save_folder / Path(f"{mode_name}/{self.thistime}")
            if len(self.cfg.exp_name) == 0
            else self.save_folder / Path(f"{mode_name}/{self.cfg.exp_name}")
        )
        save_folder = self.save_folder
        os.makedirs(str(save_folder), exist_ok=True)
        # logger result
        self.logger = setup_logging(save_folder / Path("result.log"))
        self.logger.info(f"Info logs are saved on the {save_folder}/result.log")
        # pipeline config
        save_config(self.cfg, save_folder / Path("cfg_pipeline.log"))
        # model config
        if self.cfg._mode == "single_mode":
            save_config(self.model.cfg, save_folder / Path("cfg_model.log"))
        elif self.cfg._mode == "multi_mode":
            for model in self.model_list:
                save_config(model.cfg, save_folder / Path("cfg_model.log"), mode="a")
        # dataset config
        save_config(self.dataset.cfg, save_folder / Path("cfg_dataset.log"))

    def spk2img_from_dataset(self, idx=0):
        """Func---Save the recoverd image and calculate the metric from the given dataset."""
        # save folder
        self.logger.info("*********************** spk2img_from_dataset ***********************")
        save_folder = self.save_folder / Path(f"spk2img_from_dataset/{self.dataset.cfg.dataset_name}_dataset/{self.dataset.cfg.split}/{idx:06d}")
        os.makedirs(str(save_folder), exist_ok=True)

        # data process
        batch = self.dataset[idx]
        spike, img = batch["spike"], batch["img"]
        spike = spike[None].to(self.device)
        if self.dataset.cfg.with_img == True:
            img = img[None].to(self.device)
        else:
            img = None
        return self._spk2img(spike, img, save_folder)

    def spk2img_from_file(self, file_path, height = -1, width  = -1, img_path=None, remove_head=False):
        """Func---Save the recoverd image and calculate the metric from the given input file."""
        # save folder
        self.logger.info("*********************** spk2img_from_file ***********************")
        save_folder = self.save_folder / Path(f"spk2img_from_file/{os.path.basename(file_path)}")
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
        return self._spk2img(spike, img, save_folder)

    def spk2img_from_spk(self, spike, img=None):
        """Func---Save the recoverd image and calculate the metric from the given spike stream."""
        # save folder
        self.logger.info("*********************** spk2img_from_spk ***********************")
        save_folder = self.save_folder / Path(f"spk2img_from_spk/{self.thistime}")
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
        return self._spk2img(spike, img, save_folder)

    def save_imgs_from_dataset(self):
        """Func---Save all images from the given dataset."""
        for idx in range(len(self.dataset)):
            self.spk2img_from_dataset(idx=idx)

    # TODO: To be overridden
    def cal_params(self):
        """Func---Calculate the parameters/flops/latency of the given method."""
        self._cal_prams_model(self.model)

    # TODO: To be overridden
    def cal_metrics(self):
        """Func---Calculate the metric of the given method."""
        self._cal_metrics_model(self.model)

    # TODO: To be overridden
    def _spk2img(self, spike, img, save_folder):
        """Spike-to-image: spike:[bs,c,h,w] (0-1), img:[bs,1,h,w] (0-1)"""
        return self._spk2img_model(self.model, spike, img, save_folder)

    def _spk2img_model(self, model, spike, img, save_folder):
        """Spike-to-image from the given model."""
        # spike2image conversion
        model_name = model.cfg.model_name
        recon_img = model(spike)
        recon_img_copy = recon_img.clone()
        # normalization
        recon_img, img = self._post_process_img(model_name, recon_img, img)
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

    def _post_process_img(self, model_name, recon_img, gt_img):
        """Post process the reconstructed image."""
        # TFP and TFI algorithms are normalized automatically, others are normalized based on the self.cfg.use_norm
        if model_name in ["tfp", "tfi", "spikeformer", "spikeclip"]:
            recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
        elif self.cfg.save_img_norm == True:
            recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
        recon_img = recon_img.clip(0, 1)
        gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min()) if self.cfg.gt_img_norm == True and gt_img is not None else gt_img
        return recon_img, gt_img

    def _cal_metrics_model(self, model: BaseModel):
        """Calculate the metrics for the given model."""
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
            recon_img, img = self._post_process_img(model_name, recon_img, img)
            for metric_name in metrics_dict.keys():
                if metric_name in metric_pair_names:
                    metrics_dict[metric_name].update(cal_metric_pair(recon_img, img, metric_name))
                elif metric_name in metric_single_names:
                    metrics_dict[metric_name].update(cal_metric_single(recon_img, metric_name))

        # metrics self.logger.info
        self.logger.info(f"----------------------Method: {model_name.upper()}----------------------")
        for metric_name in metrics_dict.keys():
            self.logger.info(f"{metric_name.upper()}: {metrics_dict[metric_name].avg}")

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
            model(spike)
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
