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
import subprocess
import webbrowser
import time
import re
from spikezoo.utils.other_utils import set_random_seed
from spikingjelly.clock_driven import functional
from spikezoo.utils.accelerator_utils import AcceleratorManager, AcceleratorConfig
from spikezoo.utils.checkpoint_utils import CheckpointManager
from spikezoo.utils.visualization_utils import (
    VisualizationConfig,
    VisualizationManager,
    get_visualization_manager,
    log_scalar,
    log_image,
    log_config,
    flush_visualization,
    close_visualization
)


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
    optimizer_cfg: OptimizerConfig = field(default_factory=lambda: AdamOptimizerConfig(lr=1e-3))
    "Scheduler config."
    scheduler_cfg: Optional[SchedulerConfig] = None
    "Loss dict {loss_name,weight}."
    loss_weight_dict: Dict[Literal["l1", "l2"], float] = field(default_factory=lambda: {"l1": 1})
    
    # Accelerator config
    "Accelerator config for multi-GPU training."
    accelerator_cfg: Optional[Dict[str, Any]] = None
    
    # Checkpoint config
    "Enable checkpoint saving."
    enable_checkpoint: bool = True
    "Resume training from checkpoint."
    resume_from_checkpoint: Optional[str] = None
    
    # Visualization config
    "Enable visualization."
    enable_visualization: bool = True
    "Visualization config."
    visualization_cfg: Optional[VisualizationConfig] = None


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
        self._setup_accelerator()
        self._setup_checkpoint()

    def _setup_pipeline(self):
        super()._setup_pipeline()
        set_random_seed(self.cfg.seed)
        if self.cfg.use_tensorboard:
            self.writer = SummaryWriter(self.save_folder / Path(""))
            subprocess.Popen(["tensorboard", f"--logdir={self.save_folder}"])

    def _setup_model_data(self, model_cfg, dataset_cfg):
        """Model and Data setup."""
        # Update state if state management is enabled
        if self.state_manager:
            self.state_manager.transition_to_state(PipelineState.INITIALIZING)
        
        # model
        self.model: BaseModel = build_model_name(model_cfg) if isinstance(model_cfg, str) else build_model_cfg(model_cfg)
        self.model.build_network(mode="train", version="local")
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
        self.dataloader = build_dataloader(self.dataset, self.cfg)
        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Update state to ready
        if self.state_manager:
            self.state_manager.transition_to_state(PipelineState.READY)

    def _setup_training(self):
        """Setup training optimizer."""
        self.optimizer = self.cfg.optimizer_cfg.setup(self.model.net.parameters())
        self.scheduler = self.cfg.scheduler_cfg.setup(self.optimizer) if self.cfg.scheduler_cfg != None else None
        self.cnt_grad = 0
    
    def _setup_accelerator(self):
        """Setup accelerator for multi-GPU training."""
        if self.cfg.accelerator_cfg is not None:
            acc_config = AcceleratorConfig(**self.cfg.accelerator_cfg)
            self.accelerator_manager = AcceleratorManager(acc_config)
            
            # Prepare model, optimizer, and dataloaders with accelerator
            if self.scheduler is not None:
                self.model.net, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator_manager.initialize(
                    self.model.net, self.optimizer, self.train_dataloader, self.scheduler
                )
            else:
                self.model.net, self.optimizer, self.train_dataloader = self.accelerator_manager.initialize(
                    self.model.net, self.optimizer, self.train_dataloader
                )
        else:
            self.accelerator_manager = None
    
    def _setup_visualization(self):
        """Setup visualization manager."""
        if self.cfg.enable_visualization:
            # Use provided config or create default
            vis_config = self.cfg.visualization_cfg or VisualizationConfig(
                experiment_name=f"{self.cfg.exp_name or 'experiment'}_{self.thistime}",
                log_dir=str(self.save_folder / "logs")
            )
            
            # Initialize visualization manager
            self.visualization_manager = get_visualization_manager(vis_config)
            
            # Log initial configuration
            config_dict = {
                "model": str(self.model.cfg.model_name),
                "dataset": str(self.dataset.cfg.dataset_name),
                "epochs": self.cfg.epochs,
                "batch_size": self.cfg.bs_train,
                "learning_rate": self.cfg.optimizer_cfg.lr if hasattr(self.cfg.optimizer_cfg, 'lr') else 'unknown',
                "save_folder": str(self.save_folder)
            }
            self.visualization_manager.log_config(config_dict)
        else:
            self.visualization_manager = None
    
    def _setup_checkpoint(self):
        """Setup checkpoint manager."""
        if self.cfg.enable_checkpoint:
            self.checkpoint_manager = CheckpointManager(self.save_folder / Path("checkpoints"))
            
            # Resume from checkpoint if specified
            if self.cfg.resume_from_checkpoint is not None:
                self._resume_from_checkpoint(self.cfg.resume_from_checkpoint)
        else:
            self.checkpoint_manager = None
    
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
        
        # Setup visualization after save folder is created
        self._setup_visualization()

    def save_network(self, epoch):
        """Save the network."""
        save_folder = self.save_folder / Path("ckpt")
        os.makedirs(save_folder, exist_ok=True)
        
        # If using accelerator, unwrap model before saving
        if self.accelerator_manager is not None:
            unwrapped_model = self.accelerator_manager.unwrap_model(self.model.net)
            self.model.save_network(save_folder / f"{epoch:06d}.pth", unwrapped_model)
        else:
            self.model.save_network(save_folder / f"{epoch:06d}.pth")
    
    def save_checkpoint(self, epoch, step, additional_state=None):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            step: Current step
            additional_state: Additional state to save
        """
        if self.checkpoint_manager is not None:
            # If using accelerator, unwrap model before saving
            if self.accelerator_manager is not None:
                unwrapped_model = self.accelerator_manager.unwrap_model(self.model.net)
                self.checkpoint_manager.save_checkpoint(
                    unwrapped_model, self.optimizer, self.scheduler, epoch, step, 
                    additional_state=additional_state
                )
            else:
                self.checkpoint_manager.save_checkpoint(
                    self.model.net, self.optimizer, self.scheduler, epoch, step,
                    additional_state=additional_state
                )
    
    def _resume_from_checkpoint(self, checkpoint_path):
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if self.checkpoint_manager is not None:
            checkpoint = self.checkpoint_manager.load_checkpoint(
                self.model.net, self.optimizer, self.scheduler, checkpoint_path
            )
            
            # Update training state
            self.start_epoch = checkpoint['epoch'] + 1
            self.start_step = checkpoint['step']
            
            self.logger.info(f"Resumed from checkpoint: {checkpoint_path}")
            self.logger.info(f"Starting from epoch {self.start_epoch}, step {self.start_step}")
        else:
            self.start_epoch = 0
            self.start_step = 0

    def train(self):
        """Training code."""
        self.logger.info("Start Training!")
        
        # Update state if state management is enabled
        if self.state_manager:
            self.state_manager.transition_to_state(PipelineState.TRAINING)
        
        try:
            # Set starting epoch and step
            start_epoch = getattr(self, 'start_epoch', 0)
            start_step = getattr(self, 'start_step', 0)
            step_count = start_step
            
            for epoch in range(start_epoch, self.cfg.epochs):
                # training
                for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
                    batch = self.model.feed_to_device(batch)
                    outputs = self.model.get_outputs_dict(batch)
                    loss_dict, loss_values_dict = self.model.get_loss_dict(outputs, batch, self.cfg.loss_weight_dict)
                    self.optimize_parameters(loss_dict, batch_idx == len(self.train_dataloader) - 1)
                    
                    # Increment step count
                    step_count += 1
                    
                    # Save checkpoint periodically
                    if self.checkpoint_manager is not None and step_count % (self.cfg.steps_per_save_ckpt * 10) == 0:
                        self.save_checkpoint(epoch, step_count)
                
                self.update_learning_rate()
                self.write_log_train(epoch, loss_values_dict)

                # Log training metrics to visualization
                if self.visualization_manager is not None:
                    try:
                        # Log average loss
                        avg_loss = sum(loss_values_dict.values()) / len(loss_values_dict)
                        self.visualization_manager.log_scalar("train/avg_loss", avg_loss, epoch)
                        
                        # Log individual losses
                        for loss_name, loss_value in loss_values_dict.items():
                            self.visualization_manager.log_scalar(f"train/{loss_name}", loss_value, epoch)
                    except Exception as e:
                        self.logger.warning(f"Failed to log training metrics to visualization: {e}")

                #  save visual results & save ckpt & evaluate metrics
                with torch.no_grad():
                    if epoch % self.cfg.steps_per_save_imgs == 0 or epoch == self.cfg.epochs - 1:
                        self.save_visual(epoch)
                    if epoch % self.cfg.steps_per_save_ckpt == 0 or epoch == self.cfg.epochs - 1:
                        self.save_network(epoch)
                        # Save checkpoint when saving network
                        if self.checkpoint_manager is not None:
                            self.save_checkpoint(epoch, step_count)
                    if epoch % self.cfg.steps_per_cal_metrics == 0 or epoch == self.cfg.epochs - 1:
                        metrics_dict = self.cal_metrics()
                        self.write_log_test(epoch, metrics_dict)
                        
                        # Log evaluation metrics to visualization
                        if self.visualization_manager is not None:
                            try:
                                for metric_name, metric_value in metrics_dict.items():
                                    self.visualization_manager.log_scalar(f"eval/{metric_name}", metric_value.avg, epoch)
                            except Exception as e:
                                self.logger.warning(f"Failed to log evaluation metrics to visualization: {e}")
            
            # Update state back to ready
            if self.state_manager:
                self.state_manager.transition_to_state(PipelineState.READY)
        except Exception as e:
            # Update state to error
            if self.state_manager:
                self.state_manager.transition_to_state(PipelineState.ERROR)
            self.logger.error(f"Training error: {e}")
            raise
        finally:
            # Close visualization manager
            if self.visualization_manager is not None:
                try:
                    self.visualization_manager.close()
                except Exception as e:
                    self.logger.warning(f"Failed to close visualization manager: {e}")

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
                
                # Log images to visualization manager
                if self.visualization_manager is not None and batch_idx == 0:
                    try:
                        # Convert image for visualization
                        vis_img = img[0].detach().cpu().numpy()
                        if vis_img.ndim == 3 and vis_img.shape[0] in [1, 3, 4]:
                            # Channel-first format (C, H, W) -> (H, W, C)
                            vis_img = np.transpose(vis_img, (1, 2, 0))
                        elif vis_img.ndim == 2:
                            # Grayscale (H, W) -> (H, W, 1)
                            vis_img = np.expand_dims(vis_img, axis=-1)
                        
                        self.visualization_manager.log_image(f"imgs/{key}", vis_img, epoch)
                    except Exception as e:
                        self.logger.warning(f"Failed to log image {key} to visualization: {e}")

    def optimize_parameters(self, loss_dict, final_flag):
        """Optimize the parameters from the loss_dict."""
        loss = functools.reduce(torch.add, loss_dict.values())
        step_grad = self.cfg.steps_grad_accumulation
        
        # Use accelerator if available
        if self.accelerator_manager is not None:
            # for snn methods
            if self.model.cfg.model_name == "ssir":
                if self.cnt_grad % step_grad != step_grad - 1 and final_flag == False:
                    self.accelerator_manager.backward(loss)
                    self.cnt_grad += 1
                else:
                    self.accelerator_manager.backward(loss)
                    self.accelerator_manager.step(self.optimizer)
                    self.accelerator_manager.zero_grad(self.optimizer)
                    self._state_reset(self.model)
                    self.cnt_grad = 0
            else:
                # for cnn methods
                self.accelerator_manager.zero_grad(self.optimizer)
                self.accelerator_manager.backward(loss)
                self.accelerator_manager.step(self.optimizer)
        else:
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
        if self.scheduler is not None:
            # If using accelerator, let it handle the scheduler step
            if self.accelerator_manager is not None:
                # Accelerator handles scheduler step automatically in certain cases
                # For manual control, we can still call step()
                self.scheduler.step()
            else:
                self.scheduler.step()

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
