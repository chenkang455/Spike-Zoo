import torch
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal, Dict, Optional, Union
from spikezoo.utils.img_utils import tensor2npy
import cv2
from tqdm import tqdm
from spikezoo.models import BaseModel, BaseModelConfig
from spikezoo.models.model_registry import create_model, get_config_class
from spikezoo.datasets import build_dataset_cfg, build_dataset_name, BaseDataset, BaseDatasetConfig, build_dataloader
import torch.nn as nn
import torch.optim as optimizer
import torch.optim.lr_scheduler as lr_scheduler
import functools
from spikezoo.utils.optimizer_utils import OptimizerConfig, AdamOptimizerConfig
from spikezoo.utils.scheduler_utils import SchedulerConfig, MultiStepSchedulerConfig, CosineAnnealingLRConfig
from spikezoo.utils.other_utils import set_random_seed
from spikingjelly.clock_driven import functional
from spikezoo.pipeline.base_pipeline import Pipeline, PipelineConfig
from spikezoo.pipeline.train_pipeline import TrainPipelineConfig
import numpy as np
import shutil
from datetime import datetime


@dataclass
class RecognitionPipelineConfig(TrainPipelineConfig):
    # Recognition specific parameters
    "Recognition task type"
    task_type: Literal["classification", "detection", "tracking"] = "classification"
    "Number of classes for classification task"
    num_classes: int = 10
    "Save predictions or not"
    save_predictions: bool = True
    "Evaluation metrics for recognition task"
    eval_metrics: list = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    "Confidence threshold for detection tasks"
    confidence_threshold: float = 0.5
    "IoU threshold for detection tasks"
    iou_threshold: float = 0.5


class RecognitionPipeline(Pipeline):
    def __init__(
        self,
        cfg: RecognitionPipelineConfig,
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
            self.writer = SummaryWriter(self.save_folder)
            subprocess.Popen(["tensorboard", f"--logdir={self.save_folder}"])

    def _setup_model_data(self, model_cfg, dataset_cfg):
        """Model and Data setup for recognition task."""
        # Update state if state management is enabled
        if self.state_manager:
            self.state_manager.transition_to_state(PipelineState.INITIALIZING)
        
        # model
        if isinstance(model_cfg, str):
            # For string model names, create model using registry
            model_config_class = get_config_class(model_cfg)
            if model_config_class:
                model_config = model_config_class()
                self.model: BaseModel = create_model(model_cfg, model_config)
            else:
                # Fallback to old method if registry doesn't have the model
                self.model: BaseModel = build_model_name(model_cfg)
        else:
            # For config objects, we need to determine the model name and use registry
            # This is a simplified approach - in practice, you might need to map configs to model names
            # For now, we'll assume the config has a model_name attribute or fallback to old method
            if hasattr(model_cfg, 'model_name'):
                self.model: BaseModel = create_model(model_cfg.model_name, model_cfg)
            else:
                # Fallback to old method if we can't determine model name
                self.model: BaseModel = build_model_cfg(model_cfg)
        
        if self.model:
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
        """Setup training optimizer for recognition task."""
        self.optimizer = self.cfg.optimizer_cfg.setup(self.model.net.parameters())
        self.scheduler = self.cfg.scheduler_cfg.setup(self.optimizer) if self.cfg.scheduler_cfg != None else None
        self.cnt_grad = 0

    def train(self):
        """Training code for recognition task."""
        self.logger.info("Start Recognition Training!")
        
        # Update state if state management is enabled
        if self.state_manager:
            self.state_manager.transition_to_state(PipelineState.TRAINING)
        
        try:
            for epoch in range(self.cfg.epochs):
                # Training phase
                self.model.net.train()
                train_loss = 0.0
                correct_predictions = 0
                total_samples = 0
                
                for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
                    batch = self.model.feed_to_device(batch)
                    outputs = self.model.get_outputs_dict(batch)
                    loss_dict, loss_values_dict = self.model.get_loss_dict(outputs, batch, self.cfg.loss_weight_dict)
                    self.optimize_parameters(loss_dict, batch_idx == len(self.train_dataloader) - 1)
                    
                    # Calculate training metrics
                    train_loss += sum(loss_values_dict.values()).item()
                    if "pred" in outputs and "gt_label" in batch:
                        predictions = torch.argmax(outputs["pred"], dim=1)
                        correct_predictions += (predictions == batch["gt_label"]).sum().item()
                        total_samples += batch["gt_label"].size(0)
                
                # Update learning rate
                self.update_learning_rate()
                
                # Log training metrics
                avg_train_loss = train_loss / len(self.train_dataloader)
                train_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
                self.write_log_train(epoch, {"loss": avg_train_loss, "accuracy": train_accuracy})

                # Validation phase
                with torch.no_grad():
                    self.model.net.eval()
                    val_loss = 0.0
                    correct_predictions_val = 0
                    total_samples_val = 0
                    
                    for batch_idx, batch in enumerate(tqdm(self.dataloader)):
                        batch = self.model.feed_to_device(batch)
                        outputs = self.model.get_outputs_dict(batch)
                        loss_dict, loss_values_dict = self.model.get_loss_dict(outputs, batch, self.cfg.loss_weight_dict)
                        
                        # Calculate validation metrics
                        val_loss += sum(loss_values_dict.values()).item()
                        if "pred" in outputs and "gt_label" in batch:
                            predictions = torch.argmax(outputs["pred"], dim=1)
                            correct_predictions_val += (predictions == batch["gt_label"]).sum().item()
                            total_samples_val += batch["gt_label"].size(0)
                    
                    # Log validation metrics
                    avg_val_loss = val_loss / len(self.dataloader)
                    val_accuracy = correct_predictions_val / total_samples_val if total_samples_val > 0 else 0
                    self.write_log_validation(epoch, {"loss": avg_val_loss, "accuracy": val_accuracy})

                    # Save visual results, checkpoints, and evaluate metrics periodically
                    if epoch % self.cfg.steps_per_save_imgs == 0 or epoch == self.cfg.epochs - 1:
                        self.save_visual(epoch)
                    if epoch % self.cfg.steps_per_save_ckpt == 0 or epoch == self.cfg.epochs - 1:
                        self.save_network(epoch)
                    if epoch % self.cfg.steps_per_cal_metrics == 0 or epoch == self.cfg.epochs - 1:
                        metrics_dict = self.evaluate()
                        self.write_log_test(epoch, metrics_dict)
            
            # Update state back to ready
            if self.state_manager:
                self.state_manager.transition_to_state(PipelineState.READY)
        except Exception as e:
            # Update state to error
            if self.state_manager:
                self.state_manager.transition_to_state(PipelineState.ERROR)
            self.logger.error(f"Recognition training error: {e}")
            raise

    def evaluate(self):
        """Evaluate the model on test dataset."""
        self.logger.info("Evaluating model...")
        self.model.net.eval()
        
        # Initialize metrics
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader)):
                batch = self.model.feed_to_device(batch)
                outputs = self.model.get_outputs_dict(batch)
                
                # Collect predictions and labels
                if "pred" in outputs and "gt_label" in batch:
                    predictions = torch.argmax(outputs["pred"], dim=1)
                    correct_predictions += (predictions == batch["gt_label"]).sum().item()
                    total_samples += batch["gt_label"].size(0)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch["gt_label"].cpu().numpy())
        
        # Calculate metrics
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        metrics_dict = {
            "accuracy": accuracy,
            "correct": correct_predictions,
            "total": total_samples
        }
        
        self.logger.info(f"Evaluation Results - Accuracy: {accuracy:.4f}")
        return metrics_dict

    def predict(self, spike_data):
        """Make predictions on spike data."""
        # Input validation
        if spike_data is None:
            raise ValueError("Input spike_data cannot be None")
        if len(spike_data) == 0:
            raise ValueError("Input spike_data cannot be empty")
            
        self.model.net.eval()
        with torch.no_grad():
            spike_data = spike_data.to(self.device)
            outputs = self.model.get_outputs_dict({"spike": spike_data})
            predictions = torch.argmax(outputs["pred"], dim=1)
            return predictions

    def save_visual(self, epoch):
        """Save visual results for recognition task."""
        self.logger.info("Saving visual results...")
        save_folder = self.save_folder / Path("imgs") / Path(f"{epoch:06d}")
        os.makedirs(save_folder, exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader)):
                if batch_idx % (len(self.dataloader) // 3) != 0:
                    continue
                batch = self.model.feed_to_device(batch)
                outputs = self.model.get_outputs_dict(batch)
                
                # Save predictions
                if "pred" in outputs:
                    predictions = torch.argmax(outputs["pred"], dim=1)
                    for i in range(min(5, predictions.size(0))):  # Save first 5 predictions
                        pred_label = predictions[i].item()
                        true_label = batch["gt_label"][i].item() if "gt_label" in batch else "unknown"
                        result_text = f"Pred: {pred_label}, True: {true_label}"
                        
                        # Save text result
                        with open(save_folder / f"{batch_idx:06d}_sample_{i}.txt", "w") as f:
                            f.write(result_text)
                
                # Save visualization if available
                visual_dict = self.model.get_visual_dict(batch, outputs)
                self._state_reset(self.model)
                
                # Save images
                for key, img in visual_dict.items():
                    img = self._post_process_img(img, model_name=self.model.cfg.model_name)
                    cv2.imwrite(str(save_folder / Path(f"{batch_idx:06d}_{key}.png")), tensor2npy(img))
                    if self.cfg.use_tensorboard == True and batch_idx == 0:
                        self.writer.add_image(f"imgs/{key}", img[0].detach().cpu(), epoch)

    def optimize_parameters(self, loss_dict, final_flag):
        """Optimize the parameters from the loss_dict."""
        loss = functools.reduce(torch.add, loss_dict.values())
        step_grad = self.cfg.steps_grad_accumulation
        
        # For SNN methods
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
            # For CNN methods
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_learning_rate(self):
        """Update the learning rate."""
        if self.scheduler is not None:
            self.scheduler.step()

    def write_log_train(self, epoch, metrics_dict):
        """Write the train log information."""
        self.logger.info(f"EPOCH {epoch}/{self.cfg.epochs}: Train Metrics: {metrics_dict}")
        if self.cfg.use_tensorboard:
            for name, val in metrics_dict.items():
                self.writer.add_scalar(f"Train/{name}", val, epoch)
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar(f"Train/lr", lr, epoch)

    def write_log_validation(self, epoch, metrics_dict):
        """Write the validation log information."""
        self.logger.info(f"EPOCH {epoch}/{self.cfg.epochs}: Validation Metrics: {metrics_dict}")
        if self.cfg.use_tensorboard:
            for name, val in metrics_dict.items():
                self.writer.add_scalar(f"Validation/{name}", val, epoch)

    def write_log_test(self, epoch, metrics_dict):
        """Write the test log information."""
        self.logger.info(f"EPOCH {epoch}/{self.cfg.epochs}: Test Metrics: {metrics_dict}")
        if self.cfg.use_tensorboard:
            for name, val in metrics_dict.items():
                self.writer.add_scalar(f"Test/{name}", val, epoch)