from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import logging
import os
import json
import numpy as np
from datetime import datetime


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    # General settings
    enabled: bool = True
    experiment_name: str = "experiment"
    log_dir: str = "logs"
    save_frequency: int = 100  # Save every N steps
    
    # TensorBoard settings
    tensorboard_enabled: bool = True
    tensorboard_flush_secs: int = 120
    
    # WandB settings
    wandb_enabled: bool = False
    wandb_project: str = "spikezoo"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Plot settings
    plot_enabled: bool = True
    plot_format: str = "png"  # png, jpg, pdf
    plot_dpi: int = 100
    plot_save_frequency: int = 10  # Save plots every N epochs
    
    # Metrics settings
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "loss", "accuracy", "precision", "recall", "f1_score"
    ])
    images_to_track: List[str] = field(default_factory=lambda: [
        "input_images", "output_images", "ground_truth"
    ])


class BaseVisualizer:
    """Base class for visualization backends."""
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_initialized = False
    
    def initialize(self):
        """Initialize visualization backend."""
        raise NotImplementedError
    
    def log_scalar(self, name: str, value: float, step: int):
        """
        Log scalar value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
        """
        raise NotImplementedError
    
    def log_image(self, name: str, image: np.ndarray, step: int):
        """
        Log image.
        
        Args:
            name: Image name
            image: Image array (H, W) or (H, W, C)
            step: Training step
        """
        raise NotImplementedError
    
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """
        Log histogram.
        
        Args:
            name: Histogram name
            values: Values array
            step: Training step
        """
        raise NotImplementedError
    
    def log_text(self, name: str, text: str, step: int):
        """
        Log text.
        
        Args:
            name: Text name
            text: Text content
            step: Training step
        """
        raise NotImplementedError
    
    def log_config(self, config_dict: Dict[str, Any]):
        """
        Log configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        raise NotImplementedError
    
    def flush(self):
        """Flush logs."""
        pass
    
    def close(self):
        """Close visualization backend."""
        pass


class TensorBoardVisualizer(BaseVisualizer):
    """TensorBoard visualization backend."""
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize TensorBoard visualizer.
        
        Args:
            config: Visualization configuration
        """
        super().__init__(config)
        self.writer = None
        self.log_path = None
    
    def initialize(self):
        """Initialize TensorBoard writer."""
        if not self.config.tensorboard_enabled:
            return
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            # Create log directory
            self.log_path = Path(self.config.log_dir) / "tensorboard" / self.config.experiment_name
            self.log_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize writer
            self.writer = SummaryWriter(
                log_dir=str(self.log_path),
                flush_secs=self.config.tensorboard_flush_secs
            )
            
            self.is_initialized = True
            self.logger.info(f"TensorBoard initialized at {self.log_path}")
        except ImportError:
            self.logger.warning("TensorBoard not available, visualization disabled")
            self.config.tensorboard_enabled = False
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard: {e}")
            self.config.tensorboard_enabled = False
    
    def log_scalar(self, name: str, value: float, step: int):
        """
        Log scalar value to TensorBoard.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
        """
        if not self.is_initialized or not self.config.tensorboard_enabled:
            return
        
        try:
            self.writer.add_scalar(name, value, step)
        except Exception as e:
            self.logger.warning(f"Failed to log scalar {name} to TensorBoard: {e}")
    
    def log_image(self, name: str, image: np.ndarray, step: int):
        """
        Log image to TensorBoard.
        
        Args:
            name: Image name
            image: Image array (H, W) or (H, W, C)
            step: Training step
        """
        if not self.is_initialized or not self.config.tensorboard_enabled:
            return
        
        try:
            # Ensure image is in correct format (H, W, C) or (C, H, W)
            if len(image.shape) == 2:
                # Grayscale image (H, W) -> (1, H, W)
                image = np.expand_dims(image, axis=0)
            elif len(image.shape) == 3 and image.shape[2] in [1, 3, 4]:
                # (H, W, C) -> (C, H, W)
                image = np.transpose(image, (2, 0, 1))
            
            self.writer.add_image(name, image, step, dataformats='CHW')
        except Exception as e:
            self.logger.warning(f"Failed to log image {name} to TensorBoard: {e}")
    
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """
        Log histogram to TensorBoard.
        
        Args:
            name: Histogram name
            values: Values array
            step: Training step
        """
        if not self.is_initialized or not self.config.tensorboard_enabled:
            return
        
        try:
            self.writer.add_histogram(name, values, step)
        except Exception as e:
            self.logger.warning(f"Failed to log histogram {name} to TensorBoard: {e}")
    
    def log_text(self, name: str, text: str, step: int):
        """
        Log text to TensorBoard.
        
        Args:
            name: Text name
            text: Text content
            step: Training step
        """
        if not self.is_initialized or not self.config.tensorboard_enabled:
            return
        
        try:
            self.writer.add_text(name, text, step)
        except Exception as e:
            self.logger.warning(f"Failed to log text {name} to TensorBoard: {e}")
    
    def log_config(self, config_dict: Dict[str, Any]):
        """
        Log configuration to TensorBoard.
        
        Args:
            config_dict: Configuration dictionary
        """
        if not self.is_initialized or not self.config.tensorboard_enabled:
            return
        
        try:
            # Log config as text
            config_str = json.dumps(config_dict, indent=2, default=str)
            self.writer.add_text("config", f"```\n{config_str}\n```", 0)
        except Exception as e:
            self.logger.warning(f"Failed to log config to TensorBoard: {e}")
    
    def flush(self):
        """Flush TensorBoard logs."""
        if self.is_initialized and self.config.tensorboard_enabled:
            try:
                self.writer.flush()
            except Exception as e:
                self.logger.warning(f"Failed to flush TensorBoard logs: {e}")
    
    def close(self):
        """Close TensorBoard writer."""
        if self.is_initialized and self.config.tensorboard_enabled:
            try:
                self.writer.close()
                self.is_initialized = False
            except Exception as e:
                self.logger.warning(f"Failed to close TensorBoard writer: {e}")


class WandBVisualizer(BaseVisualizer):
    """Weights & Biases visualization backend."""
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize WandB visualizer.
        
        Args:
            config: Visualization configuration
        """
        super().__init__(config)
        self.run = None
    
    def initialize(self):
        """Initialize WandB run."""
        if not self.config.wandb_enabled:
            return
        
        try:
            import wandb
            
            # Initialize run
            self.run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group=self.config.wandb_group,
                job_type=self.config.wandb_job_type,
                tags=self.config.wandb_tags,
                name=self.config.experiment_name,
                config={
                    "experiment_name": self.config.experiment_name,
                    "timestamp": datetime.now().isoformat()
                },
                reinit=True
            )
            
            self.is_initialized = True
            self.logger.info("WandB initialized")
        except ImportError:
            self.logger.warning("WandB not available, visualization disabled")
            self.config.wandb_enabled = False
        except Exception as e:
            self.logger.error(f"Failed to initialize WandB: {e}")
            self.config.wandb_enabled = False
    
    def log_scalar(self, name: str, value: float, step: int):
        """
        Log scalar value to WandB.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
        """
        if not self.is_initialized or not self.config.wandb_enabled:
            return
        
        try:
            import wandb
            wandb.log({name: value}, step=step)
        except Exception as e:
            self.logger.warning(f"Failed to log scalar {name} to WandB: {e}")
    
    def log_image(self, name: str, image: np.ndarray, step: int):
        """
        Log image to WandB.
        
        Args:
            name: Image name
            image: Image array (H, W) or (H, W, C)
            step: Training step
        """
        if not self.is_initialized or not self.config.wandb_enabled:
            return
        
        try:
            import wandb
            
            # Convert to WandB Image
            wb_image = wandb.Image(image, caption=name)
            wandb.log({name: wb_image}, step=step)
        except Exception as e:
            self.logger.warning(f"Failed to log image {name} to WandB: {e}")
    
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """
        Log histogram to WandB.
        
        Args:
            name: Histogram name
            values: Values array
            step: Training step
        """
        if not self.is_initialized or not self.config.wandb_enabled:
            return
        
        try:
            import wandb
            
            # Convert to WandB Histogram
            wb_hist = wandb.Histogram(values)
            wandb.log({name: wb_hist}, step=step)
        except Exception as e:
            self.logger.warning(f"Failed to log histogram {name} to WandB: {e}")
    
    def log_text(self, name: str, text: str, step: int):
        """
        Log text to WandB.
        
        Args:
            name: Text name
            text: Text content
            step: Training step
        """
        if not self.is_initialized or not self.config.wandb_enabled:
            return
        
        try:
            import wandb
            
            # Log as HTML table for better formatting
            html_text = f"<pre>{text}</pre>"
            wandb.log({name: wandb.Html(html_text)}, step=step)
        except Exception as e:
            self.logger.warning(f"Failed to log text {name} to WandB: {e}")
    
    def log_config(self, config_dict: Dict[str, Any]):
        """
        Log configuration to WandB.
        
        Args:
            config_dict: Configuration dictionary
        """
        if not self.is_initialized or not self.config.wandb_enabled:
            return
        
        try:
            import wandb
            wandb.config.update(config_dict)
        except Exception as e:
            self.logger.warning(f"Failed to log config to WandB: {e}")
    
    def close(self):
        """Close WandB run."""
        if self.is_initialized and self.config.wandb_enabled:
            try:
                import wandb
                wandb.finish()
                self.is_initialized = False
            except Exception as e:
                self.logger.warning(f"Failed to close WandB run: {e}")


class MatplotlibVisualizer(BaseVisualizer):
    """Matplotlib visualization backend for static plots."""
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize matplotlib visualizer.
        
        Args:
            config: Visualization configuration
        """
        super().__init__(config)
        self.plot_dir = None
    
    def initialize(self):
        """Initialize matplotlib visualizer."""
        if not self.config.plot_enabled:
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Create plot directory
            self.plot_dir = Path(self.config.log_dir) / "plots" / self.config.experiment_name
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            
            self.is_initialized = True
            self.logger.info(f"Matplotlib initialized, plots will be saved to {self.plot_dir}")
        except ImportError:
            self.logger.warning("Matplotlib not available, plotting disabled")
            self.config.plot_enabled = False
        except Exception as e:
            self.logger.error(f"Failed to initialize matplotlib: {e}")
            self.config.plot_enabled = False
    
    def save_plot(self, name: str, figure, step: int = None):
        """
        Save matplotlib figure.
        
        Args:
            name: Plot name
            figure: Matplotlib figure
            step: Training step (optional)
        """
        if not self.is_initialized or not self.config.plot_enabled:
            return
        
        try:
            # Create filename
            if step is not None:
                filename = f"{name}_step_{step}.{self.config.plot_format}"
            else:
                filename = f"{name}.{self.config.plot_format}"
            
            filepath = self.plot_dir / filename
            
            # Save figure
            figure.savefig(
                filepath,
                dpi=self.config.plot_dpi,
                bbox_inches='tight',
                format=self.config.plot_format
            )
            
            self.logger.debug(f"Saved plot to {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to save plot {name}: {e}")
    
    def close(self):
        """Close matplotlib visualizer."""
        if self.is_initialized and self.config.plot_enabled:
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
                self.is_initialized = False
            except Exception as e:
                self.logger.warning(f"Failed to close matplotlib: {e}")


class VisualizationManager:
    """Manager for multiple visualization backends."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualization manager.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize backends
        self.backends: Dict[str, BaseVisualizer] = {}
        self._initialize_backends()
        
        # Tracking variables
        self.step_counter = 0
        self.last_flush_step = 0
        self.last_plot_step = 0
    
    def _initialize_backends(self):
        """Initialize visualization backends."""
        if self.config.tensorboard_enabled:
            try:
                tb_visualizer = TensorBoardVisualizer(self.config)
                tb_visualizer.initialize()
                if tb_visualizer.is_initialized:
                    self.backends["tensorboard"] = tb_visualizer
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorBoard: {e}")
        
        if self.config.wandb_enabled:
            try:
                wandb_visualizer = WandBVisualizer(self.config)
                wandb_visualizer.initialize()
                if wandb_visualizer.is_initialized:
                    self.backends["wandb"] = wandb_visualizer
            except Exception as e:
                self.logger.warning(f"Failed to initialize WandB: {e}")
        
        if self.config.plot_enabled:
            try:
                plot_visualizer = MatplotlibVisualizer(self.config)
                plot_visualizer.initialize()
                if plot_visualizer.is_initialized:
                    self.backends["matplotlib"] = plot_visualizer
            except Exception as e:
                self.logger.warning(f"Failed to initialize matplotlib: {e}")
        
        self.logger.info(f"Initialized {len(self.backends)} visualization backends")
    
    def log_scalar(self, name: str, value: float, step: int = None):
        """
        Log scalar value to all enabled backends.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step (uses internal counter if None)
        """
        if not self.config.enabled:
            return
        
        if step is None:
            step = self.step_counter
        
        for backend in self.backends.values():
            try:
                backend.log_scalar(name, value, step)
            except Exception as e:
                self.logger.warning(f"Failed to log scalar {name} to {backend.__class__.__name__}: {e}")
        
        # Update step counter
        self.step_counter = max(self.step_counter, step + 1)
        
        # Flush periodically
        if step - self.last_flush_step >= self.config.save_frequency:
            self.flush()
            self.last_flush_step = step
    
    def log_image(self, name: str, image: np.ndarray, step: int = None):
        """
        Log image to all enabled backends.
        
        Args:
            name: Image name
            image: Image array
            step: Training step (uses internal counter if None)
        """
        if not self.config.enabled:
            return
        
        if step is None:
            step = self.step_counter
        
        for backend in self.backends.values():
            try:
                backend.log_image(name, image, step)
            except Exception as e:
                self.logger.warning(f"Failed to log image {name} to {backend.__class__.__name__}: {e}")
        
        # Update step counter
        self.step_counter = max(self.step_counter, step + 1)
    
    def log_histogram(self, name: str, values: np.ndarray, step: int = None):
        """
        Log histogram to all enabled backends.
        
        Args:
            name: Histogram name
            values: Values array
            step: Training step (uses internal counter if None)
        """
        if not self.config.enabled:
            return
        
        if step is None:
            step = self.step_counter
        
        for backend in self.backends.values():
            try:
                backend.log_histogram(name, values, step)
            except Exception as e:
                self.logger.warning(f"Failed to log histogram {name} to {backend.__class__.__name__}: {e}")
        
        # Update step counter
        self.step_counter = max(self.step_counter, step + 1)
    
    def log_text(self, name: str, text: str, step: int = None):
        """
        Log text to all enabled backends.
        
        Args:
            name: Text name
            text: Text content
            step: Training step (uses internal counter if None)
        """
        if not self.config.enabled:
            return
        
        if step is None:
            step = self.step_counter
        
        for backend in self.backends.values():
            try:
                backend.log_text(name, text, step)
            except Exception as e:
                self.logger.warning(f"Failed to log text {name} to {backend.__class__.__name__}: {e}")
        
        # Update step counter
        self.step_counter = max(self.step_counter, step + 1)
    
    def log_config(self, config_dict: Dict[str, Any]):
        """
        Log configuration to all enabled backends.
        
        Args:
            config_dict: Configuration dictionary
        """
        if not self.config.enabled:
            return
        
        for backend in self.backends.values():
            try:
                backend.log_config(config_dict)
            except Exception as e:
                self.logger.warning(f"Failed to log config to {backend.__class__.__name__}: {e}")
    
    def log_plots(self, name: str, step: int = None):
        """
        Log plots to file-based backends.
        
        Args:
            name: Plot name
            step: Training step (uses internal counter if None)
        """
        if not self.config.enabled or not self.config.plot_enabled:
            return
        
        if step is None:
            step = self.step_counter
        
        # Save plots periodically
        if step - self.last_plot_step >= self.config.plot_save_frequency:
            for backend in self.backends.values():
                if isinstance(backend, MatplotlibVisualizer):
                    # This is a placeholder - actual plotting would happen in training code
                    pass
            
            self.last_plot_step = step
    
    def flush(self):
        """Flush all backends."""
        if not self.config.enabled:
            return
        
        for backend in self.backends.values():
            try:
                backend.flush()
            except Exception as e:
                self.logger.warning(f"Failed to flush {backend.__class__.__name__}: {e}")
    
    def close(self):
        """Close all backends."""
        for backend in self.backends.values():
            try:
                backend.close()
            except Exception as e:
                self.logger.warning(f"Failed to close {backend.__class__.__name__}: {e}")
        
        self.backends.clear()


# Global visualization manager instance
_visualization_manager: Optional[VisualizationManager] = None


def get_visualization_manager(config: Optional[VisualizationConfig] = None) -> VisualizationManager:
    """
    Get or create global visualization manager.
    
    Args:
        config: Visualization configuration (only used for new managers)
        
    Returns:
        VisualizationManager instance
    """
    global _visualization_manager
    if _visualization_manager is None:
        _visualization_manager = VisualizationManager(config)
    return _visualization_manager


def setup_visualization(config: Optional[VisualizationConfig] = None):
    """
    Setup global visualization manager.
    
    Args:
        config: Visualization configuration
    """
    global _visualization_manager
    if _visualization_manager is not None:
        _visualization_manager.close()
    
    _visualization_manager = VisualizationManager(config)


def log_scalar(name: str, value: float, step: int = None):
    """Log scalar value."""
    manager = get_visualization_manager()
    manager.log_scalar(name, value, step)


def log_image(name: str, image: np.ndarray, step: int = None):
    """Log image."""
    manager = get_visualization_manager()
    manager.log_image(name, image, step)


def log_histogram(name: str, values: np.ndarray, step: int = None):
    """Log histogram."""
    manager = get_visualization_manager()
    manager.log_histogram(name, values, step)


def log_text(name: str, text: str, step: int = None):
    """Log text."""
    manager = get_visualization_manager()
    manager.log_text(name, text, step)


def log_config(config_dict: Dict[str, Any]):
    """Log configuration."""
    manager = get_visualization_manager()
    manager.log_config(config_dict)


def flush_visualization():
    """Flush visualization logs."""
    manager = get_visualization_manager()
    manager.flush()


def close_visualization():
    """Close visualization manager."""
    global _visualization_manager
    if _visualization_manager is not None:
        _visualization_manager.close()
        _visualization_manager = None