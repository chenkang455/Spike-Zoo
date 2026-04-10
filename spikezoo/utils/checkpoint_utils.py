import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging


class CheckpointManager:
    """Checkpoint manager for saving and loading training states."""
    
    def __init__(self, save_dir: Union[str, Path]):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        best_metric: Optional[float] = None,
        additional_state: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Save training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Scheduler to save (optional)
            epoch: Current epoch
            step: Current step
            best_metric: Best metric value (optional)
            additional_state: Additional state to save (optional)
            filename: Checkpoint filename (optional)
            
        Returns:
            Path to saved checkpoint
        """
        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric
        }
        
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_state is not None:
            state.update(additional_state)
        
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:06d}_step_{step:06d}.pth"
        
        checkpoint_path = self.save_dir / filename
        torch.save(state, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into (optional)
            checkpoint_path: Path to checkpoint file (optional)
            filename: Checkpoint filename (optional, used if checkpoint_path not provided)
            
        Returns:
            Checkpoint state dictionary
        """
        if checkpoint_path is None:
            if filename is None:
                raise ValueError("Either checkpoint_path or filename must be provided")
            checkpoint_path = self.save_dir / filename
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint
    
    def save_best_model(
        self,
        model: torch.nn.Module,
        metric_value: float,
        filename: str = "best_model.pth"
    ) -> str:
        """
        Save best model based on metric value.
        
        Args:
            model: Model to save
            metric_value: Metric value
            filename: Filename for best model
            
        Returns:
            Path to saved best model
        """
        save_path = self.save_dir / filename
        torch.save({
            'model_state_dict': model.state_dict(),
            'metric_value': metric_value
        }, save_path)
        self.logger.info(f"Best model saved to {save_path}")
        
        return str(save_path)
    
    def load_best_model(
        self,
        model: torch.nn.Module,
        filename: str = "best_model.pth"
    ) -> Dict[str, Any]:
        """
        Load best model.
        
        Args:
            model: Model to load state into
            filename: Filename for best model
            
        Returns:
            Best model state dictionary
        """
        load_path = self.save_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Best model not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Best model loaded from {load_path}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the latest checkpoint file.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoint_files = list(self.save_dir.glob("checkpoint_epoch_*.pth"))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time to get the latest
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoint_files[0])
    
    def get_checkpoints_sorted(self) -> list:
        """
        Get all checkpoint files sorted by epoch and step.
        
        Returns:
            List of checkpoint paths sorted by epoch and step
        """
        checkpoint_files = list(self.save_dir.glob("checkpoint_epoch_*.pth"))
        
        # Sort by epoch and step extracted from filename
        def extract_epoch_step(filename):
            # Extract epoch and step from filename like "checkpoint_epoch_000010_step_000050.pth"
            parts = filename.stem.split('_')
            try:
                epoch_idx = parts.index('epoch')
                step_idx = parts.index('step')
                epoch = int(parts[epoch_idx + 1])
                step = int(parts[step_idx + 1])
                return (epoch, step)
            except (ValueError, IndexError):
                # If parsing fails, use modification time
                return (0, 0)
        
        checkpoint_files.sort(key=lambda x: extract_epoch_step(x))
        return [str(f) for f in checkpoint_files]


def create_checkpoint_manager(save_dir: Union[str, Path]) -> CheckpointManager:
    """
    Create checkpoint manager.
    
    Args:
        save_dir: Directory to save checkpoints
        
    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(save_dir)