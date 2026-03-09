from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@dataclass
class AcceleratorConfig:
    """Accelerator configuration."""
    # Mixed precision
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Device placement
    device_placement: bool = True
    
    # Split batches
    split_batches: bool = False
    
    # Dispatch batches
    dispatch_batches: Optional[bool] = None
    
    # Even batches
    even_batches: bool = True
    
    # Dataloader prefetch factor
    dataloader_prefetch_factor: Optional[int] = None
    
    # Other kwargs for DistributedDataParallel
    ddp_kwargs: Dict[str, Any] = field(default_factory=dict)


class AcceleratorManager:
    """Accelerator manager for multi-GPU training."""
    
    def __init__(self, config: Optional[AcceleratorConfig] = None):
        """
        Initialize accelerator manager.
        
        Args:
            config: Accelerator configuration
        """
        self.config = config or AcceleratorConfig()
        self.accelerator = None
        self.is_initialized = False
    
    def initialize(
        self, 
        model: nn.Module, 
        optimizer: optim.Optimizer, 
        train_dataloader: DataLoader, 
        scheduler: Optional[Any] = None,
        eval_dataloader: Optional[DataLoader] = None
    ):
        """
        Initialize accelerator with model, optimizer, and dataloaders.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            train_dataloader: Training dataloader
            scheduler: Learning rate scheduler (optional)
            eval_dataloader: Evaluation dataloader (optional)
            
        Returns:
            Prepared model, optimizer, train_dataloader, scheduler, eval_dataloader
        """
        # Setup DistributedDataParallel kwargs
        ddp_kwargs = DistributedDataParallelKwargs(**self.config.ddp_kwargs)
        
        # Create accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            device_placement=self.config.device_placement,
            split_batches=self.config.split_batches,
            dispatch_batches=self.config.dispatch_batches,
            even_batches=self.config.even_batches,
            dataloader_prefetch_factor=self.config.dataloader_prefetch_factor,
            kwargs_handlers=[ddp_kwargs]
        )
        
        # Prepare objects
        if scheduler is not None and eval_dataloader is not None:
            model, optimizer, train_dataloader, scheduler, eval_dataloader = self.accelerator.prepare(
                model, optimizer, train_dataloader, scheduler, eval_dataloader
            )
        elif scheduler is not None:
            model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
                model, optimizer, train_dataloader, scheduler
            )
        elif eval_dataloader is not None:
            model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader
            )
        else:
            model, optimizer, train_dataloader = self.accelerator.prepare(
                model, optimizer, train_dataloader
            )
        
        self.is_initialized = True
        
        # Return prepared objects
        if scheduler is not None and eval_dataloader is not None:
            return model, optimizer, train_dataloader, scheduler, eval_dataloader
        elif scheduler is not None:
            return model, optimizer, train_dataloader, scheduler
        elif eval_dataloader is not None:
            return model, optimizer, train_dataloader, eval_dataloader
        else:
            return model, optimizer, train_dataloader
    
    def backward(self, loss: torch.Tensor):
        """
        Perform backward pass with accelerator.
        
        Args:
            loss: Loss tensor
        """
        if not self.is_initialized:
            raise RuntimeError("Accelerator not initialized. Call initialize() first.")
        
        self.accelerator.backward(loss)
    
    def step(self, optimizer: optim.Optimizer):
        """
        Perform optimizer step with accelerator.
        
        Args:
            optimizer: Optimizer
        """
        if not self.is_initialized:
            raise RuntimeError("Accelerator not initialized. Call initialize() first.")
        
        optimizer.step()
    
    def zero_grad(self, optimizer: optim.Optimizer):
        """
        Perform optimizer zero grad with accelerator.
        
        Args:
            optimizer: Optimizer
        """
        if not self.is_initialized:
            raise RuntimeError("Accelerator not initialized. Call initialize() first.")
        
        optimizer.zero_grad()
    
    def gather(self, tensor):
        """
        Gather tensor across all processes.
        
        Args:
            tensor: Tensor to gather
            
        Returns:
            Gathered tensor
        """
        if not self.is_initialized:
            raise RuntimeError("Accelerator not initialized. Call initialize() first.")
        
        return self.accelerator.gather(tensor)
    
    def save_state(self, output_dir: str):
        """
        Save accelerator state.
        
        Args:
            output_dir: Directory to save state
        """
        if not self.is_initialized:
            raise RuntimeError("Accelerator not initialized. Call initialize() first.")
        
        self.accelerator.save_state(output_dir)
    
    def load_state(self, input_dir: str):
        """
        Load accelerator state.
        
        Args:
            input_dir: Directory to load state from
        """
        if not self.is_initialized:
            raise RuntimeError("Accelerator not initialized. Call initialize() first.")
        
        self.accelerator.load_state(input_dir)
    
    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Unwrap model from accelerator.
        
        Args:
            model: Wrapped model
            
        Returns:
            Unwrapped model
        """
        if not self.is_initialized:
            raise RuntimeError("Accelerator not initialized. Call initialize() first.")
        
        return self.accelerator.unwrap_model(model)
    
    def print(self, *args, **kwargs):
        """
        Print only on main process.
        
        Args:
            *args: Arguments to print
            **kwargs: Keyword arguments
        """
        if not self.is_initialized:
            raise RuntimeError("Accelerator not initialized. Call initialize() first.")
        
        self.accelerator.print(*args, **kwargs)
    
    def wait_for_everyone(self):
        """
        Wait for all processes to reach this point.
        """
        if not self.is_initialized:
            raise RuntimeError("Accelerator not initialized. Call initialize() first.")
        
        self.accelerator.wait_for_everyone()
    
    def is_main_process(self) -> bool:
        """
        Check if current process is main process.
        
        Returns:
            True if main process
        """
        if not self.is_initialized:
            raise RuntimeError("Accelerator not initialized. Call initialize() first.")
        
        return self.accelerator.is_main_process


def create_accelerator_manager(config: Optional[Dict[str, Any]] = None) -> AcceleratorManager:
    """
    Create accelerator manager with optional configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AcceleratorManager instance
    """
    if config is not None:
        acc_config = AcceleratorConfig(**config)
    else:
        acc_config = AcceleratorConfig()
    
    return AcceleratorManager(acc_config)