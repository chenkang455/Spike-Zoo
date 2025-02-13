# code borrow from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/engine/optimizers.py
from dataclasses import dataclass
import torch
from typing import Any, Dict, List, Optional, Type,Tuple

@dataclass
class SchedulerConfig:
    def setup(self, optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        return self._target(optimizer, **kwargs)

@dataclass
class CosineAnnealingLRConfig(SchedulerConfig):
    T_max: int
    eta_min: float = 0
    _target: Type = torch.optim.lr_scheduler.CosineAnnealingLR


@dataclass
class MultiStepSchedulerConfig(SchedulerConfig):
    milestones: List[int]
    gamma: float = 0.1
    _target: Type = torch.optim.lr_scheduler.MultiStepLR
