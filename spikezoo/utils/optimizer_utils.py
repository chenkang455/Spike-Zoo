# code borrow from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/engine/optimizers.py
from dataclasses import dataclass
import torch
from typing import Any, Dict, List, Optional, Type,Tuple

@dataclass
class OptimizerConfig:
    def setup(self, model_params) -> torch.optim.Optimizer:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        return self._target(model_params, **kwargs)

@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    _target: Type = torch.optim.Adam

