# SpikeZoo Model Directory Structure Standard

This document defines the standard directory structure for models in SpikeZoo, following the reference structure from SpikeCV.

## Overview

The model directory structure in SpikeZoo follows a modular, hierarchical approach that separates concerns and promotes reusability. Each model has its own dedicated directory with a consistent internal structure.

## Directory Structure

```
spikezoo/
├── archs/
│   ├── base/
│   │   ├── __init__.py
│   │   ├── nets.py
│   │   └── ...
│   ├── model_name/
│   │   ├── __init__.py
│   │   ├── nets.py
│   │   ├── layers.py
│   │   ├── blocks.py
│   │   ├── modules.py
│   │   └── utils.py
│   └── __init__.py
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── model_registry.py
│   └── modules/
│       ├── __init__.py
│       ├── network_loader.py
│       ├── loss_functions.py
│       └── metric_utils.py
└── ...
```

## Detailed Structure

### 1. Architecture Directory (`archs/`)

The `archs/` directory contains the actual neural network implementations.

#### Base Architecture (`archs/base/`)
- Contains fundamental building blocks and base classes
- Shared across all models
- Includes common layers, utilities, and base network definitions

#### Model-Specific Architecture (`archs/model_name/`)
Each model gets its own subdirectory under `archs/`:

- `__init__.py`: Package initializer
- `nets.py`: Main network definitions and architectures
- `layers.py`: Custom layer implementations
- `blocks.py`: Building block definitions (e.g., residual blocks, attention blocks)
- `modules.py`: Reusable network modules
- `utils.py`: Model-specific utility functions

### 2. Model Interface Directory (`models/`)

The `models/` directory contains the high-level model interfaces and management utilities.

#### Core Files
- `base_model.py`: Base model class defining the standard interface
- `model_registry.py`: Model registration and discovery system
- `__init__.py`: Package exports

#### Modules Subdirectory (`models/modules/`)
Contains modular utilities for model operations:

- `network_loader.py`: Network loading and saving utilities
- `loss_functions.py`: Loss function definitions and registry
- `metric_utils.py`: Metric calculation and visualization utilities

## Naming Conventions

### Directory Names
- Use lowercase with underscores: `model_name` (not `ModelName` or `modelName`)
- Be descriptive but concise
- Match the model's registered name when applicable

### File Names
- Use lowercase with underscores: `network_loader.py`
- Main network file should be `nets.py`
- Utility files should describe their purpose: `layers.py`, `blocks.py`

### Class Names
- Use PascalCase: `BaseModel`, `ResidualBlock`
- Prefix with model name when specific: `UNetEncoder`, `TransformerAttention`

### Function Names
- Use snake_case: `load_network_weights`, `compute_loss`
- Be descriptive: `get_paired_images`, `prepare_visualization_dict`

## Implementation Guidelines

### 1. Model Architecture Files (`archs/model_name/nets.py`)

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple

class ModelNameNet(nn.Module):
    """Main network class for ModelName."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize the network.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        # Implementation here
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Implementation here
        return x
```

### 2. Model Interface Files (`models/model_name_model.py`)

```python
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from spikezoo.models.model_registry import register_model

class ModelNameModel(BaseModel):
    """Model interface for ModelName."""
    
    def __init__(self, cfg: BaseModelConfig):
        """Initialize model interface.
        
        Args:
            cfg: Model configuration
        """
        super().__init__(cfg)
    
    def build_network(self, mode: str = "train", version: str = "local"):
        """Build the network.
        
        Args:
            mode: Model mode ("train" or "eval")
            version: Model version
            
        Returns:
            Self for chaining
        """
        # Implementation here
        return self

# Register the model
register_model("model_name", ModelNameModel, ModelNameModelConfig)
```

### 3. Configuration Classes

```python
from dataclasses import dataclass, field
from spikezoo.models.base_model import BaseModelConfig

@dataclass
class ModelNameModelConfig(BaseModelConfig):
    """Configuration for ModelName model."""
    
    # Model-specific parameters
    hidden_size: int = 256
    num_layers: int = 6
    
    # Override inherited parameters if needed
    model_file_name: str = "nets"
    model_cls_name: str = "ModelNameNet"
```

## Best Practices

### 1. Modularity
- Keep files focused on single responsibilities
- Break large networks into logical components
- Reuse common building blocks from `archs/base/`

### 2. Documentation
- Document all public classes and methods
- Include usage examples in docstrings
- Maintain this structural documentation

### 3. Testing
- Each model should have corresponding tests
- Test both the architecture and model interface
- Include integration tests for complex models

### 4. Dependencies
- Minimize external dependencies
- Clearly document required packages
- Handle optional dependencies gracefully

## Migration Guide

For existing models:

1. Create new directory under `archs/model_name/`
2. Move network implementation to `archs/model_name/nets.py`
3. Create model interface in `models/model_name_model.py`
4. Update imports and references
5. Register the model with the registry
6. Update documentation and tests

## Examples

See existing models in the repository for concrete examples of this structure in practice.

## Version Control

- Each model change should be in its own commit when possible
- Major structural changes should be documented
- Follow semantic versioning for model releases