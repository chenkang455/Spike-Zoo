# Contributing to SpikeZoo

Thank you for your interest in contributing to SpikeZoo! This document provides guidelines and procedures for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Process](#development-process)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Requests](#pull-requests)
9. [Issue Reporting](#issue-reporting)
10. [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all contributors and users with respect and kindness.

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Familiarity with PyTorch and deep learning concepts

### Setting Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/your-username/spikezoo.git
cd spikezoo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### Understanding the Codebase

Before contributing, familiarize yourself with:

- [Project Structure](model_structure.md)
- [Development Guide](development_guide.md)
- [API Documentation](../api/)
- Existing models and pipelines

## How to Contribute

### Ways to Contribute

1. **Bug Reports**: Report issues you encounter
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Implement features or fix bugs
4. **Documentation**: Improve documentation and examples
5. **Testing**: Add test cases and improve coverage
6. **Reviews**: Review pull requests from other contributors

### Finding Issues

- Check [Good First Issues](https://github.com/your-org/spikezoo/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- Look for [Help Wanted](https://github.com/your-org/spikezoo/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) issues
- Browse all [open issues](https://github.com/your-org/spikezoo/issues)

## Development Process

### Branching Strategy

We follow the GitFlow workflow:

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
git add .
git commit -m "feat: add your feature"

# Push to your fork
git push origin feature/your-feature-name
```

### Task System

For internal contributors using the task system:

1. Receive task assignment
2. Acknowledge immediately
3. Follow development state machine
4. Create pull request when ready

## Coding Standards

### Style Guide

Follow PEP 8 with these additions:

- Use 4 spaces for indentation
- Line length: 88 characters (Black default)
- Use meaningful variable names
- Add type hints to all function signatures

### Code Quality

```python
# Good example
def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy between predictions and targets.
    
    Args:
        predictions: Model predictions tensor
        targets: Ground truth targets tensor
        
    Returns:
        Accuracy as float between 0 and 1
    """
    correct = (predictions.argmax(dim=1) == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0
```

### Imports

Organize imports in this order:

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Correct import organization
import os
import sys
from typing import Dict, List, Optional

import torch
import numpy as np

from spikezoo.models import BaseModel
from spikezoo.utils import load_network
```

## Testing

### Test Structure

Follow the project's test structure:

```
tests/
├── test_models/
│   ├── test_base_model.py
│   └── test_custom_model.py
├── test_pipelines/
│   └── test_base_pipeline.py
└── conftest.py
```

### Writing Tests

Use pytest for new tests:

```python
import pytest
import torch
from spikezoo.models import BaseModel, BaseModelConfig

class TestBaseModel:
    """Test suite for BaseModel."""
    
    @pytest.fixture
    def model_config(self):
        """Fixture for model configuration."""
        return BaseModelConfig(model_name="test_model")
    
    def test_model_initialization(self, model_config):
        """Test model initialization."""
        model = BaseModel(model_config)
        assert model.cfg == model_config
        assert model.net is None
    
    def test_model_creation(self, model_config):
        """Test model creation."""
        model = BaseModel(model_config)
        assert isinstance(model, BaseModel)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models/test_base_model.py

# Run with coverage
pytest --cov=spikezoo tests/

# Run specific test
pytest tests/test_models/test_base_model.py::TestBaseModel::test_model_initialization
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """Example function with docstring.
    
    Args:
        param1: The first parameter.
        param2: The second parameter. Defaults to 10.
        
    Returns:
        True if successful, False otherwise.
        
    Raises:
        ValueError: If param1 is invalid.
        
    Example:
        >>> example_function("test", 5)
        True
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    return True
```

### README Updates

Update relevant README files when:

- Adding new features
- Changing APIs
- Updating installation instructions
- Modifying usage examples

### API Documentation

For significant API changes:

1. Update docstrings
2. Add usage examples
3. Update API reference documentation

## Pull Requests

### Before Submitting

1. Ensure all tests pass
2. Add new tests for new functionality
3. Update documentation
4. Follow coding standards
5. Squash related commits

### Pull Request Template

```markdown
## Description

Brief description of changes.

## Related Issue

Fixes #123

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist

- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have added tests
- [ ] I have updated documentation
- [ ] All tests pass
```

### Review Process

1. Automated checks run on submission
2. Maintainers review code quality
3. Tests and documentation verified
4. Feedback provided within 48 hours
5. PR merged after approval

## Issue Reporting

### Good Bug Reports

Include:

- Clear title
- Steps to reproduce
- Expected vs actual behavior
- Environment information
- Minimal reproducible example
- Relevant logs or screenshots

### Feature Requests

Include:

- Problem statement
- Proposed solution
- Alternatives considered
- Benefits and drawbacks

### Security Issues

Report security vulnerabilities privately to the maintainers.

## Community

### Communication Channels

- GitHub Issues: For bugs and feature requests
- Discussions: For general questions and community interaction
- Email: For private communications

### Recognition

Contributors are recognized in:

- Release notes
- Contributor list
- Social media announcements
- Conference presentations

### Mentorship

We offer mentorship for new contributors:

- Pair programming sessions
- Code review guidance
- Architecture walkthroughs
- Career development advice

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to ask questions in:

- GitHub Discussions
- Issue comments
- Community meetings
- Direct messages to maintainers

Thank you for contributing to SpikeZoo!