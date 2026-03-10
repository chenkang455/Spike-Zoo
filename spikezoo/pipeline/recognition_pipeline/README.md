# Recognition Pipeline

This pipeline implements recognition tasks for spike-based data, including classification, detection, and tracking.

## Features

- Support for classification tasks with configurable number of classes
- Training and evaluation pipelines
- Visualization of results
- TensorBoard integration for monitoring training progress
- Support for both SNN and CNN models

## Usage

```python
from spikezoo.pipeline.recognition_pipeline import RecognitionPipeline, RecognitionPipelineConfig
from spikezoo.models import BaseModelConfig
from spikezoo.datasets import BaseDatasetConfig

# Configure the pipeline
cfg = RecognitionPipelineConfig(
    epochs=50,
    num_classes=10,
    save_folder="results/recognition"
)

# Initialize the pipeline
pipeline = RecognitionPipeline(
    cfg=cfg,
    model_cfg=BaseModelConfig(),  # Replace with your model config
    dataset_cfg=BaseDatasetConfig()  # Replace with your dataset config
)

# Train the model
pipeline.train()

# Evaluate the model
metrics = pipeline.evaluate()
```

## Configuration Options

- `task_type`: Type of recognition task (classification, detection, tracking)
- `num_classes`: Number of classes for classification tasks
- `save_predictions`: Whether to save prediction results
- `eval_metrics`: Evaluation metrics to compute
- `confidence_threshold`: Confidence threshold for detection tasks
- `iou_threshold`: IoU threshold for detection tasks