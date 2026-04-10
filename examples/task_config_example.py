#!/usr/bin/env python3
"""
Example of using the SpikeZoo multi-task configuration system.
"""

import sys
import os

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.config import (
    MultiTaskConfig, 
    TaskConfig, 
    load_task_config, 
    save_task_config
)


def example_programmatic_config():
    """Example of creating task configuration programmatically."""
    print("=== SpikeZoo Multi-Task Configuration Example ===\n")
    
    # Create multi-task configuration
    multitask_config = MultiTaskConfig(
        project_name="SpikeZoo Example Project",
        project_version="1.0.0",
        default_task_settings={
            "output_base_dir": "./results",
            "log_level": "INFO",
            "timeout": 3600
        },
        global_parameters={
            "data_root": "./data",
            "model_zoo": "./models",
            "num_workers": 4
        }
    )
    
    # Create individual task configurations
    preprocess_task = TaskConfig(
        task_id="task_001",
        task_name="Data Preprocessing",
        task_description="Preprocess spike data for training",
        enabled=True,
        priority=1,
        parameters={
            "input_dir": "${data_root}/raw",
            "output_dir": "${data_root}/processed",
            "normalize": True,
            "resize": [256, 256]
        },
        dependencies=[],
        output_dir="${output_base_dir}/preprocessing"
    )
    
    train_task = TaskConfig(
        task_id="task_002",
        task_name="Model Training",
        task_description="Train spike-to-image model",
        enabled=True,
        priority=2,
        parameters={
            "model_type": "cnn",
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam"
        },
        dependencies=["task_001"],
        output_dir="${output_base_dir}/training"
    )
    
    eval_task = TaskConfig(
        task_id="task_003",
        task_name="Model Evaluation",
        task_description="Evaluate trained model performance",
        enabled=True,
        priority=3,
        parameters={
            "metrics": ["psnr", "ssim", "lpips"],
            "test_batch_size": 16
        },
        dependencies=["task_002"],
        output_dir="${output_base_dir}/evaluation"
    )
    
    # Add tasks to configuration
    multitask_config.add_task(preprocess_task)
    multitask_config.add_task(train_task)
    multitask_config.add_task(eval_task)
    
    # Display configuration
    print("1. Created multi-task configuration:")
    print(f"   Project: {multitask_config.project_name} v{multitask_config.project_version}")
    print(f"   Global parameters: {multitask_config.global_parameters}")
    print(f"   Default settings: {multitask_config.default_task_settings}")
    print()
    
    # Display tasks
    print("2. Task configurations:")
    for task_id, task in multitask_config.tasks.items():
        print(f"   Task {task_id}: {task.task_name}")
        print(f"     Description: {task.task_description}")
        print(f"     Enabled: {task.enabled}, Priority: {task.priority}")
        print(f"     Parameters: {task.parameters}")
        print(f"     Dependencies: {task.dependencies}")
        print(f"     Output: {task.output_dir}")
        print()
    
    # Get enabled tasks
    enabled_tasks = multitask_config.get_enabled_tasks()
    print("3. Enabled tasks (sorted by priority):")
    for task in enabled_tasks:
        print(f"   {task.task_id}: {task.task_name} (Priority: {task.priority})")
    print()
    
    # Merge with defaults
    multitask_config.merge_with_defaults()
    print("4. After merging with defaults:")
    for task_id, task in multitask_config.tasks.items():
        print(f"   Task {task_id} parameters: {task.parameters}")
    print()
    
    # Save configuration
    config_file = "example_multitask_config.yaml"
    save_task_config(multitask_config, config_file)
    print(f"5. Configuration saved to {config_file}")
    print()
    
    # Load configuration
    loaded_config = load_task_config(config_file)
    print("6. Loaded configuration from file:")
    print(f"   Project: {loaded_config.project_name} v{loaded_config.project_version}")
    print(f"   Number of tasks: {len(loaded_config.tasks)}")
    print()


def example_file_based_config():
    """Example of using file-based configuration."""
    print("=== File-Based Configuration Example ===\n")
    
    # This example assumes the multitask_example.yaml file exists
    config_file = os.path.join(os.path.dirname(__file__), "configs", "multitask_example.yaml")
    
    if os.path.exists(config_file):
        # Load configuration from file
        config = load_task_config(config_file)
        
        print("1. Loaded configuration from file:")
        print(f"   Project: {config.project_name} v{config.project_version}")
        print(f"   Global parameters: {config.global_parameters}")
        print()
        
        # Display tasks
        print("2. Tasks:")
        for task_id, task in config.tasks.items():
            print(f"   Task {task_id}: {task.task_name}")
            print(f"     Enabled: {task.enabled}")
            print(f"     Priority: {task.priority}")
            print()
        
        # Get enabled tasks
        enabled_tasks = config.get_enabled_tasks()
        print("3. Enabled tasks:")
        for task in enabled_tasks:
            print(f"   {task.task_id}: {task.task_name}")
        print()
    else:
        print(f"Configuration file {config_file} not found")
        print("Please check the examples/configs directory")
        print()


if __name__ == "__main__":
    example_programmatic_config()
    example_file_based_config()