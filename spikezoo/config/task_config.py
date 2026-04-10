from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import yaml
import json


@dataclass
class TaskConfig:
    """Base configuration for a single task."""
    # Task identification
    task_id: str = ""
    task_name: str = ""
    task_description: str = ""
    
    # Task execution
    enabled: bool = True
    priority: int = 0  # Lower number means higher priority
    
    # Task parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    
    # Output configuration
    output_dir: str = ""
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create TaskConfig from dictionary."""
        # Filter out keys that are not in the dataclass fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        return cls(**filtered_dict)
    
    def to_dict(self):
        """Convert TaskConfig to dictionary."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_description": self.task_description,
            "enabled": self.enabled,
            "priority": self.priority,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "output_dir": self.output_dir
        }


@dataclass
class MultiTaskConfig:
    """Configuration for multiple tasks."""
    # Global settings
    project_name: str = ""
    project_version: str = "1.0.0"
    
    # Default task settings
    default_task_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Tasks
    tasks: Dict[str, TaskConfig] = field(default_factory=dict)
    
    # Global parameters
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create MultiTaskConfig from dictionary."""
        # Handle tasks
        tasks = {}
        if "tasks" in config_dict:
            for task_id, task_dict in config_dict["tasks"].items():
                tasks[task_id] = TaskConfig.from_dict(task_dict)
        
        # Create config with tasks
        config = cls(
            project_name=config_dict.get("project_name", ""),
            project_version=config_dict.get("project_version", "1.0.0"),
            default_task_settings=config_dict.get("default_task_settings", {}),
            global_parameters=config_dict.get("global_parameters", {}),
            tasks=tasks
        )
        
        return config
    
    def to_dict(self):
        """Convert MultiTaskConfig to dictionary."""
        return {
            "project_name": self.project_name,
            "project_version": self.project_version,
            "default_task_settings": self.default_task_settings,
            "global_parameters": self.global_parameters,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        }
    
    def add_task(self, task_config: TaskConfig):
        """Add a task to the configuration."""
        self.tasks[task_config.task_id] = task_config
    
    def get_task(self, task_id: str) -> Optional[TaskConfig]:
        """Get a task configuration by ID."""
        return self.tasks.get(task_id)
    
    def get_enabled_tasks(self) -> List[TaskConfig]:
        """Get all enabled tasks, sorted by priority."""
        enabled_tasks = [task for task in self.tasks.values() if task.enabled]
        return sorted(enabled_tasks, key=lambda t: t.priority)
    
    def merge_with_defaults(self):
        """Merge task configurations with default settings."""
        for task in self.tasks.values():
            # Merge parameters with defaults
            merged_params = self.default_task_settings.copy()
            merged_params.update(task.parameters)
            task.parameters = merged_params


class TaskConfigManager:
    """Manager for task configurations."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> MultiTaskConfig:
        """
        Load multi-task configuration from file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            MultiTaskConfig instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        config_dict = config_dict or {}
        return MultiTaskConfig.from_dict(config_dict)
    
    @staticmethod
    def save_config(config: MultiTaskConfig, config_path: Union[str, Path], format: str = 'yaml'):
        """
        Save multi-task configuration to file.
        
        Args:
            config: MultiTaskConfig instance
            config_path: Path to save the configuration file
            format: Output format ('yaml' or 'json')
        """
        config_path = Path(config_path)
        
        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if format.lower() in ['yaml', 'yml']:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
            elif format.lower() == 'json':
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {format}")


def load_task_config(config_path: Union[str, Path]) -> MultiTaskConfig:
    """
    Load task configuration from file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        MultiTaskConfig instance
    """
    return TaskConfigManager.load_config(config_path)


def save_task_config(config: MultiTaskConfig, config_path: Union[str, Path], format: str = 'yaml'):
    """
    Save task configuration to file.
    
    Args:
        config: MultiTaskConfig instance
        config_path: Path to save the configuration file
        format: Output format ('yaml' or 'json')
    """
    TaskConfigManager.save_config(config, config_path, format)