import yaml
import json
from pathlib import Path
from typing import Any, Dict, Union
import logging


class ConfigManager:
    """Configuration manager for YAML/JSON config files."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                
        return config or {}
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: Union[str, Path], format: str = 'yaml') -> None:
        """
        Save configuration to YAML or JSON file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save the configuration file
            format: Output format ('yaml' or 'json')
        """
        config_path = Path(config_path)
        
        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if format.lower() in ['yaml', 'yml']:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            elif format.lower() == 'json':
                json.dump(config, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {format}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    return ConfigManager.load_config(config_path)


def save_config(config: Dict[str, Any], config_path: Union[str, Path], format: str = 'yaml') -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
        format: Output format ('yaml' or 'json')
    """
    ConfigManager.save_config(config, config_path, format)