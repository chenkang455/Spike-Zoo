from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PluginConfig:
    """Base plugin configuration."""
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class PluginBase(ABC):
    """Base class for all plugins."""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        """
        Initialize plugin.
        
        Args:
            config: Plugin configuration
        """
        self.config = config or PluginConfig()
        self.name = self.config.name or self.__class__.__name__
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize plugin.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute plugin functionality.
        
        Args:
            **kwargs: Execution parameters
            
        Returns:
            Execution result
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup plugin resources.
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get plugin information.
        
        Returns:
            Plugin information dictionary
        """
        return {
            "name": self.name,
            "version": self.config.version,
            "description": self.config.description,
            "author": self.config.author,
            "enabled": self.config.enabled
        }