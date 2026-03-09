from spikezoo.core.plugin_base import PluginBase, PluginConfig
from typing import Any, Dict, Optional


class ExamplePlugin(PluginBase):
    """Example plugin implementation."""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        """Initialize example plugin."""
        super().__init__(config)
        self.initialized = False
        self.execution_count = 0
    
    def initialize(self) -> bool:
        """
        Initialize plugin.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.config.enabled:
            self.initialized = True
            print(f"ExamplePlugin {self.name} initialized")
            return True
        else:
            print(f"ExamplePlugin {self.name} is disabled")
            return False
    
    def execute(self, **kwargs) -> Any:
        """
        Execute plugin functionality.
        
        Args:
            **kwargs: Execution parameters
            
        Returns:
            Execution result
        """
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        self.execution_count += 1
        message = kwargs.get("message", "Hello from ExamplePlugin!")
        repeat = kwargs.get("repeat", 1)
        
        result = message * repeat
        print(f"ExamplePlugin {self.name} executed {self.execution_count} times: {result}")
        return result
    
    def cleanup(self) -> None:
        """
        Cleanup plugin resources.
        """
        if self.initialized:
            print(f"ExamplePlugin {self.name} cleaned up after {self.execution_count} executions")
            self.initialized = False
            self.execution_count = 0


# Example usage when running as script
if __name__ == "__main__":
    # Create plugin config
    config = PluginConfig(
        name="example_task",
        version="1.0.0",
        description="An example plugin for demonstration",
        author="SpikeZoo Team"
    )
    
    # Create and initialize plugin
    plugin = ExamplePlugin(config)
    plugin.initialize()
    
    # Execute plugin
    result = plugin.execute(message="Hello! ", repeat=3)
    print(f"Result: {result}")
    
    # Cleanup
    plugin.cleanup()