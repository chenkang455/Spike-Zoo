from typing import Dict, List, Optional, Type, Any
from pathlib import Path
import importlib.util
import logging
from .plugin_base import PluginBase, PluginConfig


class TaskRegistry:
    """Registry for managing plugins/tasks."""
    
    def __init__(self):
        """Initialize task registry."""
        self.plugins: Dict[str, Type[PluginBase]] = {}
        self.instances: Dict[str, PluginBase] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_plugin(self, name: str, plugin_class: Type[PluginBase]) -> None:
        """
        Register a plugin class.
        
        Args:
            name: Plugin name
            plugin_class: Plugin class
        """
        if name in self.plugins:
            self.logger.warning(f"Plugin {name} already registered, overwriting")
        
        self.plugins[name] = plugin_class
        self.logger.info(f"Registered plugin: {name}")
    
    def unregister_plugin(self, name: str) -> None:
        """
        Unregister a plugin class.
        
        Args:
            name: Plugin name
        """
        if name in self.plugins:
            del self.plugins[name]
            self.logger.info(f"Unregistered plugin: {name}")
        
        # Also remove instance if exists
        if name in self.instances:
            del self.instances[name]
    
    def get_plugin_class(self, name: str) -> Optional[Type[PluginBase]]:
        """
        Get registered plugin class.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin class or None if not found
        """
        return self.plugins.get(name)
    
    def create_plugin_instance(
        self, 
        name: str, 
        config: Optional[PluginConfig] = None
    ) -> Optional[PluginBase]:
        """
        Create plugin instance.
        
        Args:
            name: Plugin name
            config: Plugin configuration
            
        Returns:
            Plugin instance or None if plugin not found
        """
        plugin_class = self.get_plugin_class(name)
        if plugin_class is None:
            self.logger.error(f"Plugin {name} not found")
            return None
        
        try:
            instance = plugin_class(config)
            self.instances[name] = instance
            return instance
        except Exception as e:
            self.logger.error(f"Failed to create instance of plugin {name}: {e}")
            return None
    
    def get_plugin_instance(self, name: str) -> Optional[PluginBase]:
        """
        Get existing plugin instance.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self.instances.get(name)
    
    def list_plugins(self) -> List[str]:
        """
        List all registered plugins.
        
        Returns:
            List of plugin names
        """
        return list(self.plugins.keys())
    
    def load_plugin_from_file(self, file_path: str, class_name: str) -> bool:
        """
        Load plugin from Python file.
        
        Args:
            file_path: Path to Python file
            class_name: Name of plugin class in file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location("plugin_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get plugin class
            plugin_class = getattr(module, class_name, None)
            if plugin_class is None:
                self.logger.error(f"Class {class_name} not found in {file_path}")
                return False
            
            # Check if class is subclass of PluginBase
            if not issubclass(plugin_class, PluginBase):
                self.logger.error(f"Class {class_name} is not a subclass of PluginBase")
                return False
            
            # Register plugin
            plugin_name = plugin_class.__name__
            self.register_plugin(plugin_name, plugin_class)
            return True
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {file_path}: {e}")
            return False


class TaskManager:
    """Task manager for handling multiple plugins."""
    
    def __init__(self):
        """Initialize task manager."""
        self.registry = TaskRegistry()
        self.logger = logging.getLogger(__name__)
        self.active_tasks: Dict[str, PluginBase] = {}
    
    def register_task(self, name: str, task_class: Type[PluginBase]) -> None:
        """
        Register a task.
        
        Args:
            name: Task name
            task_class: Task class
        """
        self.registry.register_plugin(name, task_class)
    
    def load_task_from_file(self, file_path: str, class_name: str) -> bool:
        """
        Load task from Python file.
        
        Args:
            file_path: Path to Python file
            class_name: Name of task class in file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        return self.registry.load_plugin_from_file(file_path, class_name)
    
    def start_task(
        self, 
        name: str, 
        config: Optional[PluginConfig] = None
    ) -> bool:
        """
        Start a task.
        
        Args:
            name: Task name
            config: Task configuration
            
        Returns:
            True if started successfully, False otherwise
        """
        # Check if task is already active
        if name in self.active_tasks:
            self.logger.warning(f"Task {name} is already active")
            return True
        
        # Create instance if not exists
        instance = self.registry.get_plugin_instance(name)
        if instance is None:
            instance = self.registry.create_plugin_instance(name, config)
            if instance is None:
                return False
        
        # Initialize task
        try:
            if instance.initialize():
                self.active_tasks[name] = instance
                self.logger.info(f"Started task: {name}")
                return True
            else:
                self.logger.error(f"Failed to initialize task: {name}")
                return False
        except Exception as e:
            self.logger.error(f"Exception during task {name} initialization: {e}")
            return False
    
    def stop_task(self, name: str) -> bool:
        """
        Stop a task.
        
        Args:
            name: Task name
            
        Returns:
            True if stopped successfully, False otherwise
        """
        if name not in self.active_tasks:
            self.logger.warning(f"Task {name} is not active")
            return False
        
        try:
            instance = self.active_tasks[name]
            instance.cleanup()
            del self.active_tasks[name]
            self.logger.info(f"Stopped task: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Exception during task {name} cleanup: {e}")
            return False
    
    def execute_task(self, name: str, **kwargs) -> Any:
        """
        Execute a task.
        
        Args:
            name: Task name
            **kwargs: Execution parameters
            
        Returns:
            Execution result or None if task not active
        """
        if name not in self.active_tasks:
            self.logger.error(f"Task {name} is not active")
            return None
        
        try:
            instance = self.active_tasks[name]
            return instance.execute(**kwargs)
        except Exception as e:
            self.logger.error(f"Exception during task {name} execution: {e}")
            return None
    
    def list_tasks(self) -> Dict[str, str]:
        """
        List all tasks and their status.
        
        Returns:
            Dictionary of task names and their status
        """
        all_tasks = self.registry.list_plugins()
        status_dict = {}
        
        for task_name in all_tasks:
            if task_name in self.active_tasks:
                status_dict[task_name] = "active"
            else:
                status_dict[task_name] = "inactive"
        
        return status_dict
    
    def get_task_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get task information.
        
        Args:
            name: Task name
            
        Returns:
            Task information or None if task not found
        """
        instance = self.registry.get_plugin_instance(name)
        if instance is None:
            return None
        
        return instance.get_info()