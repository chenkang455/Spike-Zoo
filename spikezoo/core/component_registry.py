from typing import Dict, Type, Optional, Callable, Any, Union
from dataclasses import dataclass, field
import importlib
import logging
from pathlib import Path
import os
from enum import Enum


class ComponentType(Enum):
    """Component types."""
    MODEL = "model"
    DATASET = "dataset"
    PIPELINE = "pipeline"
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    METRIC = "metric"
    LOSS = "loss"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    CUSTOM = "custom"


@dataclass
class ComponentInfo:
    """Component information."""
    name: str
    component_type: ComponentType
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    category: str = "general"
    tags: list = field(default_factory=list)
    config_class: Optional[Type] = None
    component_class: Optional[Type] = None
    factory_function: Optional[Callable] = None
    dependencies: list = field(default_factory=list)


class ComponentRegistry:
    """Registry for managing pipeline components and their plugins."""
    
    def __init__(self):
        """Initialize component registry."""
        self.components: Dict[str, ComponentInfo] = {}
        self.factories: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
        self._discovered_modules = set()
    
    def register_component(
        self, 
        name: str,
        component_type: ComponentType,
        component_class: Optional[Type] = None,
        config_class: Optional[Type] = None,
        factory_function: Optional[Callable] = None,
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        category: str = "general",
        tags: Optional[list] = None,
        dependencies: Optional[list] = None
    ):
        """
        Register a component.
        
        Args:
            name: Component name
            component_type: Component type
            component_class: Component class (optional)
            config_class: Configuration class (optional)
            factory_function: Factory function to create component instance (optional)
            version: Component version
            description: Component description
            author: Component author
            category: Component category
            tags: Component tags
            dependencies: Component dependencies
        """
        if name in self.components:
            self.logger.warning(f"Component {name} already registered, overwriting")
        
        component_info = ComponentInfo(
            name=name,
            component_type=component_type,
            version=version,
            description=description,
            author=author,
            category=category,
            tags=tags or [],
            config_class=config_class,
            component_class=component_class,
            factory_function=factory_function,
            dependencies=dependencies or []
        )
        
        self.components[name] = component_info
        
        # Register factory function if provided
        if factory_function:
            self.factories[name] = factory_function
        
        self.logger.info(f"Registered component: {name} ({component_type.value})")
    
    def unregister_component(self, name: str):
        """
        Unregister a component.
        
        Args:
            name: Component name
        """
        if name in self.components:
            component_type = self.components[name].component_type.value
            del self.components[name]
            self.logger.info(f"Unregistered component: {name} ({component_type})")
        
        if name in self.factories:
            del self.factories[name]
    
    def get_component_info(self, name: str) -> Optional[ComponentInfo]:
        """
        Get component information.
        
        Args:
            name: Component name
            
        Returns:
            ComponentInfo or None if not found
        """
        return self.components.get(name)
    
    def list_components(self) -> list:
        """
        List all registered components.
        
        Returns:
            List of component names
        """
        return list(self.components.keys())
    
    def list_components_by_type(self, component_type: Union[ComponentType, str]) -> list:
        """
        List components by type.
        
        Args:
            component_type: Component type
            
        Returns:
            List of component names of the specified type
        """
        if isinstance(component_type, str):
            component_type = ComponentType(component_type)
        
        return [name for name, info in self.components.items() if info.component_type == component_type]
    
    def list_components_by_category(self, category: str) -> list:
        """
        List components by category.
        
        Args:
            category: Component category
            
        Returns:
            List of component names in the category
        """
        return [name for name, info in self.components.items() if info.category == category]
    
    def list_components_by_tag(self, tag: str) -> list:
        """
        List components by tag.
        
        Args:
            tag: Component tag
            
        Returns:
            List of component names with the tag
        """
        return [name for name, info in self.components.items() if tag in info.tags]
    
    def create_component(self, name: str, *args, **kwargs) -> Any:
        """
        Create component instance.
        
        Args:
            name: Component name
            *args: Positional arguments for component constructor
            **kwargs: Keyword arguments for component constructor
            
        Returns:
            Component instance
            
        Raises:
            ValueError: If component not found or cannot be created
        """
        if name not in self.components:
            raise ValueError(f"Component {name} not found in registry")
        
        component_info = self.components[name]
        
        # Try factory function first
        if name in self.factories:
            try:
                return self.factories[name](*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to create component {name} using factory: {e}")
                raise
        
        # Try component class
        if component_info.component_class:
            try:
                return component_info.component_class(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to create component {name} using class: {e}")
                raise
        
        raise ValueError(f"Cannot create component {name}: no factory function or component class available")
    
    def create_component_with_config(self, name: str, config: Any) -> Any:
        """
        Create component instance with configuration.
        
        Args:
            name: Component name
            config: Component configuration
            
        Returns:
            Component instance
            
        Raises:
            ValueError: If component not found or cannot be created
        """
        if name not in self.components:
            raise ValueError(f"Component {name} not found in registry")
        
        component_info = self.components[name]
        
        # Try factory function with config
        if name in self.factories:
            try:
                return self.factories[name](config)
            except Exception as e:
                self.logger.error(f"Failed to create component {name} using factory with config: {e}")
                raise
        
        # Try component class with config
        if component_info.component_class:
            try:
                return component_info.component_class(config)
            except Exception as e:
                self.logger.error(f"Failed to create component {name} using class with config: {e}")
                raise
        
        raise ValueError(f"Cannot create component {name}: no factory function or component class available")
    
    def discover_components_from_directory(self, directory: str, package_prefix: str = ""):
        """
        Discover and register components from a directory.
        
        Args:
            directory: Directory to search for components
            package_prefix: Package prefix for import
        """
        directory = Path(directory)
        if not directory.exists():
            self.logger.warning(f"Directory {directory} does not exist")
            return
        
        # Walk through directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("_component.py") or file.endswith("_plugin.py"):
                    module_name = file[:-3]  # Remove .py extension
                    full_module_name = f"{package_prefix}.{module_name}" if package_prefix else module_name
                    
                    # Skip if already discovered
                    if full_module_name in self._discovered_modules:
                        continue
                    
                    try:
                        # Import module
                        module = importlib.import_module(full_module_name)
                        self._discovered_modules.add(full_module_name)
                        
                        # Look for component registration function
                        if hasattr(module, "register_components"):
                            module.register_components(self)
                            self.logger.info(f"Discovered components from {full_module_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to import module {full_module_name}: {e}")
    
    def get_component_types(self) -> list:
        """
        Get all component types.
        
        Returns:
            List of unique component types
        """
        types = set()
        for info in self.components.values():
            types.add(info.component_type)
        return list(types)
    
    def get_component_categories(self) -> list:
        """
        Get all component categories.
        
        Returns:
            List of unique categories
        """
        categories = set()
        for info in self.components.values():
            categories.add(info.category)
        return list(categories)
    
    def get_component_tags(self) -> list:
        """
        Get all component tags.
        
        Returns:
            List of unique tags
        """
        tags = set()
        for info in self.components.values():
            tags.update(info.tags)
        return list(tags)
    
    def get_component_dependencies(self, name: str) -> list:
        """
        Get component dependencies.
        
        Args:
            name: Component name
            
        Returns:
            List of component dependencies
        """
        if name not in self.components:
            return []
        
        return self.components[name].dependencies
    
    def check_dependencies(self, name: str) -> bool:
        """
        Check if all component dependencies are satisfied.
        
        Args:
            name: Component name
            
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        if name not in self.components:
            return False
        
        dependencies = self.components[name].dependencies
        for dep in dependencies:
            if dep not in self.components:
                return False
        return True


# Global component registry instance
_component_registry = ComponentRegistry()


def register_component(
    name: str,
    component_type: ComponentType,
    component_class: Optional[Type] = None,
    config_class: Optional[Type] = None,
    factory_function: Optional[Callable] = None,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    category: str = "general",
    tags: Optional[list] = None,
    dependencies: Optional[list] = None
):
    """
    Register a component in the global registry.
    
    Args:
        name: Component name
        component_type: Component type
        component_class: Component class (optional)
        config_class: Configuration class (optional)
        factory_function: Factory function to create component instance (optional)
        version: Component version
        description: Component description
        author: Component author
        category: Component category
        tags: Component tags
        dependencies: Component dependencies
    """
    _component_registry.register_component(
        name, component_type, component_class, config_class, factory_function,
        version, description, author, category, tags, dependencies
    )


def unregister_component(name: str):
    """
    Unregister a component from the global registry.
    
    Args:
        name: Component name
    """
    _component_registry.unregister_component(name)


def get_component_info(name: str) -> Optional[ComponentInfo]:
    """
    Get component information from the global registry.
    
    Args:
        name: Component name
        
    Returns:
        ComponentInfo or None if not found
    """
    return _component_registry.get_component_info(name)


def list_components() -> list:
    """
    List all registered components from the global registry.
    
    Returns:
        List of component names
    """
    return _component_registry.list_components()


def list_components_by_type(component_type: Union[ComponentType, str]) -> list:
    """
    List components by type from the global registry.
    
    Args:
        component_type: Component type
        
    Returns:
        List of component names of the specified type
    """
    return _component_registry.list_components_by_type(component_type)


def create_component(name: str, *args, **kwargs) -> Any:
    """
    Create component instance from the global registry.
    
    Args:
        name: Component name
        *args: Positional arguments for component constructor
        **kwargs: Keyword arguments for component constructor
        
    Returns:
        Component instance
    """
    return _component_registry.create_component(name, *args, **kwargs)


def create_component_with_config(name: str, config: Any) -> Any:
    """
    Create component instance with configuration from the global registry.
    
    Args:
        name: Component name
        config: Component configuration
        
    Returns:
        Component instance
    """
    return _component_registry.create_component_with_config(name, config)


def get_component_registry() -> ComponentRegistry:
    """
    Get the global component registry instance.
    
    Returns:
        ComponentRegistry instance
    """
    return _component_registry


def discover_components_from_directory(directory: str, package_prefix: str = ""):
    """
    Discover and register components from a directory in the global registry.
    
    Args:
        directory: Directory to search for components
        package_prefix: Package prefix for import
    """
    _component_registry.discover_components_from_directory(directory, package_prefix)


def check_component_dependencies(name: str) -> bool:
    """
    Check if all component dependencies are satisfied in the global registry.
    
    Args:
        name: Component name
        
    Returns:
        True if all dependencies are satisfied, False otherwise
    """
    return _component_registry.check_dependencies(name)