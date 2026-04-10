"""
Architecture loader for SpikeZoo models with plugin support.
"""

import importlib
import importlib.util
from typing import Optional, Type, Dict, Any, Union
from pathlib import Path
import os
import sys
from spikezoo.models.base_model import BaseModel, BaseModelConfig
from spikezoo.archs.base.nets import BaseNet


class ArchitectureLoader:
    """Loader for model architectures with plugin support."""
    
    def __init__(self):
        """Initialize architecture loader."""
        self._loaded_modules: Dict[str, Any] = {}
        self._architecture_classes: Dict[str, Type[BaseNet]] = {}
        self._search_paths: list = []
        self._initialize_search_paths()
    
    def _initialize_search_paths(self):
        """Initialize search paths for architectures."""
        # Add default search paths
        self._search_paths.extend([
            Path(__file__).parent.parent / "archs",  # Built-in architectures
            Path.cwd() / "custom_archs",             # Custom architectures in current directory
        ])
        
        # Add paths from environment variable
        custom_path = os.environ.get("SPIKEZOO_ARCH_PATH")
        if custom_path:
            self._search_paths.append(Path(custom_path))
    
    def add_search_path(self, path: Union[str, Path]):
        """
        Add a custom search path for architectures.
        
        Args:
            path: Path to add to search paths
        """
        path = Path(path)
        if path not in self._search_paths:
            self._search_paths.append(path)
    
    def load_architecture_class(
        self, 
        model_name: str, 
        class_name: Optional[str] = None,
        module_name: Optional[str] = None
    ) -> Optional[Type[BaseNet]]:
        """
        Load architecture class dynamically.
        
        Args:
            model_name: Name of the model
            class_name: Name of the class to load (defaults to model_name + 'Net')
            module_name: Name of the module to load from (defaults to 'nets')
            
        Returns:
            Architecture class or None if not found
        """
        # Use defaults if not specified
        if class_name is None:
            class_name = f"{model_name.capitalize()}Net"
        if module_name is None:
            module_name = "nets"
        
        # Check cache first
        cache_key = f"{model_name}.{module_name}.{class_name}"
        if cache_key in self._architecture_classes:
            return self._architecture_classes[cache_key]
        
        # Try to load from various locations
        architecture_class = None
        
        # 1. Try built-in architectures
        architecture_class = self._load_from_built_in(model_name, module_name, class_name)
        if architecture_class is not None:
            self._architecture_classes[cache_key] = architecture_class
            return architecture_class
        
        # 2. Try custom architectures in search paths
        architecture_class = self._load_from_search_paths(model_name, module_name, class_name)
        if architecture_class is not None:
            self._architecture_classes[cache_key] = architecture_class
            return architecture_class
        
        # 3. Try direct module specification
        architecture_class = self._load_direct_module(model_name, module_name, class_name)
        if architecture_class is not None:
            self._architecture_classes[cache_key] = architecture_class
            return architecture_class
        
        return None
    
    def _load_from_built_in(
        self, 
        model_name: str, 
        module_name: str, 
        class_name: str
    ) -> Optional[Type[BaseNet]]:
        """
        Load architecture from built-in architectures.
        
        Args:
            model_name: Name of the model
            module_name: Name of the module
            class_name: Name of the class
            
        Returns:
            Architecture class or None if not found
        """
        try:
            full_module_name = f"spikezoo.archs.{model_name}.{module_name}"
            module = importlib.import_module(full_module_name)
            self._loaded_modules[full_module_name] = module
            
            if hasattr(module, class_name):
                architecture_class = getattr(module, class_name)
                if issubclass(architecture_class, BaseNet):
                    return architecture_class
        except (ImportError, AttributeError):
            pass
        
        return None
    
    def _load_from_search_paths(
        self, 
        model_name: str, 
        module_name: str, 
        class_name: str
    ) -> Optional[Type[BaseNet]]:
        """
        Load architecture from custom search paths.
        
        Args:
            model_name: Name of the model
            module_name: Name of the module
            class_name: Name of the class
            
        Returns:
            Architecture class or None if not found
        """
        for search_path in self._search_paths:
            try:
                # Construct module path
                module_path = search_path / model_name / f"{module_name}.py"
                
                if module_path.exists():
                    # Load module from file
                    spec = importlib.util.spec_from_file_location(
                        f"custom_{model_name}_{module_name}", 
                        module_path
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Cache module
                        module_key = f"custom.{model_name}.{module_name}"
                        self._loaded_modules[module_key] = module
                        
                        # Get class
                        if hasattr(module, class_name):
                            architecture_class = getattr(module, class_name)
                            if issubclass(architecture_class, BaseNet):
                                return architecture_class
            except Exception:
                continue
        
        return None
    
    def _load_direct_module(
        self, 
        model_name: str, 
        module_name: str, 
        class_name: str
    ) -> Optional[Type[BaseNet]]:
        """
        Load architecture by directly importing module.
        
        Args:
            model_name: Name of the model
            module_name: Name of the module
            class_name: Name of the class
            
        Returns:
            Architecture class or None if not found
        """
        try:
            # Try direct import
            module = importlib.import_module(module_name)
            self._loaded_modules[module_name] = module
            
            if hasattr(module, class_name):
                architecture_class = getattr(module, class_name)
                if issubclass(architecture_class, BaseNet):
                    return architecture_class
        except (ImportError, AttributeError):
            pass
        
        return None
    
    def create_architecture(
        self, 
        model_name: str, 
        config: Optional[Dict[str, Any]] = None,
        class_name: Optional[str] = None,
        module_name: Optional[str] = None
    ) -> Optional[BaseNet]:
        """
        Create architecture instance.
        
        Args:
            model_name: Name of the model
            config: Configuration dictionary for architecture
            class_name: Name of the class to load
            module_name: Name of the module to load from
            
        Returns:
            Architecture instance or None if creation failed
        """
        # Load architecture class
        architecture_class = self.load_architecture_class(model_name, class_name, module_name)
        
        if architecture_class is None:
            return None
        
        # Create instance with config
        try:
            if config is not None:
                return architecture_class(**config)
            else:
                return architecture_class()
        except Exception:
            return None
    
    def list_available_architectures(self) -> list:
        """
        List available architectures in built-in directories.
        
        Returns:
            List of available architecture names
        """
        architectures = []
        
        # Check built-in architectures
        archs_dir = Path(__file__).parent.parent / "archs"
        if archs_dir.exists():
            for item in archs_dir.iterdir():
                if item.is_dir() and not item.name.startswith("__"):
                    # Check if it has a nets.py file
                    nets_file = item / "nets.py"
                    if nets_file.exists():
                        architectures.append(item.name)
        
        return architectures
    
    def clear_cache(self):
        """Clear loaded modules and classes cache."""
        self._loaded_modules.clear()
        self._architecture_classes.clear()


# Global architecture loader instance
_architecture_loader = ArchitectureLoader()


def get_architecture_loader() -> ArchitectureLoader:
    """
    Get the global architecture loader instance.
    
    Returns:
        ArchitectureLoader instance
    """
    return _architecture_loader


def load_architecture_class(
    model_name: str, 
    class_name: Optional[str] = None,
    module_name: Optional[str] = None
) -> Optional[Type[BaseNet]]:
    """
    Load architecture class using global loader.
    
    Args:
        model_name: Name of the model
        class_name: Name of the class to load
        module_name: Name of the module to load from
        
    Returns:
        Architecture class or None if not found
    """
    return _architecture_loader.load_architecture_class(model_name, class_name, module_name)


def create_architecture(
    model_name: str, 
    config: Optional[Dict[str, Any]] = None,
    class_name: Optional[str] = None,
    module_name: Optional[str] = None
) -> Optional[BaseNet]:
    """
    Create architecture instance using global loader.
    
    Args:
        model_name: Name of the model
        config: Configuration dictionary for architecture
        class_name: Name of the class to load
        module_name: Name of the module to load from
        
    Returns:
        Architecture instance or None if creation failed
    """
    return _architecture_loader.create_architecture(model_name, config, class_name, module_name)


def add_architecture_search_path(path: Union[str, Path]):
    """
    Add a custom search path for architectures using global loader.
    
    Args:
        path: Path to add to search paths
    """
    _architecture_loader.add_search_path(path)


def list_available_architectures() -> list:
    """
    List available architectures using global loader.
    
    Returns:
        List of available architecture names
    """
    return _architecture_loader.list_available_architectures()


def clear_architecture_cache():
    """Clear architecture loader cache using global loader."""
    _architecture_loader.clear_cache()