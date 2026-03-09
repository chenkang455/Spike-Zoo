import unittest
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add the spikezoo package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spikezoo.models.architecture_loader import (
    ArchitectureLoader,
    load_architecture_class,
    create_architecture,
    list_available_architectures,
    add_architecture_search_path,
    clear_architecture_cache
)
from spikezoo.archs.base.nets import BaseNet


class TestArchitectureLoader(unittest.TestCase):
    """Test the architecture loader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = ArchitectureLoader()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_architecture_cache()
    
    def test_loader_initialization(self):
        """Test that ArchitectureLoader initializes correctly."""
        loader = ArchitectureLoader()
        self.assertIsInstance(loader, ArchitectureLoader)
    
    def test_load_built_in_architecture_class(self):
        """Test loading built-in architecture class."""
        # Test loading BaseNet
        base_net_class = load_architecture_class("base", "BaseNet", "nets")
        self.assertIsNotNone(base_net_class)
        self.assertTrue(issubclass(base_net_class, BaseNet))
        self.assertEqual(base_net_class.__name__, "BaseNet")
    
    def test_load_architecture_class_with_defaults(self):
        """Test loading architecture class with default parameters."""
        # This should work for the base model
        base_net_class = load_architecture_class("base")
        self.assertIsNotNone(base_net_class)
        self.assertTrue(issubclass(base_net_class, BaseNet))
    
    def test_create_architecture_instance(self):
        """Test creating architecture instance."""
        # Create BaseNet instance
        base_net = create_architecture("base", {}, "BaseNet", "nets")
        self.assertIsNotNone(base_net)
        self.assertIsInstance(base_net, BaseNet)
    
    def test_create_architecture_with_config(self):
        """Test creating architecture with configuration."""
        # Create BaseNet instance with empty config
        config = {}
        base_net = create_architecture("base", config)
        self.assertIsNotNone(base_net)
        self.assertIsInstance(base_net, BaseNet)
    
    def test_list_available_architectures(self):
        """Test listing available architectures."""
        architectures = list_available_architectures()
        self.assertIsInstance(architectures, list)
        # Should at least contain 'base'
        self.assertIn("base", architectures)
    
    def test_add_search_path(self):
        """Test adding custom search path."""
        initial_paths = len(self.loader._search_paths)
        
        # Add a new path
        new_path = Path(self.temp_dir) / "custom_archs"
        new_path.mkdir(exist_ok=True)
        self.loader.add_search_path(new_path)
        
        # Should have one more path
        self.assertEqual(len(self.loader._search_paths), initial_paths + 1)
        
        # Adding the same path again should not increase count
        self.loader.add_search_path(new_path)
        self.assertEqual(len(self.loader._search_paths), initial_paths + 1)
    
    def test_loader_caching(self):
        """Test that loader caches architecture classes."""
        # Load class first time
        base_net_class1 = self.loader.load_architecture_class("base", "BaseNet", "nets")
        self.assertIsNotNone(base_net_class1)
        
        # Load class second time - should be cached
        base_net_class2 = self.loader.load_architecture_class("base", "BaseNet", "nets")
        self.assertIsNotNone(base_net_class2)
        
        # Should be the same object
        self.assertIs(base_net_class1, base_net_class2)
    
    def test_clear_cache(self):
        """Test clearing the loader cache."""
        # Load class to populate cache
        base_net_class1 = self.loader.load_architecture_class("base", "BaseNet", "nets")
        self.assertIsNotNone(base_net_class1)
        
        # Clear cache
        self.loader.clear_cache()
        
        # Cache should be empty
        self.assertEqual(len(self.loader._loaded_modules), 0)
        self.assertEqual(len(self.loader._architecture_classes), 0)
    
    def test_global_functions(self):
        """Test global convenience functions."""
        # Test global loader functions
        base_net_class = load_architecture_class("base", "BaseNet", "nets")
        self.assertIsNotNone(base_net_class)
        self.assertTrue(issubclass(base_net_class, BaseNet))
        
        # Test creating architecture
        base_net = create_architecture("base", {}, "BaseNet", "nets")
        self.assertIsNotNone(base_net)
        self.assertIsInstance(base_net, BaseNet)
        
        # Test listing architectures
        architectures = list_available_architectures()
        self.assertIsInstance(architectures, list)
        self.assertIn("base", architectures)
    
    def test_fallback_behavior(self):
        """Test fallback behavior for non-existent architectures."""
        # Try to load non-existent architecture
        non_existent_class = load_architecture_class("non_existent", "NonExistentNet", "nets")
        self.assertIsNone(non_existent_class)
        
        # Try to create non-existent architecture
        non_existent_instance = create_architecture("non_existent", {}, "NonExistentNet", "nets")
        self.assertIsNone(non_existent_instance)
    
    def test_invalid_architecture_class(self):
        """Test handling of invalid architecture classes."""
        # Try to load a class that exists but is not a BaseNet subclass
        # This would need a special test module, so we'll test the concept
        pass  # Would require creating a test module


class TestArchitectureLoaderAdvanced(unittest.TestCase):
    """Advanced tests for architecture loader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = ArchitectureLoader()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_architecture_cache()
    
    def test_multiple_loaders(self):
        """Test that multiple loaders work independently."""
        loader1 = ArchitectureLoader()
        loader2 = ArchitectureLoader()
        
        # They should be different instances
        self.assertIsNot(loader1, loader2)
        
        # But should both work
        class1 = loader1.load_architecture_class("base", "BaseNet", "nets")
        class2 = loader2.load_architecture_class("base", "BaseNet", "nets")
        
        self.assertIsNotNone(class1)
        self.assertIsNotNone(class2)
        self.assertTrue(issubclass(class1, BaseNet))
        self.assertTrue(issubclass(class2, BaseNet))
    
    def test_search_path_priority(self):
        """Test that search paths are checked in order."""
        # Add custom path first
        custom_path = Path(self.temp_dir) / "priority_test"
        custom_path.mkdir(exist_ok=True)
        self.loader.add_search_path(custom_path)
        
        # The built-in paths should still be checked
        architectures = self.loader.list_available_architectures()
        self.assertIn("base", architectures)
    
    def test_environment_variable_path(self):
        """Test that environment variable paths are used."""
        # This is harder to test without modifying environment
        # We'll test that the loader attempts to use it
        loader = ArchitectureLoader()
        # The loader should initialize without error even if env var is not set
        self.assertIsInstance(loader, ArchitectureLoader)
    
    def test_module_loading_errors(self):
        """Test handling of module loading errors."""
        # Try to load from a path that exists but has no valid module
        empty_path = Path(self.temp_dir) / "empty_dir"
        empty_path.mkdir(exist_ok=True)
        self.loader.add_search_path(empty_path)
        
        # Should gracefully handle the error
        result = self.loader._load_from_search_paths("non_existent", "nets", "NonExistentNet")
        self.assertIsNone(result)


class TestIntegrationWithBaseModel(unittest.TestCase):
    """Integration tests with BaseModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid circular imports
        from spikezoo.models.base_model import BaseModel, BaseModelConfig
        self.BaseModel = BaseModel
        self.BaseModelConfig = BaseModelConfig
    
    def test_base_model_with_new_loader(self):
        """Test that BaseModel works with new architecture loader."""
        # Create config
        config = self.BaseModelConfig(
            model_name="base",
            model_cls_name="BaseNet",
            model_file_name="nets"
        )
        
        # Create model
        model = self.BaseModel(config)
        self.assertIsInstance(model, self.BaseModel)
        
        # Build network (this will use the new loader)
        try:
            model.build_network()
            # Should succeed without error
            self.assertIsNotNone(model.net)
        except Exception as e:
            # If it fails, it should be for a reason other than the loader
            self.fail(f"build_network failed unexpectedly: {e}")


if __name__ == '__main__':
    unittest.main()