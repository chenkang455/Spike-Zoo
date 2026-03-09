import unittest
import sys
import os
from pathlib import Path

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.models.model_registry import ModelRegistry
from spikezoo.models import (
    BaseModel, 
    BaseModelConfig,
    get_model_registry,
    register_model,
    unregister_model,
    get_model_class,
    get_config_class,
    create_model,
    list_models,
    is_model_registered
)


class TestModelRegistry(unittest.TestCase):
    """ModelRegistry unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.registry = ModelRegistry()
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        # Registry should discover some models
        models = self.registry.list_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        
        # Base model should be available
        self.assertIn("base", models)
    
    def test_register_model(self):
        """Test registering a model."""
        class TestModel(BaseModel):
            def __init__(self, cfg: BaseModelConfig):
                super().__init__(cfg)
            
            def spk2img(self, spike):
                return spike
        
        class TestModelConfig(BaseModelConfig):
            pass
        
        # Register model
        self.registry.register_model("test_model", TestModel, TestModelConfig)
        
        # Check registration
        self.assertTrue(self.registry.is_model_registered("test_model"))
        self.assertEqual(self.registry.get_model_class("test_model"), TestModel)
        self.assertEqual(self.registry.get_config_class("test_model"), TestModelConfig)
    
    def test_unregister_model(self):
        """Test unregistering a model."""
        class TestModel(BaseModel):
            def __init__(self, cfg: BaseModelConfig):
                super().__init__(cfg)
            
            def spk2img(self, spike):
                return spike
        
        # Register model
        self.registry.register_model("temp_model", TestModel)
        self.assertTrue(self.registry.is_model_registered("temp_model"))
        
        # Unregister model
        self.registry.unregister_model("temp_model")
        self.assertFalse(self.registry.is_model_registered("temp_model"))
    
    def test_get_model_class(self):
        """Test getting model class."""
        # Get existing model class
        model_class = self.registry.get_model_class("base")
        self.assertIsNotNone(model_class)
        self.assertTrue(issubclass(model_class, BaseModel))
        
        # Get non-existent model class
        model_class = self.registry.get_model_class("non_existent")
        self.assertIsNone(model_class)
    
    def test_get_config_class(self):
        """Test getting config class."""
        # Get existing config class
        config_class = self.registry.get_config_class("base")
        self.assertIsNotNone(config_class)
        self.assertTrue(issubclass(config_class, BaseModelConfig))
        
        # Get non-existent config class
        config_class = self.registry.get_config_class("non_existent")
        self.assertIsNone(config_class)
    
    def test_create_model(self):
        """Test creating model instance."""
        # Create base model
        model = self.registry.create_model("base")
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.cfg.model_name, "base")
        
        # Create model with custom config
        config = BaseModelConfig(model_name="base", load_state=True)
        model = self.registry.create_model("base", config)
        self.assertIsNotNone(model)
        self.assertTrue(model.cfg.load_state)
        
        # Try to create non-existent model
        model = self.registry.create_model("non_existent")
        self.assertIsNone(model)
    
    def test_list_models(self):
        """Test listing models."""
        models = self.registry.list_models()
        self.assertIsInstance(models, list)
        self.assertIn("base", models)
    
    def test_is_model_registered(self):
        """Test checking if model is registered."""
        # Check existing model
        self.assertTrue(self.registry.is_model_registered("base"))
        
        # Check non-existent model
        self.assertFalse(self.registry.is_model_registered("non_existent"))


class TestGlobalFunctions(unittest.TestCase):
    """Test global registry functions."""
    
    def test_global_registry_access(self):
        """Test accessing global registry."""
        registry = get_model_registry()
        self.assertIsInstance(registry, ModelRegistry)
    
    def test_global_register_unregister(self):
        """Test global register/unregister functions."""
        class TestModel(BaseModel):
            def __init__(self, cfg: BaseModelConfig):
                super().__init__(cfg)
            
            def spk2img(self, spike):
                return spike
        
        # Register model globally
        register_model("global_test_model", TestModel)
        self.assertTrue(is_model_registered("global_test_model"))
        
        # Unregister model globally
        unregister_model("global_test_model")
        self.assertFalse(is_model_registered("global_test_model"))
    
    def test_global_get_functions(self):
        """Test global get functions."""
        # Get model class
        model_class = get_model_class("base")
        self.assertIsNotNone(model_class)
        self.assertTrue(issubclass(model_class, BaseModel))
        
        # Get config class
        config_class = get_config_class("base")
        self.assertIsNotNone(config_class)
        self.assertTrue(issubclass(config_class, BaseModelConfig))
    
    def test_global_create_model(self):
        """Test global create_model function."""
        # Create model
        model = create_model("base")
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BaseModel)
    
    def test_global_list_models(self):
        """Test global list_models function."""
        models = list_models()
        self.assertIsInstance(models, list)
        self.assertIn("base", models)
    
    def test_global_is_model_registered(self):
        """Test global is_model_registered function."""
        # Check existing model
        self.assertTrue(is_model_registered("base"))
        
        # Check non-existent model
        self.assertFalse(is_model_registered("non_existent"))


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with old APIs."""
    
    def test_import_old_functions(self):
        """Test that old functions can still be imported."""
        # These imports should work
        from spikezoo.models import build_model_cfg, build_model_name
        self.assertIsNotNone(build_model_cfg)
        self.assertIsNotNone(build_model_name)
    
    def test_build_model_cfg_backward_compatibility(self):
        """Test build_model_cfg backward compatibility."""
        from spikezoo.models import build_model_cfg
        
        # Test with standard config
        config = BaseModelConfig(model_name="base")
        model = build_model_cfg(config)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BaseModel)
    
    def test_build_model_name_backward_compatibility(self):
        """Test build_model_name backward compatibility."""
        from spikezoo.models import build_model_name
        
        # Test with standard model name
        model = build_model_name("base")
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BaseModel)


if __name__ == '__main__':
    unittest.main()