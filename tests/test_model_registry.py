import unittest
import sys
import os
import torch
import torch.nn as nn

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.core.model_registry import (
    ModelRegistry,
    ModelInfo,
    register_model,
    unregister_model,
    get_model_info,
    list_models,
    create_model,
    create_model_with_config,
    get_model_registry,
    discover_models_from_directory
)


class TestModelInfo(unittest.TestCase):
    """ModelInfo unit tests."""
    
    def test_model_info_creation(self):
        """Test ModelInfo creation."""
        model_info = ModelInfo(
            name="test_model",
            version="1.0.0",
            description="Test model",
            author="Test Author",
            category="test",
            tags=["tag1", "tag2"]
        )
        
        self.assertEqual(model_info.name, "test_model")
        self.assertEqual(model_info.version, "1.0.0")
        self.assertEqual(model_info.description, "Test model")
        self.assertEqual(model_info.author, "Test Author")
        self.assertEqual(model_info.category, "test")
        self.assertEqual(model_info.tags, ["tag1", "tag2"])
        self.assertIsNone(model_info.config_class)
        self.assertIsNone(model_info.model_class)
        self.assertIsNone(model_info.factory_function)


class TestModelRegistry(unittest.TestCase):
    """ModelRegistry unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.registry = ModelRegistry()
    
    def test_register_model_with_class(self):
        """Test registering model with class."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        self.registry.register_model(
            name="test_model",
            model_class=TestModel,
            version="1.0.0",
            description="Test model with class"
        )
        
        # Check registration
        self.assertIn("test_model", self.registry.models)
        model_info = self.registry.get_model_info("test_model")
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info.name, "test_model")
        self.assertEqual(model_info.model_class, TestModel)
    
    def test_register_model_with_factory(self):
        """Test registering model with factory function."""
        def create_model():
            return nn.Linear(10, 1)
        
        self.registry.register_model(
            name="factory_model",
            factory_function=create_model,
            version="1.0.0",
            description="Test model with factory"
        )
        
        # Check registration
        self.assertIn("factory_model", self.registry.models)
        self.assertIn("factory_model", self.registry.factories)
        model_info = self.registry.get_model_info("factory_model")
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info.name, "factory_model")
        self.assertEqual(model_info.factory_function, create_model)
    
    def test_unregister_model(self):
        """Test unregistering model."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        self.registry.register_model(name="test_model", model_class=TestModel)
        self.assertIn("test_model", self.registry.models)
        
        self.registry.unregister_model("test_model")
        self.assertNotIn("test_model", self.registry.models)
        self.assertNotIn("test_model", self.registry.factories)
    
    def test_get_model_info(self):
        """Test getting model info."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        self.registry.register_model(
            name="test_model",
            model_class=TestModel,
            description="Test model"
        )
        
        # Get existing model info
        model_info = self.registry.get_model_info("test_model")
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info.name, "test_model")
        self.assertEqual(model_info.description, "Test model")
        
        # Get non-existent model info
        model_info = self.registry.get_model_info("non_existent")
        self.assertIsNone(model_info)
    
    def test_list_models(self):
        """Test listing models."""
        class TestModel1(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        class TestModel2(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        self.registry.register_model(name="model1", model_class=TestModel1)
        self.registry.register_model(name="model2", model_class=TestModel2)
        
        models = self.registry.list_models()
        self.assertIn("model1", models)
        self.assertIn("model2", models)
        self.assertEqual(len(models), 2)
    
    def test_list_models_by_category(self):
        """Test listing models by category."""
        class TestModel1(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        class TestModel2(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        self.registry.register_model(name="model1", model_class=TestModel1, category="classification")
        self.registry.register_model(name="model2", model_class=TestModel2, category="regression")
        
        classification_models = self.registry.list_models_by_category("classification")
        regression_models = self.registry.list_models_by_category("regression")
        
        self.assertIn("model1", classification_models)
        self.assertNotIn("model2", classification_models)
        self.assertIn("model2", regression_models)
        self.assertNotIn("model1", regression_models)
    
    def test_list_models_by_tag(self):
        """Test listing models by tag."""
        class TestModel1(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        class TestModel2(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        self.registry.register_model(name="model1", model_class=TestModel1, tags=["cnn", "image"])
        self.registry.register_model(name="model2", model_class=TestModel2, tags=["rnn", "text"])
        self.registry.register_model(name="model3", model_class=TestModel1, tags=["cnn", "audio"])
        
        cnn_models = self.registry.list_models_by_tag("cnn")
        rnn_models = self.registry.list_models_by_tag("rnn")
        
        self.assertIn("model1", cnn_models)
        self.assertIn("model3", cnn_models)
        self.assertNotIn("model2", cnn_models)
        self.assertIn("model2", rnn_models)
        self.assertNotIn("model1", rnn_models)
        self.assertNotIn("model3", rnn_models)
    
    def test_create_model_with_class(self):
        """Test creating model with class."""
        class TestModel(nn.Module):
            def __init__(self, input_size=10):
                super().__init__()
                self.input_size = input_size
                self.linear = nn.Linear(input_size, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        self.registry.register_model(name="test_model", model_class=TestModel)
        
        # Create model
        model = self.registry.create_model("test_model", input_size=20)
        self.assertIsInstance(model, TestModel)
        self.assertEqual(model.input_size, 20)
    
    def test_create_model_with_factory(self):
        """Test creating model with factory function."""
        def create_model(input_size=10):
            return nn.Linear(input_size, 1)
        
        self.registry.register_model(name="factory_model", factory_function=create_model)
        
        # Create model
        model = self.registry.create_model("factory_model", 15)
        self.assertIsInstance(model, nn.Linear)
        self.assertEqual(model.in_features, 15)
    
    def test_create_model_with_config(self):
        """Test creating model with config."""
        class TestModel(nn.Module):
            def __init__(self, config=None):
                super().__init__()
                if config:
                    self.input_size = config.input_size
                    self.output_size = config.output_size
                else:
                    self.input_size = 10
                    self.output_size = 1
                self.linear = nn.Linear(self.input_size, self.output_size)
            
            def forward(self, x):
                return self.linear(x)
        
        class TestConfig:
            def __init__(self, input_size=10, output_size=1):
                self.input_size = input_size
                self.output_size = output_size
        
        self.registry.register_model(name="config_model", model_class=TestModel)
        
        # Create model with config
        config = TestConfig(input_size=20, output_size=5)
        model = self.registry.create_model_with_config("config_model", config)
        self.assertIsInstance(model, TestModel)
        self.assertEqual(model.input_size, 20)
        self.assertEqual(model.output_size, 5)
    
    def test_create_model_not_found(self):
        """Test creating non-existent model."""
        with self.assertRaises(ValueError):
            self.registry.create_model("non_existent_model")
    
    def test_create_model_no_creation_method(self):
        """Test creating model without creation method."""
        self.registry.register_model(name="incomplete_model", description="No class or factory")
        
        with self.assertRaises(ValueError):
            self.registry.create_model("incomplete_model")
    
    def test_get_model_categories(self):
        """Test getting model categories."""
        class TestModel1(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        class TestModel2(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        self.registry.register_model(name="model1", model_class=TestModel1, category="classification")
        self.registry.register_model(name="model2", model_class=TestModel2, category="regression")
        self.registry.register_model(name="model3", model_class=TestModel1, category="classification")
        
        categories = self.registry.get_model_categories()
        self.assertIn("classification", categories)
        self.assertIn("regression", categories)
        self.assertEqual(len(categories), 2)
    
    def test_get_model_tags(self):
        """Test getting model tags."""
        class TestModel1(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        class TestModel2(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        self.registry.register_model(name="model1", model_class=TestModel1, tags=["cnn", "image"])
        self.registry.register_model(name="model2", model_class=TestModel2, tags=["rnn", "text"])
        self.registry.register_model(name="model3", model_class=TestModel1, tags=["cnn", "audio"])
        
        tags = self.registry.get_model_tags()
        self.assertIn("cnn", tags)
        self.assertIn("image", tags)
        self.assertIn("rnn", tags)
        self.assertIn("text", tags)
        self.assertIn("audio", tags)
        self.assertEqual(len(tags), 5)


class TestGlobalFunctions(unittest.TestCase):
    """Test global model registry functions."""
    
    def setUp(self):
        """Test setup - clear global registry."""
        # Clear global registry for testing
        global_registry = get_model_registry()
        global_registry.models.clear()
        global_registry.factories.clear()
    
    def test_register_model_global(self):
        """Test global register_model function."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        register_model(name="global_test_model", model_class=TestModel)
        
        model_names = list_models()
        self.assertIn("global_test_model", model_names)
        
        model_info = get_model_info("global_test_model")
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info.name, "global_test_model")
    
    def test_unregister_model_global(self):
        """Test global unregister_model function."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        register_model(name="global_test_model", model_class=TestModel)
        self.assertIn("global_test_model", list_models())
        
        unregister_model("global_test_model")
        self.assertNotIn("global_test_model", list_models())
    
    def test_create_model_global(self):
        """Test global create_model function."""
        class TestModel(nn.Module):
            def __init__(self, input_size=10):
                super().__init__()
                self.input_size = input_size
                self.linear = nn.Linear(input_size, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        register_model(name="global_test_model", model_class=TestModel)
        
        model = create_model("global_test_model", input_size=15)
        self.assertIsInstance(model, TestModel)
        self.assertEqual(model.input_size, 15)
    
    def test_create_model_with_config_global(self):
        """Test global create_model_with_config function."""
        class TestModel(nn.Module):
            def __init__(self, config=None):
                super().__init__()
                if config:
                    self.input_size = config.input_size
                else:
                    self.input_size = 10
                self.linear = nn.Linear(self.input_size, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        class TestConfig:
            def __init__(self, input_size=10):
                self.input_size = input_size
        
        register_model(name="global_config_model", model_class=TestModel)
        
        config = TestConfig(input_size=20)
        model = create_model_with_config("global_config_model", config)
        self.assertIsInstance(model, TestModel)
        self.assertEqual(model.input_size, 20)
    
    def test_get_model_registry_global(self):
        """Test global get_model_registry function."""
        registry = get_model_registry()
        self.assertIsInstance(registry, ModelRegistry)
        
        # Test that it's the same instance
        registry2 = get_model_registry()
        self.assertIs(registry, registry2)


if __name__ == '__main__':
    unittest.main()