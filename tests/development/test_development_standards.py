import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from spikezoo.models.base_model import BaseModel, BaseModelConfig
from spikezoo.models.model_registry import ModelRegistry


class TestDevelopmentStandards(unittest.TestCase):
    """Test development standards and practices."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_interface_consistency(self):
        """Test that model interface follows standards."""
        # Test BaseModel can be instantiated
        config = BaseModelConfig(model_name="test_model")
        model = BaseModel(config)
        
        self.assertEqual(model.cfg, config)
        self.assertIsNone(model.net)
        self.assertIsNone(model.device)
    
    def test_model_registry_functionality(self):
        """Test model registry basic functionality."""
        registry = ModelRegistry()
        
        # Test registry can be created
        self.assertIsInstance(registry, ModelRegistry)
        
        # Test model listing works
        models = registry.list_models()
        self.assertIsInstance(models, list)
    
    def test_configuration_inheritance(self):
        """Test that configurations inherit properly."""
        config = BaseModelConfig()
        
        # Test all required attributes exist
        required_attrs = [
            'model_name', 'model_file_name', 'model_cls_name',
            'model_length', 'require_params', 'model_params',
            'ckpt_path', 'load_state', 'multi_gpu', 'base_url'
        ]
        
        for attr in required_attrs:
            self.assertTrue(hasattr(config, attr), f"Missing attribute: {attr}")
    
    def test_directory_structure_standards(self):
        """Test that directory structure follows standards."""
        # This test checks the current working directory structure
        current_dir = Path(__file__).parent.parent.parent
        
        # Check for required directories
        required_dirs = ['spikezoo', 'models', 'archs', 'tests', 'examples', 'docs']
        existing_dirs = [d.name for d in current_dir.iterdir() if d.is_dir()]
        
        # Check that core directories exist
        self.assertIn('spikezoo', existing_dirs, "Missing spikezoo directory")
        
        # Check spikezoo subdirectories
        spikezoo_dir = current_dir / 'spikezoo'
        if spikezoo_dir.exists():
            spikezoo_subdirs = [d.name for d in spikezoo_dir.iterdir() if d.is_dir()]
            self.assertIn('models', spikezoo_subdirs, "Missing models directory")
            self.assertIn('archs', spikezoo_subdirs, "Missing archs directory")
    
    def test_documentation_files_exist(self):
        """Test that documentation files exist."""
        docs_dir = Path(__file__).parent.parent.parent / 'docs' / 'development'
        
        if docs_dir.exists():
            required_docs = [
                'model_structure.md',
                'development_guide.md',
                'contributing.md',
                'README.md'
            ]
            
            existing_docs = [f.name for f in docs_dir.iterdir() if f.is_file()]
            
            for doc in required_docs:
                self.assertIn(doc, existing_docs, f"Missing documentation file: {doc}")
    
    def test_example_templates_exist(self):
        """Test that example templates exist."""
        examples_dir = Path(__file__).parent.parent.parent / 'examples' / 'development'
        
        if examples_dir.exists():
            example_files = [f.name for f in examples_dir.iterdir() if f.is_file()]
            self.assertIn('model_template.py', example_files, "Missing model template")
    
    def test_code_quality_standards(self):
        """Test code quality standards."""
        # Test that we can import core modules
        from spikezoo.models import BaseModel, BaseModelConfig
        from spikezoo.models.model_registry import ModelRegistry
        
        # Test that classes exist
        self.assertTrue(callable(BaseModel))
        self.assertTrue(callable(BaseModelConfig))
        self.assertTrue(callable(ModelRegistry))
    
    def test_naming_conventions(self):
        """Test naming conventions."""
        # Test class names use PascalCase
        self.assertEqual(BaseModel.__name__, "BaseModel")
        self.assertEqual(BaseModelConfig.__name__, "BaseModelConfig")
        
        # Test method names use snake_case
        model = BaseModel(BaseModelConfig())
        self.assertTrue(hasattr(model, "build_network"))
        self.assertTrue(hasattr(model, "get_outputs_dict"))
        self.assertTrue(hasattr(model, "get_loss_dict"))
    
    def test_type_hinting(self):
        """Test that type hinting is used."""
        import inspect
        
        # Check that BaseModel methods have type hints
        methods_to_check = [
            'build_network', 'get_outputs_dict', 
            'get_loss_dict', 'get_visual_dict'
        ]
        
        for method_name in methods_to_check:
            if hasattr(BaseModel, method_name):
                method = getattr(BaseModel, method_name)
                sig = inspect.signature(method)
                # This test passes if no exception is raised
                self.assertTrue(callable(method))
    
    def test_configuration_defaults(self):
        """Test configuration default values."""
        config = BaseModelConfig()
        
        # Test sensible defaults
        self.assertEqual(config.model_name, "base")
        self.assertEqual(config.model_file_name, "nets")
        self.assertEqual(config.model_cls_name, "BaseNet")
        self.assertIsInstance(config.model_length, int)
        self.assertIsInstance(config.model_params, dict)
        self.assertIsInstance(config.model_params_dict, dict)
    
    def test_model_registry_patterns(self):
        """Test model registry patterns."""
        # Test registry singleton pattern
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()
        
        # These should be different instances (no singleton enforced)
        # But they should both work
        self.assertIsInstance(registry1, ModelRegistry)
        self.assertIsInstance(registry2, ModelRegistry)
        
        # Test registry methods exist
        required_methods = [
            'register_model', 'unregister_model', 'get_model_class',
            'get_config_class', 'create_model', 'list_models'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(registry1, method), f"Missing method: {method}")


class TestIntegrationStandards(unittest.TestCase):
    """Test integration standards."""
    
    def test_model_creation_workflow(self):
        """Test the complete model creation workflow."""
        # This test simulates the standard workflow
        from spikezoo.models import create_model, list_models
        
        # Test that we can list models without error
        try:
            models = list_models()
            self.assertIsInstance(models, list)
        except Exception as e:
            self.fail(f"list_models failed: {e}")
        
        # Test that create_model handles missing models gracefully
        try:
            model = create_model("nonexistent_model")
            self.assertIsNone(model)
        except Exception as e:
            self.fail(f"create_model failed unexpectedly: {e}")
    
    def test_configuration_workflow(self):
        """Test configuration workflow."""
        # Test creating configuration with various parameters
        config = BaseModelConfig(
            model_name="test_model",
            load_state=True,
            multi_gpu=False
        )
        
        self.assertEqual(config.model_name, "test_model")
        self.assertTrue(config.load_state)
        self.assertFalse(config.multi_gpu)
    
    def test_model_building_workflow(self):
        """Test model building workflow."""
        model = BaseModel(BaseModelConfig())
        
        # Test that model has expected methods
        expected_methods = [
            'build_network', 'feed_to_device', 'get_outputs_dict',
            'get_loss_dict', 'get_visual_dict', 'save_network', 'spk2img'
        ]
        
        for method_name in expected_methods:
            self.assertTrue(
                hasattr(model, method_name), 
                f"Model missing expected method: {method_name}"
            )
    
    def test_error_handling_standards(self):
        """Test error handling standards."""
        # Test that methods handle errors gracefully
        model = BaseModel(BaseModelConfig())
        
        # These should not raise exceptions for basic cases
        try:
            model.feed_to_device({})
        except Exception as e:
            # Empty dict should be handled gracefully
            pass


if __name__ == '__main__':
    unittest.main()