import unittest
import sys
import os
import torch

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.core.component_registry import (
    ComponentRegistry,
    ComponentInfo,
    ComponentType,
    register_component,
    unregister_component,
    get_component_info,
    list_components,
    list_components_by_type,
    create_component,
    create_component_with_config,
    get_component_registry,
    discover_components_from_directory,
    check_component_dependencies
)


class TestComponentInfo(unittest.TestCase):
    """ComponentInfo unit tests."""
    
    def test_component_info_creation(self):
        """Test ComponentInfo creation."""
        component_info = ComponentInfo(
            name="test_component",
            component_type=ComponentType.MODEL,
            version="1.0.0",
            description="Test component",
            author="Test Author",
            category="test",
            tags=["tag1", "tag2"],
            dependencies=["dep1", "dep2"]
        )
        
        self.assertEqual(component_info.name, "test_component")
        self.assertEqual(component_info.component_type, ComponentType.MODEL)
        self.assertEqual(component_info.version, "1.0.0")
        self.assertEqual(component_info.description, "Test component")
        self.assertEqual(component_info.author, "Test Author")
        self.assertEqual(component_info.category, "test")
        self.assertEqual(component_info.tags, ["tag1", "tag2"])
        self.assertEqual(component_info.dependencies, ["dep1", "dep2"])
        self.assertIsNone(component_info.config_class)
        self.assertIsNone(component_info.component_class)
        self.assertIsNone(component_info.factory_function)


class TestComponentRegistry(unittest.TestCase):
    """ComponentRegistry unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.registry = ComponentRegistry()
    
    def test_register_component_with_class(self):
        """Test registering component with class."""
        class TestComponent:
            def __init__(self):
                self.name = "test"
            
            def process(self):
                return "processed"
        
        self.registry.register_component(
            name="test_component",
            component_type=ComponentType.PREPROCESSOR,
            component_class=TestComponent,
            version="1.0.0",
            description="Test component with class"
        )
        
        # Check registration
        self.assertIn("test_component", self.registry.components)
        component_info = self.registry.get_component_info("test_component")
        self.assertIsNotNone(component_info)
        self.assertEqual(component_info.name, "test_component")
        self.assertEqual(component_info.component_type, ComponentType.PREPROCESSOR)
        self.assertEqual(component_info.component_class, TestComponent)
    
    def test_register_component_with_factory(self):
        """Test registering component with factory function."""
        def create_component():
            class SimpleComponent:
                def __init__(self):
                    self.name = "simple"
                
                def process(self):
                    return "simple_processed"
            return SimpleComponent()
        
        self.registry.register_component(
            name="factory_component",
            component_type=ComponentType.POSTPROCESSOR,
            factory_function=create_component,
            version="1.0.0",
            description="Test component with factory"
        )
        
        # Check registration
        self.assertIn("factory_component", self.registry.components)
        self.assertIn("factory_component", self.registry.factories)
        component_info = self.registry.get_component_info("factory_component")
        self.assertIsNotNone(component_info)
        self.assertEqual(component_info.name, "factory_component")
        self.assertEqual(component_info.component_type, ComponentType.POSTPROCESSOR)
        self.assertEqual(component_info.factory_function, create_component)
    
    def test_unregister_component(self):
        """Test unregistering component."""
        class TestComponent:
            def __init__(self):
                self.name = "test"
            
            def process(self):
                return "processed"
        
        self.registry.register_component(name="test_component", component_type=ComponentType.MODEL, component_class=TestComponent)
        self.assertIn("test_component", self.registry.components)
        
        self.registry.unregister_component("test_component")
        self.assertNotIn("test_component", self.registry.components)
        self.assertNotIn("test_component", self.registry.factories)
    
    def test_get_component_info(self):
        """Test getting component info."""
        class TestComponent:
            def __init__(self):
                self.name = "test"
            
            def process(self):
                return "processed"
        
        self.registry.register_component(
            name="test_component",
            component_type=ComponentType.MODEL,
            component_class=TestComponent,
            description="Test component"
        )
        
        # Get existing component info
        component_info = self.registry.get_component_info("test_component")
        self.assertIsNotNone(component_info)
        self.assertEqual(component_info.name, "test_component")
        self.assertEqual(component_info.description, "Test component")
        
        # Get non-existent component info
        component_info = self.registry.get_component_info("non_existent")
        self.assertIsNone(component_info)
    
    def test_list_components(self):
        """Test listing components."""
        class TestComponent1:
            def __init__(self):
                self.name = "test1"
        
        class TestComponent2:
            def __init__(self):
                self.name = "test2"
        
        self.registry.register_component(name="component1", component_type=ComponentType.MODEL, component_class=TestComponent1)
        self.registry.register_component(name="component2", component_type=ComponentType.DATASET, component_class=TestComponent2)
        
        components = self.registry.list_components()
        self.assertIn("component1", components)
        self.assertIn("component2", components)
        self.assertEqual(len(components), 2)
    
    def test_list_components_by_type(self):
        """Test listing components by type."""
        class TestComponent1:
            def __init__(self):
                self.name = "test1"
        
        class TestComponent2:
            def __init__(self):
                self.name = "test2"
        
        self.registry.register_component(name="component1", component_type=ComponentType.MODEL, component_class=TestComponent1)
        self.registry.register_component(name="component2", component_type=ComponentType.DATASET, component_class=TestComponent2)
        
        model_components = self.registry.list_components_by_type(ComponentType.MODEL)
        dataset_components = self.registry.list_components_by_type(ComponentType.DATASET)
        
        self.assertIn("component1", model_components)
        self.assertNotIn("component2", model_components)
        self.assertIn("component2", dataset_components)
        self.assertNotIn("component1", dataset_components)
    
    def test_list_components_by_category(self):
        """Test listing components by category."""
        class TestComponent1:
            def __init__(self):
                self.name = "test1"
        
        class TestComponent2:
            def __init__(self):
                self.name = "test2"
        
        self.registry.register_component(name="component1", component_type=ComponentType.MODEL, component_class=TestComponent1, category="vision")
        self.registry.register_component(name="component2", component_type=ComponentType.MODEL, component_class=TestComponent2, category="nlp")
        
        vision_components = self.registry.list_components_by_category("vision")
        nlp_components = self.registry.list_components_by_category("nlp")
        
        self.assertIn("component1", vision_components)
        self.assertNotIn("component2", vision_components)
        self.assertIn("component2", nlp_components)
        self.assertNotIn("component1", nlp_components)
    
    def test_list_components_by_tag(self):
        """Test listing components by tag."""
        class TestComponent1:
            def __init__(self):
                self.name = "test1"
        
        class TestComponent2:
            def __init__(self):
                self.name = "test2"
        
        self.registry.register_component(name="component1", component_type=ComponentType.MODEL, component_class=TestComponent1, tags=["cnn", "image"])
        self.registry.register_component(name="component2", component_type=ComponentType.MODEL, component_class=TestComponent2, tags=["rnn", "text"])
        self.registry.register_component(name="component3", component_type=ComponentType.MODEL, component_class=TestComponent1, tags=["cnn", "audio"])
        
        cnn_components = self.registry.list_components_by_tag("cnn")
        rnn_components = self.registry.list_components_by_tag("rnn")
        
        self.assertIn("component1", cnn_components)
        self.assertIn("component3", cnn_components)
        self.assertNotIn("component2", cnn_components)
        self.assertIn("component2", rnn_components)
        self.assertNotIn("component1", rnn_components)
        self.assertNotIn("component3", rnn_components)
    
    def test_create_component_with_class(self):
        """Test creating component with class."""
        class TestComponent:
            def __init__(self, name="default"):
                self.name = name
            
            def process(self):
                return f"processed_{self.name}"
        
        self.registry.register_component(name="test_component", component_type=ComponentType.PREPROCESSOR, component_class=TestComponent)
        
        # Create component
        component = self.registry.create_component("test_component", name="custom")
        self.assertIsInstance(component, TestComponent)
        self.assertEqual(component.name, "custom")
        self.assertEqual(component.process(), "processed_custom")
    
    def test_create_component_with_factory(self):
        """Test creating component with factory function."""
        def create_component(name="default"):
            class SimpleComponent:
                def __init__(self, name):
                    self.name = name
                
                def process(self):
                    return f"simple_{self.name}"
            return SimpleComponent(name)
        
        self.registry.register_component(name="factory_component", component_type=ComponentType.POSTPROCESSOR, factory_function=create_component)
        
        # Create component
        component = self.registry.create_component("factory_component", "custom")
        self.assertEqual(component.process(), "simple_custom")
    
    def test_create_component_with_config(self):
        """Test creating component with config."""
        class TestComponent:
            def __init__(self, config=None):
                if config:
                    self.name = config.get('name', 'default')
                    self.value = config.get('value', 0)
                else:
                    self.name = 'default'
                    self.value = 0
            
            def process(self):
                return f"{self.name}_{self.value}"
        
        class TestConfig:
            def __init__(self, name="default", value=0):
                self.name = name
                self.value = value
        
        self.registry.register_component(name="config_component", component_type=ComponentType.METRIC, component_class=TestComponent)
        
        # Create component with config
        config = {'name': 'custom', 'value': 42}
        component = self.registry.create_component_with_config("config_component", config)
        self.assertIsInstance(component, TestComponent)
        self.assertEqual(component.name, "custom")
        self.assertEqual(component.value, 42)
        self.assertEqual(component.process(), "custom_42")
    
    def test_create_component_not_found(self):
        """Test creating non-existent component."""
        with self.assertRaises(ValueError):
            self.registry.create_component("non_existent_component")
    
    def test_create_component_no_creation_method(self):
        """Test creating component without creation method."""
        self.registry.register_component(name="incomplete_component", component_type=ComponentType.CUSTOM, description="No class or factory")
        
        with self.assertRaises(ValueError):
            self.registry.create_component("incomplete_component")
    
    def test_get_component_types(self):
        """Test getting component types."""
        class TestComponent1:
            def __init__(self):
                self.name = "test1"
        
        class TestComponent2:
            def __init__(self):
                self.name = "test2"
        
        self.registry.register_component(name="component1", component_type=ComponentType.MODEL, component_class=TestComponent1)
        self.registry.register_component(name="component2", component_type=ComponentType.DATASET, component_class=TestComponent2)
        self.registry.register_component(name="component3", component_type=ComponentType.MODEL, component_class=TestComponent1)
        
        types = self.registry.get_component_types()
        self.assertIn(ComponentType.MODEL, types)
        self.assertIn(ComponentType.DATASET, types)
        self.assertEqual(len(types), 2)
    
    def test_get_component_categories(self):
        """Test getting component categories."""
        class TestComponent1:
            def __init__(self):
                self.name = "test1"
        
        class TestComponent2:
            def __init__(self):
                self.name = "test2"
        
        self.registry.register_component(name="component1", component_type=ComponentType.MODEL, component_class=TestComponent1, category="vision")
        self.registry.register_component(name="component2", component_type=ComponentType.MODEL, component_class=TestComponent2, category="nlp")
        self.registry.register_component(name="component3", component_type=ComponentType.MODEL, component_class=TestComponent1, category="vision")
        
        categories = self.registry.get_component_categories()
        self.assertIn("vision", categories)
        self.assertIn("nlp", categories)
        self.assertEqual(len(categories), 2)
    
    def test_get_component_tags(self):
        """Test getting component tags."""
        class TestComponent1:
            def __init__(self):
                self.name = "test1"
        
        class TestComponent2:
            def __init__(self):
                self.name = "test2"
        
        self.registry.register_component(name="component1", component_type=ComponentType.MODEL, component_class=TestComponent1, tags=["cnn", "image"])
        self.registry.register_component(name="component2", component_type=ComponentType.MODEL, component_class=TestComponent2, tags=["rnn", "text"])
        self.registry.register_component(name="component3", component_type=ComponentType.MODEL, component_class=TestComponent1, tags=["cnn", "audio"])
        
        tags = self.registry.get_component_tags()
        self.assertIn("cnn", tags)
        self.assertIn("image", tags)
        self.assertIn("rnn", tags)
        self.assertIn("text", tags)
        self.assertIn("audio", tags)
        self.assertEqual(len(tags), 5)
    
    def test_get_component_dependencies(self):
        """Test getting component dependencies."""
        class TestComponent:
            def __init__(self):
                self.name = "test"
        
        self.registry.register_component(
            name="dependent_component",
            component_type=ComponentType.CUSTOM,
            component_class=TestComponent,
            dependencies=["base", "helper"]
        )
        
        dependencies = self.registry.get_component_dependencies("dependent_component")
        self.assertEqual(dependencies, ["base", "helper"])
        
        # Test non-existent component
        dependencies = self.registry.get_component_dependencies("non_existent")
        self.assertEqual(dependencies, [])
    
    def test_check_dependencies(self):
        """Test checking component dependencies."""
        # Register component with dependencies
        self.registry.register_component(
            name="dependent_component",
            component_type=ComponentType.CUSTOM,
            dependencies=["base_component", "helper_component"]
        )
        
        # Check dependencies before registering dependencies (should fail)
        satisfied = self.registry.check_dependencies("dependent_component")
        self.assertFalse(satisfied)
        
        # Register dependencies
        self.registry.register_component(name="base_component", component_type=ComponentType.CUSTOM)
        self.registry.register_component(name="helper_component", component_type=ComponentType.CUSTOM)
        
        # Check dependencies after registering dependencies (should pass)
        satisfied = self.registry.check_dependencies("dependent_component")
        self.assertTrue(satisfied)
        
        # Test non-existent component
        satisfied = self.registry.check_dependencies("non_existent")
        self.assertFalse(satisfied)


class TestGlobalFunctions(unittest.TestCase):
    """Test global component registry functions."""
    
    def setUp(self):
        """Test setup - clear global registry."""
        # Clear global registry for testing
        global_registry = get_component_registry()
        global_registry.components.clear()
        global_registry.factories.clear()
    
    def test_register_component_global(self):
        """Test global register_component function."""
        class TestComponent:
            def __init__(self):
                self.name = "test"
            
            def process(self):
                return "processed"
        
        register_component(name="global_test_component", component_type=ComponentType.MODEL, component_class=TestComponent)
        
        component_names = list_components()
        self.assertIn("global_test_component", component_names)
        
        component_info = get_component_info("global_test_component")
        self.assertIsNotNone(component_info)
        self.assertEqual(component_info.name, "global_test_component")
    
    def test_unregister_component_global(self):
        """Test global unregister_component function."""
        class TestComponent:
            def __init__(self):
                self.name = "test"
            
            def process(self):
                return "processed"
        
        register_component(name="global_test_component", component_type=ComponentType.MODEL, component_class=TestComponent)
        self.assertIn("global_test_component", list_components())
        
        unregister_component("global_test_component")
        self.assertNotIn("global_test_component", list_components())
    
    def test_list_components_by_type_global(self):
        """Test global list_components_by_type function."""
        class TestComponent1:
            def __init__(self):
                self.name = "test1"
        
        class TestComponent2:
            def __init__(self):
                self.name = "test2"
        
        register_component(name="global_component1", component_type=ComponentType.MODEL, component_class=TestComponent1)
        register_component(name="global_component2", component_type=ComponentType.DATASET, component_class=TestComponent2)
        
        model_components = list_components_by_type(ComponentType.MODEL)
        dataset_components = list_components_by_type(ComponentType.DATASET)
        
        self.assertIn("global_component1", model_components)
        self.assertNotIn("global_component2", model_components)
        self.assertIn("global_component2", dataset_components)
        self.assertNotIn("global_component1", dataset_components)
    
    def test_create_component_global(self):
        """Test global create_component function."""
        class TestComponent:
            def __init__(self, name="default"):
                self.name = name
            
            def process(self):
                return f"global_{self.name}"
        
        register_component(name="global_test_component", component_type=ComponentType.PREPROCESSOR, component_class=TestComponent)
        
        component = create_component("global_test_component", name="custom")
        self.assertIsInstance(component, TestComponent)
        self.assertEqual(component.name, "custom")
        self.assertEqual(component.process(), "global_custom")
    
    def test_create_component_with_config_global(self):
        """Test global create_component_with_config function."""
        class TestComponent:
            def __init__(self, config=None):
                if config:
                    self.name = config.get('name', 'default')
                else:
                    self.name = 'default'
            
            def process(self):
                return f"global_{self.name}"
        
        register_component(name="global_config_component", component_type=ComponentType.POSTPROCESSOR, component_class=TestComponent)
        
        config = {'name': 'custom'}
        component = create_component_with_config("global_config_component", config)
        self.assertIsInstance(component, TestComponent)
        self.assertEqual(component.name, "custom")
        self.assertEqual(component.process(), "global_custom")
    
    def test_get_component_registry_global(self):
        """Test global get_component_registry function."""
        registry = get_component_registry()
        self.assertIsInstance(registry, ComponentRegistry)
        
        # Test that it's the same instance
        registry2 = get_component_registry()
        self.assertIs(registry, registry2)
    
    def test_check_component_dependencies_global(self):
        """Test global check_component_dependencies function."""
        # Register component with dependencies
        register_component(
            name="global_dependent_component",
            component_type=ComponentType.CUSTOM,
            dependencies=["global_base_component"]
        )
        
        # Check dependencies before registering dependencies (should fail)
        satisfied = check_component_dependencies("global_dependent_component")
        self.assertFalse(satisfied)
        
        # Register dependency
        register_component(name="global_base_component", component_type=ComponentType.CUSTOM)
        
        # Check dependencies after registering dependencies (should pass)
        satisfied = check_component_dependencies("global_dependent_component")
        self.assertTrue(satisfied)


if __name__ == '__main__':
    unittest.main()