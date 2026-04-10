import unittest
from spikezoo.core.plugin_base import PluginBase, PluginConfig


class TestPluginBase(unittest.TestCase):
    """PluginBase unit tests."""
    
    def test_plugin_config_creation(self):
        """Test PluginConfig creation."""
        config = PluginConfig(
            name="test_plugin",
            version="1.0.0",
            description="A test plugin",
            author="Test Author",
            enabled=True,
            config={"param1": "value1"}
        )
        
        self.assertEqual(config.name, "test_plugin")
        self.assertEqual(config.version, "1.0.0")
        self.assertEqual(config.description, "A test plugin")
        self.assertEqual(config.author, "Test Author")
        self.assertTrue(config.enabled)
        self.assertEqual(config.config, {"param1": "value1"})
    
    def test_plugin_config_default_values(self):
        """Test PluginConfig default values."""
        config = PluginConfig()
        
        self.assertEqual(config.name, "")
        self.assertEqual(config.version, "1.0.0")
        self.assertEqual(config.description, "")
        self.assertEqual(config.author, "")
        self.assertTrue(config.enabled)
        self.assertEqual(config.config, {})
    
    def test_abstract_methods(self):
        """Test that PluginBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            PluginBase(PluginConfig())
    
    def test_plugin_base_subclass(self):
        """Test creating a concrete subclass of PluginBase."""
        class ConcretePlugin(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        config = PluginConfig(name="concrete_plugin")
        plugin = ConcretePlugin(config)
        
        self.assertEqual(plugin.name, "concrete_plugin")
        self.assertEqual(plugin.config, config)
        self.assertTrue(plugin.initialize())
        self.assertEqual(plugin.execute(), "executed")
        # cleanup should not raise exception
        plugin.cleanup()
    
    def test_plugin_base_get_info(self):
        """Test PluginBase get_info method."""
        class ConcretePlugin(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        config = PluginConfig(
            name="info_plugin",
            version="2.0.0",
            description="Plugin for info test",
            author="Info Author",
            enabled=False
        )
        
        plugin = ConcretePlugin(config)
        info = plugin.get_info()
        
        self.assertEqual(info["name"], "info_plugin")
        self.assertEqual(info["version"], "2.0.0")
        self.assertEqual(info["description"], "Plugin for info test")
        self.assertEqual(info["author"], "Info Author")
        self.assertFalse(info["enabled"])


if __name__ == '__main__':
    unittest.main()