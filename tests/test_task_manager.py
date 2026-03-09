import unittest
import tempfile
import os
from pathlib import Path
from spikezoo.core.task_manager import TaskManager, TaskRegistry
from spikezoo.core.plugin_base import PluginBase, PluginConfig


class TestTaskRegistry(unittest.TestCase):
    """TaskRegistry unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.registry = TaskRegistry()
    
    def test_register_plugin(self):
        """Test registering a plugin."""
        class TestPlugin(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.registry.register_plugin("test_plugin", TestPlugin)
        self.assertIn("test_plugin", self.registry.plugins)
        self.assertEqual(self.registry.plugins["test_plugin"], TestPlugin)
    
    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        class TestPlugin(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.registry.register_plugin("test_plugin", TestPlugin)
        self.registry.unregister_plugin("test_plugin")
        self.assertNotIn("test_plugin", self.registry.plugins)
    
    def test_get_plugin_class(self):
        """Test getting a plugin class."""
        class TestPlugin(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.registry.register_plugin("test_plugin", TestPlugin)
        plugin_class = self.registry.get_plugin_class("test_plugin")
        self.assertEqual(plugin_class, TestPlugin)
        
        # Test getting non-existent plugin
        plugin_class = self.registry.get_plugin_class("non_existent")
        self.assertIsNone(plugin_class)
    
    def test_create_plugin_instance(self):
        """Test creating a plugin instance."""
        class TestPlugin(PluginBase):
            def __init__(self, config=None):
                super().__init__(config)
                self.initialized = False
            
            def initialize(self):
                self.initialized = True
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.registry.register_plugin("test_plugin", TestPlugin)
        instance = self.registry.create_plugin_instance("test_plugin")
        
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, TestPlugin)
        self.assertIn("test_plugin", self.registry.instances)
    
    def test_create_plugin_instance_failure(self):
        """Test creating a plugin instance with failure."""
        class FailingPlugin(PluginBase):
            def __init__(self, config=None):
                super().__init__(config)
                raise RuntimeError("Intentional failure")
            
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.registry.register_plugin("failing_plugin", FailingPlugin)
        instance = self.registry.create_plugin_instance("failing_plugin")
        self.assertIsNone(instance)
    
    def test_get_plugin_instance(self):
        """Test getting a plugin instance."""
        class TestPlugin(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.registry.register_plugin("test_plugin", TestPlugin)
        instance1 = self.registry.create_plugin_instance("test_plugin")
        instance2 = self.registry.get_plugin_instance("test_plugin")
        
        self.assertIs(instance1, instance2)
        
        # Test getting non-existent instance
        instance3 = self.registry.get_plugin_instance("non_existent")
        self.assertIsNone(instance3)
    
    def test_list_plugins(self):
        """Test listing plugins."""
        class TestPlugin1(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        class TestPlugin2(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.registry.register_plugin("plugin1", TestPlugin1)
        self.registry.register_plugin("plugin2", TestPlugin2)
        
        plugins = self.registry.list_plugins()
        self.assertIn("plugin1", plugins)
        self.assertIn("plugin2", plugins)
        self.assertEqual(len(plugins), 2)
    
    def test_load_plugin_from_file(self):
        """Test loading plugin from file."""
        # Create a temporary plugin file
        temp_dir = tempfile.mkdtemp()
        plugin_file = Path(temp_dir) / "temp_plugin.py"
        
        plugin_content = '''
from spikezoo.core.plugin_base import PluginBase

class TempPlugin(PluginBase):
    def initialize(self):
        return True
    
    def execute(self, **kwargs):
        return "executed from file"
    
    def cleanup(self):
        pass
'''
        
        with open(plugin_file, 'w') as f:
            f.write(plugin_content)
        
        # Load plugin from file
        success = self.registry.load_plugin_from_file(str(plugin_file), "TempPlugin")
        self.assertTrue(success)
        self.assertIn("TempPlugin", self.registry.plugins)
        
        # Cleanup
        os.remove(plugin_file)
        os.rmdir(temp_dir)


class TestTaskManager(unittest.TestCase):
    """TaskManager unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.task_manager = TaskManager()
    
    def test_register_task(self):
        """Test registering a task."""
        class TestTask(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.task_manager.register_task("test_task", TestTask)
        # The task should be registered in the underlying registry
        self.assertIn("test_task", self.task_manager.registry.plugins)
    
    def test_start_task(self):
        """Test starting a task."""
        class TestTask(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.task_manager.register_task("test_task", TestTask)
        success = self.task_manager.start_task("test_task")
        self.assertTrue(success)
        self.assertIn("test_task", self.task_manager.active_tasks)
    
    def test_start_task_failure(self):
        """Test starting a task that fails to initialize."""
        class FailingTask(PluginBase):
            def initialize(self):
                return False  # Simulate initialization failure
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.task_manager.register_task("failing_task", FailingTask)
        success = self.task_manager.start_task("failing_task")
        self.assertFalse(success)
        self.assertNotIn("failing_task", self.task_manager.active_tasks)
    
    def test_stop_task(self):
        """Test stopping a task."""
        class TestTask(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.task_manager.register_task("test_task", TestTask)
        self.task_manager.start_task("test_task")
        
        success = self.task_manager.stop_task("test_task")
        self.assertTrue(success)
        self.assertNotIn("test_task", self.task_manager.active_tasks)
    
    def test_stop_inactive_task(self):
        """Test stopping an inactive task."""
        success = self.task_manager.stop_task("inactive_task")
        self.assertFalse(success)
    
    def test_execute_task(self):
        """Test executing a task."""
        class TestTask(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                message = kwargs.get("message", "default")
                return f"executed: {message}"
            
            def cleanup(self):
                pass
        
        self.task_manager.register_task("test_task", TestTask)
        self.task_manager.start_task("test_task")
        
        result = self.task_manager.execute_task("test_task", message="hello")
        self.assertEqual(result, "executed: hello")
    
    def test_execute_inactive_task(self):
        """Test executing an inactive task."""
        result = self.task_manager.execute_task("inactive_task")
        self.assertIsNone(result)
    
    def test_list_tasks(self):
        """Test listing tasks."""
        class ActiveTask(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        class InactiveTask(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        self.task_manager.register_task("active_task", ActiveTask)
        self.task_manager.register_task("inactive_task", InactiveTask)
        self.task_manager.start_task("active_task")
        
        tasks = self.task_manager.list_tasks()
        self.assertEqual(tasks["active_task"], "active")
        self.assertEqual(tasks["inactive_task"], "inactive")
    
    def test_get_task_info(self):
        """Test getting task information."""
        class TestTask(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        config = PluginConfig(
            name="info_task",
            version="1.0.0",
            description="Task for info test",
            author="Info Author"
        )
        
        self.task_manager.register_task("info_task", TestTask)
        info = self.task_manager.get_task_info("info_task")
        # Since the task is not instantiated, info should be None
        self.assertIsNone(info)
        
        # After creating an instance, we should get info
        self.task_manager.registry.create_plugin_instance("info_task", config)
        info = self.task_manager.get_task_info("info_task")
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "info_task")


if __name__ == '__main__':
    unittest.main()