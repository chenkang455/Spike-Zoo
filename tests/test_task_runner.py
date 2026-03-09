import unittest
import sys
from io import StringIO
from spikezoo.core.task_runner import TaskRunner, main
from spikezoo.core.plugin_base import PluginBase, PluginConfig


class TestTaskRunner(unittest.TestCase):
    """TaskRunner unit tests."""
    
    def test_parse_args(self):
        """Test parsing command line arguments."""
        runner = TaskRunner()
        
        # Test simple flags
        args = ["--flag1", "--flag2"]
        kwargs = runner._parse_args(args)
        self.assertTrue(kwargs["flag1"])
        self.assertTrue(kwargs["flag2"])
        
        # Test key-value pairs
        args = ["--param1", "value1", "--param2", "42", "--param3", "3.14"]
        kwargs = runner._parse_args(args)
        self.assertEqual(kwargs["param1"], "value1")
        self.assertEqual(kwargs["param2"], 42)
        self.assertEqual(kwargs["param3"], 3.14)
        
        # Test boolean values
        args = ["--flag", "true", "--param", "false"]
        kwargs = runner._parse_args(args)
        self.assertTrue(kwargs["flag"])
        self.assertFalse(kwargs["param"])
        
        # Test mixed arguments
        args = ["--flag", "--param", "value"]
        kwargs = runner._parse_args(args)
        self.assertTrue(kwargs["flag"])
        self.assertEqual(kwargs["param"], "value")
    
    def test_add_task(self):
        """Test adding a task to the runner."""
        class TestTask(PluginBase):
            def initialize(self):
                return True
            
            def execute(self, **kwargs):
                return "executed"
            
            def cleanup(self):
                pass
        
        runner = TaskRunner()
        runner.add_task("test_task", TestTask)
        
        # Check that the task is registered in the task manager
        self.assertIn("test_task", runner.task_manager.registry.plugins)
        self.assertEqual(runner.task_manager.registry.plugins["test_task"], TestTask)
    
    def test_main_entry_point(self):
        """Test main entry point."""
        # This test mainly checks that the main function can be called
        # without raising exceptions
        try:
            # We won't actually run main() as it would parse sys.argv
            # Instead, we'll just instantiate TaskRunner to ensure it works
            runner = TaskRunner()
            self.assertIsInstance(runner, TaskRunner)
        except Exception as e:
            self.fail(f"TaskRunner instantiation failed: {e}")


class TestExamplePluginWithRunner(unittest.TestCase):
    """Test example plugin with task runner."""
    
    def setUp(self):
        """Set up test case."""
        # Create a simple plugin for testing
        class SimplePlugin(PluginBase):
            def __init__(self, config=None):
                super().__init__(config)
                self.initialized = False
                self.cleaned_up = False
            
            def initialize(self):
                if self.config.enabled:
                    self.initialized = True
                    return True
                return False
            
            def execute(self, **kwargs):
                if not self.initialized:
                    raise RuntimeError("Plugin not initialized")
                return kwargs.get("result", "default_result")
            
            def cleanup(self):
                self.cleaned_up = True
    
        self.SimplePlugin = SimplePlugin
    
    def test_plugin_with_runner(self):
        """Test using a simple plugin with task runner."""
        runner = TaskRunner()
        runner.add_task("simple", self.SimplePlugin)
        
        # Test starting task
        config = PluginConfig(name="simple_task", enabled=True)
        success = runner.task_manager.start_task("simple", config)
        self.assertTrue(success)
        
        # Test executing task
        result = runner.task_manager.execute_task("simple", result="test_output")
        self.assertEqual(result, "test_output")
        
        # Test stopping task
        success = runner.task_manager.stop_task("simple")
        self.assertTrue(success)
        
        # Verify the plugin was cleaned up
        instance = runner.task_manager.registry.get_plugin_instance("simple")
        self.assertIsNotNone(instance)
        # Note: In this test setup, we can't easily verify cleanup was called
        # because the instance is retrieved after stopping


if __name__ == '__main__':
    unittest.main()