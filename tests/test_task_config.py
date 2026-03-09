import unittest
import tempfile
import os
from pathlib import Path
from spikezoo.config.task_config import (
    TaskConfig, 
    MultiTaskConfig, 
    TaskConfigManager,
    load_task_config,
    save_task_config
)


class TestTaskConfig(unittest.TestCase):
    """TaskConfig unit tests."""
    
    def test_task_config_creation(self):
        """Test TaskConfig creation."""
        config = TaskConfig(
            task_id="test_task",
            task_name="Test Task",
            task_description="A test task",
            enabled=True,
            priority=1,
            parameters={"param1": "value1", "param2": 42},
            dependencies=["dep1", "dep2"],
            output_dir="./output"
        )
        
        self.assertEqual(config.task_id, "test_task")
        self.assertEqual(config.task_name, "Test Task")
        self.assertEqual(config.task_description, "A test task")
        self.assertTrue(config.enabled)
        self.assertEqual(config.priority, 1)
        self.assertEqual(config.parameters, {"param1": "value1", "param2": 42})
        self.assertEqual(config.dependencies, ["dep1", "dep2"])
        self.assertEqual(config.output_dir, "./output")
    
    def test_task_config_default_values(self):
        """Test TaskConfig default values."""
        config = TaskConfig()
        
        self.assertEqual(config.task_id, "")
        self.assertEqual(config.task_name, "")
        self.assertEqual(config.task_description, "")
        self.assertTrue(config.enabled)
        self.assertEqual(config.priority, 0)
        self.assertEqual(config.parameters, {})
        self.assertEqual(config.dependencies, [])
        self.assertEqual(config.output_dir, "")
    
    def test_task_config_from_dict(self):
        """Test creating TaskConfig from dictionary."""
        config_dict = {
            "task_id": "dict_task",
            "task_name": "Dict Task",
            "task_description": "Task from dictionary",
            "enabled": False,
            "priority": 5,
            "parameters": {"key": "value"},
            "dependencies": ["task1"],
            "output_dir": "/tmp/output"
        }
        
        config = TaskConfig.from_dict(config_dict)
        
        self.assertEqual(config.task_id, "dict_task")
        self.assertEqual(config.task_name, "Dict Task")
        self.assertEqual(config.task_description, "Task from dictionary")
        self.assertFalse(config.enabled)
        self.assertEqual(config.priority, 5)
        self.assertEqual(config.parameters, {"key": "value"})
        self.assertEqual(config.dependencies, ["task1"])
        self.assertEqual(config.output_dir, "/tmp/output")
    
    def test_task_config_to_dict(self):
        """Test converting TaskConfig to dictionary."""
        config = TaskConfig(
            task_id="to_dict_task",
            task_name="To Dict Task",
            task_description="Task to dictionary",
            enabled=True,
            priority=3,
            parameters={"test": "data"},
            dependencies=["dep1"],
            output_dir="./results"
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict["task_id"], "to_dict_task")
        self.assertEqual(config_dict["task_name"], "To Dict Task")
        self.assertEqual(config_dict["task_description"], "Task to dictionary")
        self.assertTrue(config_dict["enabled"])
        self.assertEqual(config_dict["priority"], 3)
        self.assertEqual(config_dict["parameters"], {"test": "data"})
        self.assertEqual(config_dict["dependencies"], ["dep1"])
        self.assertEqual(config_dict["output_dir"], "./results")


class TestMultiTaskConfig(unittest.TestCase):
    """MultiTaskConfig unit tests."""
    
    def test_multitask_config_creation(self):
        """Test MultiTaskConfig creation."""
        config = MultiTaskConfig(
            project_name="Test Project",
            project_version="2.0.0",
            default_task_settings={"timeout": 3600},
            global_parameters={"workers": 4},
            tasks={
                "task1": TaskConfig(task_id="task1", task_name="Task 1"),
                "task2": TaskConfig(task_id="task2", task_name="Task 2")
            }
        )
        
        self.assertEqual(config.project_name, "Test Project")
        self.assertEqual(config.project_version, "2.0.0")
        self.assertEqual(config.default_task_settings, {"timeout": 3600})
        self.assertEqual(config.global_parameters, {"workers": 4})
        self.assertEqual(len(config.tasks), 2)
        self.assertIn("task1", config.tasks)
        self.assertIn("task2", config.tasks)
    
    def test_multitask_config_default_values(self):
        """Test MultiTaskConfig default values."""
        config = MultiTaskConfig()
        
        self.assertEqual(config.project_name, "")
        self.assertEqual(config.project_version, "1.0.0")
        self.assertEqual(config.default_task_settings, {})
        self.assertEqual(config.global_parameters, {})
        self.assertEqual(config.tasks, {})
    
    def test_multitask_config_from_dict(self):
        """Test creating MultiTaskConfig from dictionary."""
        config_dict = {
            "project_name": "Dict Project",
            "project_version": "1.5.0",
            "default_task_settings": {"log_level": "DEBUG"},
            "global_parameters": {"gpu": True},
            "tasks": {
                "task1": {
                    "task_id": "task1",
                    "task_name": "Task 1 from dict",
                    "enabled": True,
                    "priority": 1
                }
            }
        }
        
        config = MultiTaskConfig.from_dict(config_dict)
        
        self.assertEqual(config.project_name, "Dict Project")
        self.assertEqual(config.project_version, "1.5.0")
        self.assertEqual(config.default_task_settings, {"log_level": "DEBUG"})
        self.assertEqual(config.global_parameters, {"gpu": True})
        self.assertEqual(len(config.tasks), 1)
        self.assertIn("task1", config.tasks)
        self.assertEqual(config.tasks["task1"].task_name, "Task 1 from dict")
    
    def test_multitask_config_to_dict(self):
        """Test converting MultiTaskConfig to dictionary."""
        task_config = TaskConfig(
            task_id="export_task",
            task_name="Export Task",
            enabled=False,
            priority=10
        )
        
        config = MultiTaskConfig(
            project_name="Export Project",
            project_version="3.0.0",
            tasks={"export_task": task_config}
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict["project_name"], "Export Project")
        self.assertEqual(config_dict["project_version"], "3.0.0")
        self.assertIn("export_task", config_dict["tasks"])
        task_dict = config_dict["tasks"]["export_task"]
        self.assertEqual(task_dict["task_id"], "export_task")
        self.assertEqual(task_dict["task_name"], "Export Task")
        self.assertFalse(task_dict["enabled"])
        self.assertEqual(task_dict["priority"], 10)
    
    def test_add_task(self):
        """Test adding task to MultiTaskConfig."""
        config = MultiTaskConfig()
        task = TaskConfig(task_id="added_task", task_name="Added Task")
        
        config.add_task(task)
        self.assertIn("added_task", config.tasks)
        self.assertEqual(config.tasks["added_task"], task)
    
    def test_get_task(self):
        """Test getting task from MultiTaskConfig."""
        task = TaskConfig(task_id="get_task", task_name="Get Task")
        config = MultiTaskConfig(tasks={"get_task": task})
        
        retrieved_task = config.get_task("get_task")
        self.assertEqual(retrieved_task, task)
        
        # Test getting non-existent task
        retrieved_task = config.get_task("non_existent")
        self.assertIsNone(retrieved_task)
    
    def test_get_enabled_tasks(self):
        """Test getting enabled tasks from MultiTaskConfig."""
        task1 = TaskConfig(task_id="task1", task_name="Task 1", enabled=True, priority=2)
        task2 = TaskConfig(task_id="task2", task_name="Task 2", enabled=False, priority=1)
        task3 = TaskConfig(task_id="task3", task_name="Task 3", enabled=True, priority=3)
        task4 = TaskConfig(task_id="task4", task_name="Task 4", enabled=True, priority=1)
        
        config = MultiTaskConfig(tasks={
            "task1": task1,
            "task2": task2,
            "task3": task3,
            "task4": task4
        })
        
        enabled_tasks = config.get_enabled_tasks()
        # Should get 3 enabled tasks, sorted by priority (lowest number first)
        self.assertEqual(len(enabled_tasks), 3)
        self.assertEqual(enabled_tasks[0].task_id, "task4")  # priority 1
        self.assertEqual(enabled_tasks[1].task_id, "task1")  # priority 2
        self.assertEqual(enabled_tasks[2].task_id, "task3")  # priority 3
    
    def test_merge_with_defaults(self):
        """Test merging task configurations with defaults."""
        config = MultiTaskConfig(
            default_task_settings={"timeout": 3600, "retries": 3},
            tasks={
                "task1": TaskConfig(
                    task_id="task1",
                    parameters={"timeout": 7200, "batch_size": 32}  # Override timeout
                ),
                "task2": TaskConfig(
                    task_id="task2",
                    parameters={"learning_rate": 0.001}  # No overlap with defaults
                )
            }
        )
        
        config.merge_with_defaults()
        
        # Task1 should have default retries but overridden timeout
        task1_params = config.tasks["task1"].parameters
        self.assertEqual(task1_params["timeout"], 7200)  # Overridden
        self.assertEqual(task1_params["retries"], 3)    # From defaults
        self.assertEqual(task1_params["batch_size"], 32)  # Preserved
        
        # Task2 should have all default values plus its own
        task2_params = config.tasks["task2"].parameters
        self.assertEqual(task2_params["timeout"], 3600)  # From defaults
        self.assertEqual(task2_params["retries"], 3)      # From defaults
        self.assertEqual(task2_params["learning_rate"], 0.001)  # Preserved


class TestTaskConfigManager(unittest.TestCase):
    """TaskConfigManager unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test cleanup."""
        # Remove temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)
    
    def test_load_config_yaml(self):
        """Test loading configuration from YAML file."""
        config_content = """
project_name: "YAML Test Project"
project_version: "1.0.0"
tasks:
  task1:
    task_id: "task1"
    task_name: "YAML Task"
    enabled: true
"""
        
        config_file = Path(self.temp_dir) / "test_config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        config = TaskConfigManager.load_config(config_file)
        self.assertEqual(config.project_name, "YAML Test Project")
        self.assertEqual(config.project_version, "1.0.0")
        self.assertIn("task1", config.tasks)
        self.assertEqual(config.tasks["task1"].task_name, "YAML Task")
    
    def test_load_config_json(self):
        """Test loading configuration from JSON file."""
        config_content = """
{
    "project_name": "JSON Test Project",
    "project_version": "1.0.0",
    "tasks": {
        "task1": {
            "task_id": "task1",
            "task_name": "JSON Task",
            "enabled": true
        }
    }
}
"""
        
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        config = TaskConfigManager.load_config(config_file)
        self.assertEqual(config.project_name, "JSON Test Project")
        self.assertEqual(config.project_version, "1.0.0")
        self.assertIn("task1", config.tasks)
        self.assertEqual(config.tasks["task1"].task_name, "JSON Task")
    
    def test_load_config_nonexistent(self):
        """Test loading configuration from non-existent file."""
        config_file = Path(self.temp_dir) / "nonexistent.yaml"
        
        with self.assertRaises(FileNotFoundError):
            TaskConfigManager.load_config(config_file)
    
    def test_save_config_yaml(self):
        """Test saving configuration to YAML file."""
        config = MultiTaskConfig(
            project_name="Save YAML Test",
            project_version="1.0.0",
            tasks={
                "task1": TaskConfig(task_id="task1", task_name="Save Task")
            }
        )
        
        config_file = Path(self.temp_dir) / "saved_config.yaml"
        TaskConfigManager.save_config(config, config_file)
        
        # Check file exists
        self.assertTrue(config_file.exists())
        
        # Load and verify
        loaded_config = TaskConfigManager.load_config(config_file)
        self.assertEqual(loaded_config.project_name, "Save YAML Test")
        self.assertEqual(loaded_config.project_version, "1.0.0")
        self.assertIn("task1", loaded_config.tasks)
    
    def test_save_config_json(self):
        """Test saving configuration to JSON file."""
        config = MultiTaskConfig(
            project_name="Save JSON Test",
            project_version="1.0.0",
            tasks={
                "task1": TaskConfig(task_id="task1", task_name="Save Task")
            }
        )
        
        config_file = Path(self.temp_dir) / "saved_config.json"
        TaskConfigManager.save_config(config, config_file, format='json')
        
        # Check file exists
        self.assertTrue(config_file.exists())
        
        # Load and verify
        loaded_config = TaskConfigManager.load_config(config_file)
        self.assertEqual(loaded_config.project_name, "Save JSON Test")
        self.assertEqual(loaded_config.project_version, "1.0.0")
        self.assertIn("task1", loaded_config.tasks)
    
    def test_save_config_invalid_format(self):
        """Test saving configuration with invalid format."""
        config = MultiTaskConfig()
        config_file = Path(self.temp_dir) / "invalid_config.xyz"
        
        with self.assertRaises(ValueError):
            TaskConfigManager.save_config(config, config_file, format='xyz')
    
    def test_convenience_functions(self):
        """Test convenience load_task_config and save_task_config functions."""
        config = MultiTaskConfig(
            project_name="Convenience Test",
            tasks={
                "task1": TaskConfig(task_id="task1", task_name="Convenience Task")
            }
        )
        
        config_file = Path(self.temp_dir) / "convenience_test.yaml"
        
        # Save using convenience function
        save_task_config(config, config_file)
        
        # Load using convenience function
        loaded_config = load_task_config(config_file)
        
        # Verify
        self.assertEqual(loaded_config.project_name, "Convenience Test")
        self.assertIn("task1", loaded_config.tasks)
        self.assertEqual(loaded_config.tasks["task1"].task_name, "Convenience Task")


if __name__ == '__main__':
    unittest.main()