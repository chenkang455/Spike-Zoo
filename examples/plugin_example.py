#!/usr/bin/env python3
"""
Example of using the SpikeZoo task management framework.
"""

import sys
import os

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.core import TaskManager, TaskRunner
from spikezoo.core.plugin_base import PluginBase, PluginConfig
from spikezoo.plugins.example_plugin import ExamplePlugin


def example_task_usage():
    """Example of using tasks programmatically."""
    print("=== SpikeZoo Task Management Example ===\n")
    
    # Create task manager
    task_manager = TaskManager()
    
    # Register built-in example plugin
    task_manager.register_task("example", ExamplePlugin)
    
    # Create plugin configuration
    config = PluginConfig(
        name="example_task",
        version="1.0.0",
        description="An example task for demonstration",
        author="SpikeZoo Team",
        enabled=True,
        config={
            "param1": "value1",
            "param2": 42
        }
    )
    
    # Start task
    print("1. Starting example task...")
    if task_manager.start_task("example", config):
        print("✓ Task started successfully\n")
    else:
        print("✗ Failed to start task\n")
        return
    
    # Execute task multiple times
    print("2. Executing example task...")
    result1 = task_manager.execute_task("example", message="First execution! ", repeat=2)
    result2 = task_manager.execute_task("example", message="Second execution! ", repeat=3)
    print(f"   Results: '{result1}', '{result2}'\n")
    
    # List tasks
    print("3. Listing tasks...")
    tasks = task_manager.list_tasks()
    for task_name, status in tasks.items():
        print(f"   {task_name}: {status}")
        info = task_manager.get_task_info(task_name)
        if info:
            print(f"     Info: {info}")
    print()
    
    # Stop task
    print("4. Stopping example task...")
    if task_manager.stop_task("example"):
        print("✓ Task stopped successfully\n")
    else:
        print("✗ Failed to stop task\n")


def example_runner_usage():
    """Example of using task runner."""
    print("=== SpikeZoo Task Runner Example ===\n")
    
    # Create task runner
    runner = TaskRunner()
    
    # Add example task
    runner.add_task("example", ExamplePlugin)
    
    # Note: In a real scenario, you would use command line arguments
    # For this example, we'll simulate command line usage
    print("Simulating command line usage:")
    print("  python example_runner.py --list")
    print("  python example_runner.py --start example")
    print("  python example_runner.py example --message 'CLI test! ' --repeat 2")
    print("  python example_runner.py --stop example\n")


if __name__ == "__main__":
    example_task_usage()
    example_runner_usage()