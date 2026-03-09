from typing import Dict, Any, Optional, List
import argparse
import sys
import logging
from .task_manager import TaskManager
from .plugin_base import PluginConfig


class TaskRunner:
    """Task runner for executing tasks from command line."""
    
    def __init__(self):
        """Initialize task runner."""
        self.task_manager = TaskManager()
        self.logger = logging.getLogger(__name__)
    
    def register_builtin_tasks(self) -> None:
        """Register built-in tasks."""
        # This is where built-in tasks would be registered
        # For now, we'll leave it empty as an example
        pass
    
    def run_from_command_line(self) -> None:
        """Run tasks from command line arguments."""
        parser = argparse.ArgumentParser(description="SpikeZoo Task Runner")
        parser.add_argument(
            "task", 
            nargs="?", 
            help="Task name to execute"
        )
        parser.add_argument(
            "--list", 
            action="store_true", 
            help="List all available tasks"
        )
        parser.add_argument(
            "--start", 
            metavar="TASK_NAME", 
            help="Start a task"
        )
        parser.add_argument(
            "--stop", 
            metavar="TASK_NAME", 
            help="Stop a task"
        )
        parser.add_argument(
            "--config", 
            metavar="CONFIG_FILE", 
            help="Configuration file for task"
        )
        parser.add_argument(
            "--verbose", 
            "-v", 
            action="store_true", 
            help="Enable verbose output"
        )
        
        args, unknown_args = parser.parse_known_args()
        
        # Setup logging
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        
        # Register built-in tasks
        self.register_builtin_tasks()
        
        # Handle commands
        if args.list:
            self._list_tasks()
            return
        
        if args.start:
            self._start_task(args.start, args.config)
            return
        
        if args.stop:
            self._stop_task(args.stop)
            return
        
        if args.task:
            self._execute_task(args.task, unknown_args)
            return
        
        # If no arguments, show help
        parser.print_help()
    
    def _list_tasks(self) -> None:
        """List all available tasks."""
        tasks = self.task_manager.list_tasks()
        if not tasks:
            print("No tasks available")
            return
        
        print("Available tasks:")
        for task_name, status in tasks.items():
            print(f"  {task_name} ({status})")
    
    def _start_task(self, task_name: str, config_file: Optional[str] = None) -> None:
        """
        Start a task.
        
        Args:
            task_name: Task name
            config_file: Configuration file path (optional)
        """
        # Load configuration if provided
        config = None
        if config_file:
            # In a real implementation, this would load config from file
            # For now, we'll just create a basic config
            config = PluginConfig(name=task_name)
        
        # Start task
        if self.task_manager.start_task(task_name, config):
            print(f"Task {task_name} started successfully")
        else:
            print(f"Failed to start task {task_name}")
            sys.exit(1)
    
    def _stop_task(self, task_name: str) -> None:
        """
        Stop a task.
        
        Args:
            task_name: Task name
        """
        if self.task_manager.stop_task(task_name):
            print(f"Task {task_name} stopped successfully")
        else:
            print(f"Failed to stop task {task_name}")
            sys.exit(1)
    
    def _execute_task(self, task_name: str, args: List[str]) -> None:
        """
        Execute a task.
        
        Args:
            task_name: Task name
            args: Additional arguments
        """
        # Parse additional arguments into kwargs
        kwargs = self._parse_args(args)
        
        # Start task if not already active
        if task_name not in self.task_manager.list_tasks():
            print(f"Task {task_name} not found")
            sys.exit(1)
        
        # Start task if not active
        task_status = self.task_manager.list_tasks()
        if task_status.get(task_name) != "active":
            if not self.task_manager.start_task(task_name):
                print(f"Failed to start task {task_name}")
                sys.exit(1)
        
        # Execute task
        result = self.task_manager.execute_task(task_name, **kwargs)
        if result is not None:
            print(f"Task {task_name} result: {result}")
        else:
            print(f"Task {task_name} executed with no result")
    
    def _parse_args(self, args: List[str]) -> Dict[str, Any]:
        """
        Parse command line arguments into keyword arguments.
        
        Args:
            args: Command line arguments
            
        Returns:
            Dictionary of keyword arguments
        """
        kwargs = {}
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--"):
                key = arg[2:]
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    # Next argument is value
                    value = args[i + 1]
                    # Try to convert to appropriate type
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value)
                    elif value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                    kwargs[key] = value
                    i += 2
                else:
                    # Flag argument
                    kwargs[key] = True
                    i += 1
            else:
                i += 1
        return kwargs
    
    def add_task(self, name: str, task_class) -> None:
        """
        Add a task to the runner.
        
        Args:
            name: Task name
            task_class: Task class
        """
        self.task_manager.register_task(name, task_class)


def main():
    """Main entry point."""
    runner = TaskRunner()
    runner.run_from_command_line()


if __name__ == "__main__":
    main()