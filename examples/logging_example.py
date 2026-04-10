#!/usr/bin/env python3
"""
Example of using the SpikeZoo unified logging system.
"""

import sys
import os
import time

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.utils.logging_utils import (
    UnifiedLogger,
    LogConfig,
    LogLevel,
    get_logger,
    setup_logging,
    get_default_config,
    create_file_logger,
    create_console_logger,
    debug,
    info,
    warning,
    error,
    critical
)


def example_basic_logging():
    """Example of basic logging usage."""
    print("=== Basic Logging Example ===\n")
    
    # Create logger
    logger = get_logger("basic_example")
    
    # Log messages at different levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    print()


def example_custom_configuration():
    """Example of custom logging configuration."""
    print("=== Custom Configuration Example ===\n")
    
    # Create custom configuration
    config = LogConfig(
        level=LogLevel.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        date_format="%H:%M:%S",
        log_to_console=True,
        log_to_file=True,
        log_file_path="./logs/custom_example.log",
        colored_output=True
    )
    
    # Create logger with custom config
    logger = get_logger("custom_example", config)
    
    # Log messages
    logger.debug("Debug message with custom format")
    logger.info("Info message with custom format")
    logger.warning("Warning message with custom format")
    
    print("Logged messages with custom configuration")
    print()


def example_file_logging():
    """Example of file logging."""
    print("=== File Logging Example ===\n")
    
    # Create file logger
    file_logger = create_file_logger("file_example", "./logs/file_example.log", LogLevel.INFO)
    
    # Log messages
    file_logger.info("This message goes to both console and file")
    file_logger.warning("This warning also goes to both console and file")
    file_logger.error("This error goes to both console and file")
    
    print("Logged messages to file and console")
    print()


def example_console_only_logging():
    """Example of console-only logging."""
    print("=== Console-Only Logging Example ===\n")
    
    # Create console-only logger
    console_logger = create_console_logger("console_example", LogLevel.WARNING)
    
    # These won't appear because level is WARNING
    console_logger.debug("This debug message won't appear")
    console_logger.info("This info message won't appear")
    
    # These will appear
    console_logger.warning("This warning message will appear")
    console_logger.error("This error message will appear")
    console_logger.critical("This critical message will appear")
    
    print()


def example_multiple_loggers():
    """Example of using multiple loggers."""
    print("=== Multiple Loggers Example ===\n")
    
    # Create multiple loggers with different configurations
    app_logger = get_logger("app")
    db_logger = get_logger("database")
    api_logger = get_logger("api")
    
    # Log messages from different components
    app_logger.info("Application started")
    db_logger.info("Database connection established")
    api_logger.info("API server listening on port 8000")
    
    # Simulate some operations
    app_logger.debug("Processing user request")
    db_logger.debug("Executing query: SELECT * FROM users")
    api_logger.warning("Rate limit approaching for client 192.168.1.100")
    
    print()


def example_log_levels():
    """Example of changing log levels."""
    print("=== Log Levels Example ===\n")
    
    # Create logger
    logger = get_logger("level_example")
    
    print("1. Default level (INFO):")
    logger.debug("Debug message (won't appear)")
    logger.info("Info message (will appear)")
    logger.warning("Warning message (will appear)")
    
    # Change level to DEBUG
    print("\n2. Changed to DEBUG level:")
    logger.set_level(LogLevel.DEBUG)
    logger.debug("Debug message (will now appear)")
    logger.info("Info message (will appear)")
    
    # Change level to ERROR
    print("\n3. Changed to ERROR level:")
    logger.set_level("ERROR")  # Can also use string
    logger.info("Info message (won't appear)")
    logger.error("Error message (will appear)")
    
    print()


def example_exception_logging():
    """Example of exception logging."""
    print("=== Exception Logging Example ===\n")
    
    logger = get_logger("exception_example")
    
    try:
        # Simulate an error
        result = 10 / 0
    except ZeroDivisionError as e:
        logger.exception("Division by zero error occurred")
        # Or manually log exception
        logger.error("Manual exception logging: %s", str(e))
    
    print()


def example_global_setup():
    """Example of global logging setup."""
    print("=== Global Setup Example ===\n")
    
    # Setup global logging configuration
    config = LogConfig(
        level=LogLevel.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        log_to_console=True,
        log_to_file=True,
        log_file_path="./logs/global.log"
    )
    
    setup_logging(config)
    
    # Now all loggers will use this configuration
    logger1 = get_logger("global_logger1")
    logger2 = get_logger("global_logger2")
    
    logger1.info("Message from logger 1")
    logger2.warning("Message from logger 2")
    
    print("Global logging setup complete")
    print()


def example_convenience_functions():
    """Example of convenience logging functions."""
    print("=== Convenience Functions Example ===\n")
    
    # Use convenience functions for quick logging
    debug("Quick debug message")
    info("Quick info message")
    warning("Quick warning message")
    error("Quick error message")
    critical("Quick critical message")
    
    # Use with custom logger name
    info("Custom logger message", "my_app")
    error("Custom logger error", "my_app")
    
    print()


def example_performance_logging():
    """Example of performance logging."""
    print("=== Performance Logging Example ===\n")
    
    logger = get_logger("performance_example")
    
    # Log timing information
    start_time = time.time()
    
    # Simulate some work
    time.sleep(0.1)
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info("Operation completed in %.2f seconds", duration)
    logger.debug("Detailed timing: start=%.6f, end=%.6f, duration=%.6f", 
                 start_time, end_time, duration)
    
    print()


def example_context_logging():
    """Example of context-aware logging."""
    print("=== Context Logging Example ===\n")
    
    # Create logger for specific context
    train_logger = get_logger("training")
    eval_logger = get_logger("evaluation")
    
    # Log messages with context
    train_logger.info("Starting training epoch 1")
    train_logger.debug("Batch 100/1000, loss=0.45")
    
    eval_logger.info("Starting evaluation")
    eval_logger.debug("Processing sample 50/1000")
    eval_logger.info("Evaluation completed, accuracy=0.87")
    
    print()


if __name__ == "__main__":
    # Create logs directory
    import os
    os.makedirs("./logs", exist_ok=True)
    
    example_basic_logging()
    example_custom_configuration()
    example_file_logging()
    example_console_only_logging()
    example_multiple_loggers()
    example_log_levels()
    example_exception_logging()
    example_global_setup()
    example_convenience_functions()
    example_performance_logging()
    example_context_logging()
    
    print("All logging examples completed!")
    print("Check the logs directory for log files.")