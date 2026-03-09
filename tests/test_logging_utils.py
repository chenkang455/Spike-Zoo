import unittest
import sys
import os
import logging
import tempfile
import shutil
from pathlib import Path

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


class TestLogLevel(unittest.TestCase):
    """LogLevel enum unit tests."""
    
    def test_log_level_values(self):
        """Test LogLevel enum values."""
        self.assertEqual(LogLevel.DEBUG.value, "DEBUG")
        self.assertEqual(LogLevel.INFO.value, "INFO")
        self.assertEqual(LogLevel.WARNING.value, "WARNING")
        self.assertEqual(LogLevel.ERROR.value, "ERROR")
        self.assertEqual(LogLevel.CRITICAL.value, "CRITICAL")


class TestLogConfig(unittest.TestCase):
    """LogConfig dataclass unit tests."""
    
    def test_log_config_default_values(self):
        """Test LogConfig default values."""
        config = LogConfig()
        
        self.assertEqual(config.level, LogLevel.INFO)
        self.assertEqual(config.format, "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
        self.assertEqual(config.date_format, "%Y-%m-%d %H:%M:%S")
        self.assertTrue(config.log_to_console)
        self.assertFalse(config.log_to_file)
        self.assertIsNone(config.log_file_path)
        self.assertEqual(config.max_file_size, 10 * 1024 * 1024)
        self.assertEqual(config.backup_count, 5)
        self.assertTrue(config.colored_output)
    
    def test_log_config_custom_values(self):
        """Test LogConfig with custom values."""
        config = LogConfig(
            level=LogLevel.DEBUG,
            format="%(levelname)s: %(message)s",
            date_format="%H:%M:%S",
            log_to_console=False,
            log_to_file=True,
            log_file_path="/tmp/test.log",
            max_file_size=5 * 1024 * 1024,
            backup_count=3,
            colored_output=False
        )
        
        self.assertEqual(config.level, LogLevel.DEBUG)
        self.assertEqual(config.format, "%(levelname)s: %(message)s")
        self.assertEqual(config.date_format, "%H:%M:%S")
        self.assertFalse(config.log_to_console)
        self.assertTrue(config.log_to_file)
        self.assertEqual(config.log_file_path, "/tmp/test.log")
        self.assertEqual(config.max_file_size, 5 * 1024 * 1024)
        self.assertEqual(config.backup_count, 3)
        self.assertFalse(config.colored_output)


class TestUnifiedLogger(unittest.TestCase):
    """UnifiedLogger unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test cleanup."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_unified_logger_creation(self):
        """Test UnifiedLogger creation."""
        logger = UnifiedLogger("test_logger")
        
        self.assertEqual(logger.name, "test_logger")
        self.assertIsInstance(logger.config, LogConfig)
        self.assertEqual(logger.config.level, LogLevel.INFO)
    
    def test_unified_logger_with_config(self):
        """Test UnifiedLogger creation with custom config."""
        config = LogConfig(level=LogLevel.DEBUG)
        logger = UnifiedLogger("test_logger", config)
        
        self.assertEqual(logger.config.level, LogLevel.DEBUG)
    
    def test_unified_logger_log_methods(self):
        """Test UnifiedLogger log methods."""
        logger = UnifiedLogger("test_logger")
        
        # These should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        logger.exception("Exception message")
    
    def test_unified_logger_set_level(self):
        """Test UnifiedLogger set_level method."""
        logger = UnifiedLogger("test_logger")
        
        # Change level to DEBUG
        logger.set_level(LogLevel.DEBUG)
        self.assertEqual(logger.config.level, LogLevel.DEBUG)
        
        # Change level using string
        logger.set_level("ERROR")
        self.assertEqual(logger.config.level, LogLevel.ERROR)
    
    def test_unified_logger_file_logging(self):
        """Test UnifiedLogger file logging."""
        log_file = os.path.join(self.temp_dir, "test.log")
        config = LogConfig(
            level=LogLevel.INFO,
            log_to_console=False,
            log_to_file=True,
            log_file_path=log_file
        )
        
        logger = UnifiedLogger("file_logger", config)
        logger.info("Test file logging")
        
        # Check that log file was created
        self.assertTrue(os.path.exists(log_file))
        
        # Check log content
        with open(log_file, 'r') as f:
            content = f.read()
            self.assertIn("Test file logging", content)
    
    def test_unified_logger_log_rotation(self):
        """Test UnifiedLogger log rotation."""
        log_file = os.path.join(self.temp_dir, "rotate_test.log")
        config = LogConfig(
            level=LogLevel.INFO,
            log_to_console=False,
            log_to_file=True,
            log_file_path=log_file,
            max_file_size=100,  # Small size for testing
            backup_count=2
        )
        
        logger = UnifiedLogger("rotation_logger", config)
        
        # Write many messages to trigger rotation
        for i in range(50):
            logger.info(f"Log message {i}: " + "x" * 20)
        
        # Check that log file and backups exist
        self.assertTrue(os.path.exists(log_file))
        self.assertTrue(os.path.exists(log_file + ".1"))
        # Backup count is 2, so we should have .1 and .2
        # But actual behavior depends on RotatingFileHandler implementation
    
    def test_unified_logger_context(self):
        """Test UnifiedLogger context methods."""
        logger = UnifiedLogger("context_logger")
        
        # This should not raise exceptions
        logger.add_context(user_id=123, session_id="abc")
        logger.info("Message with context")


class TestGlobalFunctions(unittest.TestCase):
    """Test global logging functions."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test cleanup."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")
        
        # Should return the same instance
        self.assertIs(logger1, logger2)
        
        # Different name should return different instance
        logger3 = get_logger("different_logger")
        self.assertIsNot(logger1, logger3)
    
    def test_get_logger_with_config(self):
        """Test get_logger function with config."""
        config = LogConfig(level=LogLevel.DEBUG)
        logger = get_logger("config_logger", config)
        
        self.assertEqual(logger.config.level, LogLevel.DEBUG)
    
    def test_setup_logging(self):
        """Test setup_logging function."""
        log_file = os.path.join(self.temp_dir, "global.log")
        config = LogConfig(
            level=LogLevel.INFO,
            log_to_console=False,
            log_to_file=True,
            log_file_path=log_file
        )
        
        setup_logging(config)
        
        # Test that root logger is configured
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.INFO)
        self.assertGreater(len(root_logger.handlers), 0)
    
    def test_get_default_config(self):
        """Test get_default_config function."""
        config = get_default_config()
        self.assertIsInstance(config, LogConfig)
        self.assertEqual(config.level, LogLevel.INFO)
    
    def test_create_file_logger(self):
        """Test create_file_logger function."""
        log_file = os.path.join(self.temp_dir, "file_logger.log")
        logger = create_file_logger("file_test", log_file, LogLevel.WARNING)
        
        self.assertIsInstance(logger, UnifiedLogger)
        self.assertEqual(logger.config.level, LogLevel.WARNING)
        self.assertTrue(logger.config.log_to_file)
        self.assertEqual(logger.config.log_file_path, log_file)
    
    def test_create_console_logger(self):
        """Test create_console_logger function."""
        logger = create_console_logger("console_test", LogLevel.ERROR)
        
        self.assertIsInstance(logger, UnifiedLogger)
        self.assertEqual(logger.config.level, LogLevel.ERROR)
        self.assertTrue(logger.config.log_to_console)
        self.assertFalse(logger.config.log_to_file)
    
    def test_convenience_functions(self):
        """Test convenience logging functions."""
        # These should not raise exceptions
        debug("Debug message")
        info("Info message")
        warning("Warning message")
        error("Error message")
        critical("Critical message")
        
        # Test with custom logger name
        info("Custom logger message", "custom_logger")
        error("Custom logger error", "custom_logger")


class TestIntegration(unittest.TestCase):
    """Integration tests for logging system."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test cleanup."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_multiple_loggers_different_configs(self):
        """Test multiple loggers with different configurations."""
        # Create loggers with different configurations
        log_file1 = os.path.join(self.temp_dir, "logger1.log")
        log_file2 = os.path.join(self.temp_dir, "logger2.log")
        
        config1 = LogConfig(
            level=LogLevel.DEBUG,
            log_to_console=True,
            log_to_file=True,
            log_file_path=log_file1
        )
        
        config2 = LogConfig(
            level=LogLevel.ERROR,
            log_to_console=False,
            log_to_file=True,
            log_file_path=log_file2
        )
        
        logger1 = get_logger("debug_logger", config1)
        logger2 = get_logger("error_logger", config2)
        
        # Log messages
        logger1.debug("Debug message from logger1")
        logger1.info("Info message from logger1")
        logger2.error("Error message from logger2")
        logger2.critical("Critical message from logger2")
        
        # Check log files
        self.assertTrue(os.path.exists(log_file1))
        self.assertTrue(os.path.exists(log_file2))
        
        # Check content
        with open(log_file1, 'r') as f:
            content1 = f.read()
            self.assertIn("Debug message from logger1", content1)
            self.assertIn("Info message from logger1", content1)
        
        with open(log_file2, 'r') as f:
            content2 = f.read()
            self.assertIn("Error message from logger2", content2)
            self.assertIn("Critical message from logger2", content2)
    
    def test_log_level_changes(self):
        """Test log level changes affect output."""
        log_file = os.path.join(self.temp_dir, "level_test.log")
        config = LogConfig(
            level=LogLevel.INFO,
            log_to_console=False,
            log_to_file=True,
            log_file_path=log_file
        )
        
        logger = get_logger("level_test_logger", config)
        
        # Log at different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        # Check initial content
        with open(log_file, 'r') as f:
            initial_content = f.read()
            self.assertNotIn("Debug message", initial_content)
            self.assertIn("Info message", initial_content)
            self.assertIn("Warning message", initial_content)
        
        # Change level and log again
        logger.set_level(LogLevel.DEBUG)
        logger.debug("New debug message")
        
        # Check updated content
        with open(log_file, 'r') as f:
            updated_content = f.read()
            self.assertIn("New debug message", updated_content)
    
    def test_exception_logging_integration(self):
        """Test exception logging integration."""
        log_file = os.path.join(self.temp_dir, "exception_test.log")
        config = LogConfig(
            level=LogLevel.INFO,
            log_to_console=False,
            log_to_file=True,
            log_file_path=log_file
        )
        
        logger = get_logger("exception_test_logger", config)
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Caught exception")
        
        # Check log content contains exception info
        with open(log_file, 'r') as f:
            content = f.read()
            self.assertIn("Caught exception", content)
            self.assertIn("Traceback", content)
            self.assertIn("ValueError: Test exception", content)


if __name__ == '__main__':
    unittest.main()