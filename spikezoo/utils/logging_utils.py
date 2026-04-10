import logging
import sys
import os
from typing import Optional, Union
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_to_console: bool = True
    log_to_file: bool = False
    log_file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    colored_output: bool = True


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, fmt: str, datefmt: str = None):
        """
        Initialize colored formatter.
        
        Args:
            fmt: Log format string
            datefmt: Date format string
        """
        super().__init__(fmt, datefmt)
    
    def format(self, record):
        """
        Format log record with colors.
        
        Args:
            record: Log record
            
        Returns:
            Formatted log string
        """
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS and hasattr(record, '_colored'):
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the record
        formatted = super().format(record)
        
        # Restore original levelname
        record.levelname = levelname
        
        return formatted


class UnifiedLogger:
    """Unified logger with consistent formatting and multiple levels."""
    
    def __init__(self, name: str, config: Optional[LogConfig] = None):
        """
        Initialize unified logger.
        
        Args:
            name: Logger name
            config: Logging configuration
        """
        self.name = name
        self.config = config or LogConfig()
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with handlers and formatters."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set level
        level = getattr(logging, self.config.level.value)
        self.logger.setLevel(level)
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            self._add_handler(console_handler)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file and self.config.log_file_path:
            # Create log directory if it doesn't exist
            log_path = Path(self.config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use RotatingFileHandler for log rotation
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    self.config.log_file_path,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count
                )
            except ImportError:
                # Fallback to regular FileHandler
                file_handler = logging.FileHandler(self.config.log_file_path)
            
            self._add_handler(file_handler)
            self.logger.addHandler(file_handler)
    
    def _add_handler(self, handler):
        """
        Add handler with appropriate formatter.
        
        Args:
            handler: Log handler
        """
        if self.config.colored_output and isinstance(handler, logging.StreamHandler):
            formatter = ColoredFormatter(self.config.format, self.config.date_format)
            # Mark record for coloring
            handler.emit = self._colored_emit(handler.emit)
        else:
            formatter = logging.Formatter(self.config.format, self.config.date_format)
        
        handler.setFormatter(formatter)
    
    def _colored_emit(self, original_emit):
        """
        Wrapper for colored emit.
        
        Args:
            original_emit: Original emit method
            
        Returns:
            Wrapped emit method
        """
        def emit(record):
            record._colored = True
            return original_emit(record)
        return emit
    
    def debug(self, message: str, *args, **kwargs):
        """
        Log debug message.
        
        Args:
            message: Debug message
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """
        Log info message.
        
        Args:
            message: Info message
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """
        Log warning message.
        
        Args:
            message: Warning message
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """
        Log error message.
        
        Args:
            message: Error message
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """
        Log critical message.
        
        Args:
            message: Critical message
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """
        Log exception message.
        
        Args:
            message: Exception message
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.exception(message, *args, **kwargs)
    
    def set_level(self, level: Union[LogLevel, str]):
        """
        Set logger level.
        
        Args:
            level: Log level
        """
        if isinstance(level, str):
            level = LogLevel(level.upper())
        
        log_level = getattr(logging, level.value)
        self.logger.setLevel(log_level)
        self.config.level = level
    
    def add_context(self, **context):
        """
        Add context to logger (for future use).
        
        Args:
            **context: Context key-value pairs
        """
        # This is a placeholder for context-aware logging
        # Could be extended to include context in log messages
        pass


# Global logger instances
_loggers = {}


def get_logger(name: str, config: Optional[LogConfig] = None) -> UnifiedLogger:
    """
    Get or create unified logger instance.
    
    Args:
        name: Logger name
        config: Logging configuration (only used for new loggers)
        
    Returns:
        UnifiedLogger instance
    """
    if name not in _loggers:
        _loggers[name] = UnifiedLogger(name, config)
    return _loggers[name]


def setup_logging(config: Optional[LogConfig] = None):
    """
    Setup global logging configuration.
    
    Args:
        config: Logging configuration
    """
    config = config or LogConfig()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    level = getattr(logging, config.level.value)
    root_logger.setLevel(level)
    
    # Console handler
    if config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(config.format, config.date_format)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if config.log_to_file and config.log_file_path:
        # Create log directory if it doesn't exist
        log_path = Path(config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use RotatingFileHandler for log rotation
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                config.log_file_path,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count
            )
        except ImportError:
            # Fallback to regular FileHandler
            file_handler = logging.FileHandler(config.log_file_path)
        
        formatter = logging.Formatter(config.format, config.date_format)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_default_config() -> LogConfig:
    """
    Get default logging configuration.
    
    Returns:
        Default LogConfig instance
    """
    return LogConfig()


def create_file_logger(name: str, log_file: str, level: LogLevel = LogLevel.INFO) -> UnifiedLogger:
    """
    Create logger that logs to file.
    
    Args:
        name: Logger name
        log_file: Log file path
        level: Log level
        
    Returns:
        UnifiedLogger instance
    """
    config = LogConfig(
        level=level,
        log_to_console=True,
        log_to_file=True,
        log_file_path=log_file,
        colored_output=False
    )
    return get_logger(name, config)


def create_console_logger(name: str, level: LogLevel = LogLevel.INFO) -> UnifiedLogger:
    """
    Create console-only logger.
    
    Args:
        name: Logger name
        level: Log level
        
    Returns:
        UnifiedLogger instance
    """
    config = LogConfig(
        level=level,
        log_to_console=True,
        log_to_file=False,
        colored_output=True
    )
    return get_logger(name, config)


# Convenience functions for quick logging
def debug(message: str, logger_name: str = "default"):
    """Log debug message."""
    logger = get_logger(logger_name)
    logger.debug(message)


def info(message: str, logger_name: str = "default"):
    """Log info message."""
    logger = get_logger(logger_name)
    logger.info(message)


def warning(message: str, logger_name: str = "default"):
    """Log warning message."""
    logger = get_logger(logger_name)
    logger.warning(message)


def error(message: str, logger_name: str = "default"):
    """Log error message."""
    logger = get_logger(logger_name)
    logger.error(message)


def critical(message: str, logger_name: str = "default"):
    """Log critical message."""
    logger = get_logger(logger_name)
    logger.critical(message)