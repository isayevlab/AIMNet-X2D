"""
Logging configuration for AIMNet-X2D.

This module provides centralized logging configuration that:
- Integrates with AIMNET_DEBUG environment variable
- Supports both console and file logging
- Provides module-level loggers with consistent formatting
- Handles distributed training (rank-aware logging)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


# Check debug mode from environment
DEBUG_MODE = os.environ.get('AIMNET_DEBUG', '').lower() in ('1', 'true', 'yes')

# Default log level based on debug mode
DEFAULT_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

# Log format with timestamp, level, and module name
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
LOG_FORMAT_SIMPLE = '%(levelname)s - %(name)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
    simple_format: bool = False
) -> logging.Logger:
    """
    Configure logging for AIMNet-X2D.

    Args:
        level: Logging level (default: DEBUG if AIMNET_DEBUG=1, else INFO)
        log_file: Optional path to log file
        rank: Process rank for distributed training (only rank 0 logs by default)
        world_size: Total number of processes
        simple_format: Use simpler format without timestamps

    Returns:
        Root logger for the application
    """
    if level is None:
        level = DEFAULT_LEVEL

    # Get root logger
    root_logger = logging.getLogger('aimnet')
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Choose format
    formatter = logging.Formatter(
        LOG_FORMAT_SIMPLE if simple_format else LOG_FORMAT,
        datefmt=DATE_FORMAT
    )

    # Console handler - only rank 0 logs to console in DDP
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler - all ranks can log to separate files if needed
    if log_file:
        log_path = Path(log_file)

        # In DDP, each rank gets its own log file
        if world_size > 1:
            log_path = log_path.with_stem(f"{log_path.stem}_rank{rank}")

        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Usage:
        from utils.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Message")

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance for the module
    """
    # Create child logger under aimnet namespace
    if name.startswith('src.'):
        name = name[4:]  # Remove 'src.' prefix

    return logging.getLogger(f'aimnet.{name}')


def set_log_level(level: int) -> None:
    """
    Set log level for all AIMNet loggers.

    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
    """
    root_logger = logging.getLogger('aimnet')
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


def enable_debug() -> None:
    """Enable debug logging."""
    set_log_level(logging.DEBUG)


def disable_debug() -> None:
    """Disable debug logging (set to INFO)."""
    set_log_level(logging.INFO)


class RankFilter(logging.Filter):
    """
    Filter that only allows logs from specified rank.

    Useful for distributed training where only rank 0 should log.
    """

    def __init__(self, rank: int = 0, allowed_ranks: Optional[list] = None):
        super().__init__()
        self.rank = rank
        self.allowed_ranks = allowed_ranks or [0]

    def filter(self, record: logging.LogRecord) -> bool:
        return self.rank in self.allowed_ranks


# Convenience function for quick setup
def quick_setup(verbose: bool = False) -> logging.Logger:
    """
    Quick logging setup with sensible defaults.

    Args:
        verbose: If True, enable debug logging

    Returns:
        Configured root logger
    """
    level = logging.DEBUG if verbose or DEBUG_MODE else logging.INFO
    return setup_logging(level=level, simple_format=True)
