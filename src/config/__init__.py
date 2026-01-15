# config/__init__.py
"""
Configuration package for AIMNet-X2D.

This package contains configuration management, argument validation,
experiment configuration utilities, and robust error handling.
"""

# Argument parsing
from .args import create_argument_parser, parse_arguments
from .validation import validate_args, ValidationError, ConfigurationError
from .experiment import save_experiment_config, load_experiment_config, ExperimentError
from .paths import setup_paths, create_directories, ensure_path_exists, PathError

# Model and training constants
from .constants import (
    MESSAGE_PASSING_RATIO,
    DEFAULT_ATTENTION_TEMPERATURE,
    TETRAHEDRAL_MAGNITUDE_SCALE,
    GRADIENT_CLIP_MAX_NORM,
    DEFAULT_EVIDENTIAL_LAMBDA,
)

# Legacy compatibility - import the main validate_args function
from .validation import validate_args

__all__ = [
    # Argument parsing
    "create_argument_parser",
    "parse_arguments",

    # Validation
    "validate_args",
    "ValidationError",
    "ConfigurationError",

    # Experiment management
    "save_experiment_config",
    "load_experiment_config",
    "ExperimentError",

    # Path management
    "setup_paths",
    "create_directories",
    "ensure_path_exists",
    "PathError",

    # Model and training constants
    "MESSAGE_PASSING_RATIO",
    "DEFAULT_ATTENTION_TEMPERATURE",
    "TETRAHEDRAL_MAGNITUDE_SCALE",
    "GRADIENT_CLIP_MAX_NORM",
    "DEFAULT_EVIDENTIAL_LAMBDA",
]