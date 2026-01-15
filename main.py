#!/usr/bin/env python3
"""
AIMNet-X2D: Advanced Graph Neural Network for Molecular Property Prediction

This is the main entry point for the AIMNet-X2D molecular property prediction system.
It supports both single experiments and legacy hyperparameter optimization.

Usage Examples:
    # Basic training
    python main.py --data_path data.csv --target_column property --epochs 100

    # Multi-task training
    python main.py --data_path data.csv --task_type multitask --multi_target_columns prop1,prop2,prop3

    # Hyperparameter optimization
    python main.py --data_path data.csv --hyperparameter_file config.yaml --num_trials 10

    # Inference
    python main.py --inference_csv test.csv --model_save_path trained_model.pth

    # Multi-GPU training
    torchrun --nproc_per_node=4 main.py --data_path data.csv --num_gpu_devices 4

For detailed documentation and examples, visit the project repository.
"""

import sys
import os
import logging
from pathlib import Path

def setup_python_path():
    """Setup Python path to ensure imports work correctly."""
    # Add src directory to Python path if not already there
    src_dir = Path(__file__).parent / "src"
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

# Always setup path first
setup_python_path()

# Initialize logging after path setup
from utils.logging import setup_logging, get_logger

# Setup logging (rank will be updated later if DDP)
setup_logging()
logger = get_logger(__name__)


def import_modules():
    """Import required modules with proper error handling."""
    try:
        # Import main modules
        from main import main_runner, parse_main_arguments, print_configuration
        from main.hyperopt import run_hyperparameter_optimization
        from main.utils import check_hyperparameter_optimization_mode
        from config import validate_args

        return {
            'main_runner': main_runner,
            'parse_main_arguments': parse_main_arguments,
            'print_configuration': print_configuration,
            'run_hyperparameter_optimization': run_hyperparameter_optimization,
            'check_hyperparameter_optimization_mode': check_hyperparameter_optimization_mode,
            'validate_args': validate_args
        }

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you're running from the project root directory and all dependencies are installed.")
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Python path: {sys.path}")
        sys.exit(1)


def run_single_experiment(args, modules):
    """Run a single experiment without hyperparameter optimization."""
    return modules['main_runner'](args)


def run_hyperparameter_optimization(args, modules):
    """Run hyperparameter optimization."""
    logger.info("Running hyperparameter optimization...")
    return modules['run_hyperparameter_optimization'](args)


def main():
    """Main entry point for AIMNet-X2D."""
    try:
        # Import all required modules
        modules = import_modules()

        # Parse command line arguments
        args = modules['parse_main_arguments']()

        # Print configuration
        modules['print_configuration'](args)

        # Validate arguments
        modules['validate_args'](args)

        # Determine execution mode
        hyperopt_mode = modules['check_hyperparameter_optimization_mode'](args)

        # Execute based on mode
        if hyperopt_mode == 'legacy':
            results = run_hyperparameter_optimization(args, modules)
        else:
            results = run_single_experiment(args, modules)

        logger.info("AIMNet-X2D execution completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Execution failed with error: {e}")

        # Print detailed traceback in debug mode
        if os.environ.get('AIMNET_DEBUG', '').lower() in ('1', 'true', 'yes'):
            import traceback
            traceback.print_exc()
        else:
            logger.info("For detailed error information, set AIMNET_DEBUG=1")

        return 1


if __name__ == "__main__":
    sys.exit(main())