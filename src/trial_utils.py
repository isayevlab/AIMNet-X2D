"""
Trial environment utilities for hyperparameter optimization.

This module contains helper functions for setting up trial environments
without circular import issues.
"""

import os
import copy
import tempfile
import shutil
from typing import Dict, Any
from pathlib import Path

import torch


def setup_trial_environment(base_args, config: Dict[str, Any]):
    """
    Setup environment for a single trial/experiment.
    
    Args:
        base_args: Base command line arguments
        config: Trial-specific configuration from hyperparameter optimization
        
    Returns:
        Modified arguments for this trial
    """
    # Create a copy of base arguments
    trial_args = copy.deepcopy(base_args)
    
    # Update with trial configuration
    for param_name, param_value in config.items():
        # Handle special parameter mappings
        if param_name == "multitask_weights" and isinstance(param_value, list):
            trial_args.multitask_weights = ",".join(map(str, param_value))
            trial_args.multitask_weights_list = param_value
        else:
            setattr(trial_args, param_name, param_value)
    
    # Create trial-specific paths to avoid conflicts
    trial_id = f"trial_{torch.randint(0, 1000000, (1,)).item()}"
    
    # Setup temporary directory for trial artifacts
    trial_temp_dir = tempfile.mkdtemp(prefix=f"aimnet_trial_{trial_id}_")
    trial_args._trial_temp_dir = trial_temp_dir
    
    # Create trial-specific model save path
    if hasattr(trial_args, 'model_save_path') and trial_args.model_save_path:
        base_name = Path(trial_args.model_save_path).stem
        extension = Path(trial_args.model_save_path).suffix
        trial_args.model_save_path = os.path.join(
            trial_temp_dir, f"{base_name}_{trial_id}{extension}"
        )
    
    # Setup trial-specific embedding paths if needed
    if hasattr(trial_args, 'embeddings_output_path') and trial_args.save_embeddings:
        base_name = Path(trial_args.embeddings_output_path).stem
        extension = Path(trial_args.embeddings_output_path).suffix
        trial_args.embeddings_output_path = os.path.join(
            trial_temp_dir, f"{base_name}_{trial_id}{extension}"
        )
    
    # Setup trial-specific HDF5 paths if using iterable datasets
    if trial_args.iterable_dataset:
        for attr in ['train_hdf5', 'val_hdf5', 'test_hdf5']:
            if hasattr(trial_args, attr) and getattr(trial_args, attr):
                original_path = getattr(trial_args, attr)
                base_name = Path(original_path).stem
                extension = Path(original_path).suffix
                trial_path = os.path.join(
                    trial_temp_dir, f"{base_name}_{trial_id}{extension}"
                )
                setattr(trial_args, attr, trial_path)
    
    # Set deterministic seed for reproducibility
    from utils.random import set_seed
    set_seed(42 + hash(trial_id) % 1000)
    
    # Disable wandb for individual trials (will be handled by hyperopt module)
    trial_args.enable_wandb = False
    
    return trial_args


def cleanup_trial_environment():
    """Clean up trial environment after completion."""
    # PyTorch cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Clear any remaining references
    import gc
    gc.collect()


def cleanup_temporary_files(args) -> None:
    """
    FIXED: Clean up temporary files created during execution with better error handling.
    """
    try:
        # Clean up trial-specific temporary directory if it exists
        if hasattr(args, '_trial_temp_dir') and os.path.exists(args._trial_temp_dir):
            try:
                shutil.rmtree(args._trial_temp_dir)
                print(f"Cleaned up temp files/directories: {args._trial_temp_dir}")
            except Exception as e:
                print(f"WARNING: Failed to clean up temporary directory {args._trial_temp_dir}: {e}")
        
        # CRITICAL FIX: Clean up any orphaned temporary directories
        import tempfile
        temp_dir = tempfile.gettempdir()
        
        # Look for directories starting with 'aimnet_trial_' or containing '???'
        for item in os.listdir(temp_dir):
            if item.startswith('aimnet_trial_') or item == '???':
                orphan_path = os.path.join(temp_dir, item)
                if os.path.isdir(orphan_path):
                    try:
                        shutil.rmtree(orphan_path)
                        print(f"Cleaned up orphaned directory: {orphan_path}")
                    except Exception as e:
                        print(f"WARNING: Could not clean up {orphan_path}: {e}")
        
        # Clean up current working directory for any '???' folders
        cwd = os.getcwd()
        mystery_dir = os.path.join(cwd, '???')
        if os.path.exists(mystery_dir):
            try:
                shutil.rmtree(mystery_dir)
                print(f"Cleaned up mystery directory: {mystery_dir}")
            except Exception as e:
                print(f"WARNING: Could not clean up mystery directory: {e}")
                
    except Exception as e:
        print(f"WARNING: Error during cleanup: {e}")



def validate_trial_arguments(args) -> bool:
    """
    Validate arguments for a trial run.
    
    Args:
        args: Trial arguments to validate
        
    Returns:
        True if arguments are valid
        
    Raises:
        ValueError: If validation fails
    """
    # Import validation from config module
    from config import validate_args
    
    try:
        validate_args(args)
        return True
    except Exception as e:
        print(f"Argument validation failed: {e}")
        return False