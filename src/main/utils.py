"""
Utility functions for main execution.

This module contains helper functions for setting up experiments,
managing trial environments, and handling configurations.
"""

import os
import copy
import tempfile
import shutil
import argparse
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import numpy as np

from utils import set_seed


# Import trial utilities from separate module to avoid circular imports
from trial_utils import setup_trial_environment, cleanup_trial_environment, cleanup_temporary_files, validate_trial_arguments

def setup_distributed_environment(args: argparse.Namespace) -> Tuple[torch.device, bool, int, int]:
    """
    Setup distributed training environment.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (device, is_ddp, local_rank, world_size)
    """
    import torch.distributed as dist
    
    is_ddp = False
    local_rank = 0
    world_size = 1
    
    # Check if we're in a distributed environment
    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # We're running under torchrun
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if torch.cuda.is_available() and world_size > 1:
            try:
                # Initialize the process group
                dist.init_process_group(backend="nccl")
                
                # CRITICAL: Set the device for this process
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
                
                is_ddp = True
                
                print(f"[DDP] Initialized: rank={dist.get_rank()}, "
                      f"local_rank={local_rank}, world_size={world_size}, device={device}")
                      
            except Exception as e:
                print(f"[DDP] Failed to initialize: {e}")
                print("Falling back to single GPU training")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            print("CUDA not available or world_size <= 1, using CPU/single GPU")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Not running under torchrun, single process
        if args.num_gpu_devices > 1:
            print("WARNING: num_gpu_devices > 1 but not running under torchrun")
            print("Use: torchrun --nproc_per_node=N main.py ... for multi-GPU training")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Single-Process] Using device: {device}")
    
    return device, is_ddp, local_rank, world_size

def setup_model_paths(args: argparse.Namespace) -> None:
    """
    Setup and validate model-related paths.
    
    Args:
        args: Command line arguments
    """
    # Ensure model save directory exists
    if args.model_save_path:
        model_dir = os.path.dirname(os.path.abspath(args.model_save_path))
        os.makedirs(model_dir, exist_ok=True)
    
    # Ensure embeddings output directory exists if needed
    if args.save_embeddings and args.embeddings_output_path:
        embeddings_dir = os.path.dirname(os.path.abspath(args.embeddings_output_path))
        os.makedirs(embeddings_dir, exist_ok=True)
    
    # Ensure inference output directory exists if needed
    if args.inference_output:
        inference_dir = os.path.dirname(os.path.abspath(args.inference_output))
        if inference_dir:  # Only create if there's actually a directory part
            os.makedirs(inference_dir, exist_ok=True)


def check_data_consistency(args: argparse.Namespace) -> None:
    """
    Check consistency of data-related arguments.
    
    Args:
        args: Command line arguments
        
    Raises:
        ValueError: If data configuration is inconsistent
    """
    # Check data path consistency
    if args.data_path:
        if args.train_data or args.val_data or args.test_data:
            raise ValueError(
                "Cannot specify both --data_path and individual train/val/test files"
            )
        if not os.path.exists(args.data_path):
            raise ValueError(f"Data file not found: {args.data_path}")
    else:
        # Check individual files
        if not (args.train_data and args.val_data and args.test_data):
            if not (args.inference_csv or args.inference_hdf5):
                raise ValueError(
                    "Must specify either --data_path or all of "
                    "--train_data, --val_data, --test_data"
                )
        
        # Check file existence
        for file_path, name in [(args.train_data, "train"), 
                               (args.val_data, "validation"), 
                               (args.test_data, "test")]:
            if file_path and not os.path.exists(file_path):
                raise ValueError(f"{name} data file not found: {file_path}")
    
    # Check task type consistency
    if args.task_type == "multitask":
        if not args.multi_target_columns:
            raise ValueError(
                "Must specify --multi_target_columns for multitask learning"
            )
        
        # Check multitask weights consistency
        if args.multitask_weights_list:
            num_targets = len(args.multi_target_columns_list)
            num_weights = len(args.multitask_weights_list)
            if num_targets != num_weights:
                raise ValueError(
                    f"Number of multitask weights ({num_weights}) must match "
                    f"number of target columns ({num_targets})"
                )


def get_experiment_id(args: argparse.Namespace) -> str:
    """
    Generate a unique experiment ID based on configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Unique experiment identifier
    """
    import hashlib
    import time
    
    # Create a string representation of key configuration
    config_str = f"{args.task_type}_{args.hidden_dim}_{args.num_shells}"
    config_str += f"_{args.learning_rate}_{args.batch_size}"
    config_str += f"_{args.pooling_type}_{args.activation_type}"
    
    # Add timestamp for uniqueness
    timestamp = str(int(time.time()))
    
    # Create hash
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    return f"aimnet_{config_hash}_{timestamp}"


def log_system_info() -> None:
    """Log system information for debugging and reproducibility."""
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # System info
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"CPU cores: {os.cpu_count()}")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total memory: {memory.total / (1024**3):.2f} GB")
        print(f"Available memory: {memory.available / (1024**3):.2f} GB")
    except ImportError:
        print("Memory info unavailable (psutil not installed)")
    
    print("="*60)


def create_experiment_summary(args: argparse.Namespace, results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive experiment summary.
    
    Args:
        args: Command line arguments
        results: Experiment results
        
    Returns:
        Experiment summary dictionary
    """
    import datetime
    
    summary = {
        "experiment_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "experiment_id": get_experiment_id(args),
            "task_type": args.task_type,
            "total_epochs": args.epochs,
        },
        "data_config": {
            "data_path": getattr(args, 'data_path', None),
            "train_data": getattr(args, 'train_data', None),
            "val_data": getattr(args, 'val_data', None),
            "test_data": getattr(args, 'test_data', None),
            "smiles_column": args.smiles_column,
            "target_column": getattr(args, 'target_column', None),
            "multi_target_columns": getattr(args, 'multi_target_columns', None),
            "iterable_dataset": args.iterable_dataset,
        },
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "num_shells": args.num_shells,
            "num_message_passing_layers": args.num_message_passing_layers,
            "embedding_dim": args.embedding_dim,
            "pooling_type": args.pooling_type,
            "activation_type": args.activation_type,
            "use_partial_charges": args.use_partial_charges,
            "use_stereochemistry": args.use_stereochemistry,
        },
        "training_config": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "loss_function": args.loss_function,
            "early_stopping": args.early_stopping,
            "patience": args.patience if args.early_stopping else None,
            "lr_scheduler": args.lr_scheduler,
        },
        "system_config": {
            "num_gpu_devices": args.num_gpu_devices,
            "num_workers": args.num_workers,
            "mixed_precision": args.mixed_precision,
        },
        "results": results,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    return summary


def save_experiment_summary(summary: Dict[str, Any], output_path: str) -> None:
    """
    Save experiment summary to file.
    
    Args:
        summary: Experiment summary dictionary
        output_path: Path to save the summary
    """
    import json
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save summary
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Experiment summary saved to: {output_path}")


def handle_inference_mode(args: argparse.Namespace) -> bool:
    """
    Check if we're in inference mode and handle appropriately.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if in inference mode, False otherwise
    """
    if args.inference_csv or args.inference_hdf5:
        print("Running in inference mode - training will be skipped")
        
        # Auto-generate output path if not provided
        if not args.inference_output:
            if args.inference_csv:
                base = os.path.splitext(args.inference_csv)[0]
                args.inference_output = f"{base}_predictions.csv"
            elif args.inference_hdf5:
                base = os.path.splitext(args.inference_hdf5)[0]
                args.inference_output = f"{base}_predictions.csv"
            print(f"Using auto-generated output path: {args.inference_output}")
        
        # Ensure output directory exists
        if args.inference_output:
            output_dir = os.path.dirname(os.path.abspath(args.inference_output))
            os.makedirs(output_dir, exist_ok=True)
        
        return True
    
    return False


def check_hyperparameter_optimization_mode(args: argparse.Namespace) -> str:
    """
    Determine which hyperparameter optimization mode to use.
    
    Args:
        args: Command line arguments
        
    Returns:
        String indicating optimization mode: 'legacy' or 'none'
    """
    if args.hyperparameter_file and args.num_trials > 1:
        return 'legacy'
    else:
        return 'none'


def prepare_wandb_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Prepare configuration dictionary for Weights & Biases logging."""
    wandb_config = {
        # Model architecture
        "hidden_dim": args.hidden_dim,
        "num_shells": args.num_shells,
        "num_message_passing_layers": args.num_message_passing_layers,
        "embedding_dim": args.embedding_dim,
        "pooling_type": args.pooling_type,
        "activation_type": args.activation_type,
        "ffn_num_layers": args.ffn_num_layers,
        "ffn_dropout": args.ffn_dropout,
        
        # Training configuration
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "task_type": args.task_type,
        "loss_function": args.loss_function,
        "early_stopping": args.early_stopping,
        
        # Features
        "use_partial_charges": args.use_partial_charges,
        "use_stereochemistry": args.use_stereochemistry,
        "calculate_sae": args.calculate_sae,
        
        # System
        "num_gpu_devices": args.num_gpu_devices,
        "mixed_precision": args.mixed_precision,
        "iterable_dataset": args.iterable_dataset,
    }
    
    # Add task-specific configuration
    if args.task_type == "multitask":
        wandb_config["multi_target_columns"] = args.multi_target_columns
        if args.multitask_weights_list:
            wandb_config["multitask_weights"] = args.multitask_weights_list
    
    return wandb_config

def setup_experiment_logging(args: argparse.Namespace) -> Optional[Any]:
    """Setup experiment logging with Weights & Biases."""
    if not args.enable_wandb:
        return None
    
    try:
        import wandb
        
        # Prepare configuration
        wandb_config = prepare_wandb_config(args)
        
        # Setup tags
        tags = ["aimnet-x2d"]  # FIXED: was unclear naming
        if args.wandb_tags_list:
            tags.extend(args.wandb_tags_list)
        
        # Add automatic tags based on configuration
        tags.append(args.task_type)
        tags.append(args.pooling_type)
        if args.use_partial_charges:
            tags.append("partial_charges")
        if args.use_stereochemistry:
            tags.append("stereochemistry")
        if args.calculate_sae:
            tags.append("sae")
        
        # Initialize wandb run
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=wandb_config,
            tags=tags,
            name=get_experiment_id(args),
        )
        
        print(f"Weights & Biases logging initialized:")
        print(f"  Project: {args.wandb_project}")
        print(f"  Run: {run.name}")
        print(f"  URL: {run.url}")
        
        return run
        
    except ImportError:
        print("WARNING: Weights & Biases requested but not available. Install with:")
        print("  pip install wandb")
        return None
    except Exception as e:
        print(f"WARNING: Failed to initialize Weights & Biases: {e}")
        return None

def finalize_experiment_logging(wandb_run, results: Dict[str, Any]) -> None:
    """
    Finalize experiment logging by uploading final results and artifacts.
    
    Args:
        wandb_run: Wandb run object
        results: Final experiment results
    """
    if wandb_run is None:
        return
    
    try:
        import wandb
        
        # Log final metrics as summary
        if "test_metrics" in results:
            test_metrics = results["test_metrics"]
            summary_metrics = {f"final_test_{k}": v for k, v in test_metrics.items()}
            wandb_run.summary.update(summary_metrics)
        
        # Log model artifacts if available
        if "model_path" in results and os.path.exists(results["model_path"]):
            wandb.save(results["model_path"])
        
        # Log embedding artifacts if available
        if "embeddings_path" in results and os.path.exists(results["embeddings_path"]):
            wandb.save(results["embeddings_path"])
        
        # Finish the run
        wandb.finish()
        
    except Exception as e:
        print(f"WARNING: Failed to finalize Weights & Biases logging: {e}")


def print_final_summary(results: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Print a final summary of the experiment results.
    
    Args:
        results: Experiment results
        args: Command line arguments
    """
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    
    if "test_metrics" in results:
        test_metrics = results["test_metrics"]
        
        print("Final Test Results:")
        if args.task_type == 'multitask':
            print(f"  Loss: {test_metrics.get('loss', 'N/A'):.6f}")
            print(f"  MAE (avg): {test_metrics.get('mae', 'N/A'):.6f}")
            print(f"  RMSE (avg): {test_metrics.get('rmse', 'N/A'):.6f}")
            print(f"  R² (avg): {test_metrics.get('r2', 'N/A'):.6f}")
            
            # Per-task metrics if available
            if 'mae_per_target' in test_metrics and args.multi_target_columns_list:
                print("\n  Per-task Results:")
                for i, col_name in enumerate(args.multi_target_columns_list):
                    if i < len(test_metrics['mae_per_target']):
                        mae_i = test_metrics['mae_per_target'][i]
                        rmse_i = test_metrics['rmse_per_target'][i]
                        r2_i = test_metrics['r2_per_target'][i]
                        print(f"    {col_name}: MAE={mae_i:.6f}, RMSE={rmse_i:.6f}, R²={r2_i:.6f}")
        else:
            print(f"  Loss: {test_metrics.get('loss', 'N/A'):.6f}")
            print(f"  MAE: {test_metrics.get('mae', 'N/A'):.6f}")
            print(f"  RMSE: {test_metrics.get('rmse', 'N/A'):.6f}")
            print(f"  R²: {test_metrics.get('r2', 'N/A'):.6f}")
    
    # Training info
    if "training_time" in results:
        print(f"\nTraining time: {results['training_time']:.2f} seconds")
    
    if "best_epoch" in results:
        print(f"Best epoch: {results['best_epoch']}")
    
    # Output paths
    if "model_path" in results:
        print(f"\nModel saved to: {results['model_path']}")
    
    if "embeddings_path" in results and os.path.exists(results["embeddings_path"]):
        print(f"Embeddings saved to: {results['embeddings_path']}")
    
    print("="*80)