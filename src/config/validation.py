# config/validation.py
"""
Argument validation for AIMNet-X2D configuration with robust error handling.
"""

import os
import warnings
from typing import List, Dict, Any, Optional
from pathlib import Path


class ValidationError(Exception):
    """Raised when argument validation fails."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass

def validate_args(args) -> bool:
    """
    Validates command-line arguments with HELPFUL error messages.
    
    FIXED: Clear, actionable error messages for common user mistakes.
    """
    try:
        errors = []
        warnings_list = []
        
        # Validate different aspects
        _validate_data_config(args, errors, warnings_list)
        _validate_task_config(args, errors, warnings_list)
        _validate_model_config(args, errors, warnings_list)
        _validate_system_config(args, errors, warnings_list)
        _validate_training_config(args, errors, warnings_list)
        _validate_inference_config(args, errors, warnings_list)
        
        # Print warnings
        for warning in warnings_list:
            warnings.warn(f"⚠️  {warning}", UserWarning)
        
        # Raise errors if any - with HELPFUL formatting
        if errors:
            error_msg = "\n" + "="*80 + "\n"
            error_msg += "❌ CONFIGURATION ERRORS FOUND\n"
            error_msg += "="*80 + "\n\n"
            error_msg += f"Found {len(errors)} error(s) that must be fixed:\n\n"
            
            for i, error in enumerate(errors, 1):
                error_msg += f"{i}. {error}\n\n"
            
            error_msg += "="*80 + "\n"
            error_msg += "💡 QUICK FIXES:\n"
            error_msg += "="*80 + "\n"
            
            # Add context-specific hints
            if any("data_path" in e or "train_data" in e for e in errors):
                error_msg += "• For data issues: Make sure your CSV files exist and have the right columns\n"
                error_msg += "  Example: --data_path data.csv --smiles_column SMILES --target_column Activity\n\n"
            
            if any("multitask" in e.lower() for e in errors):
                error_msg += "• For multitask: Specify target columns like this:\n"
                error_msg += "  --task_type multitask --multi_target_columns prop1,prop2,prop3\n\n"
            
            if any("split" in e.lower() for e in errors):
                error_msg += "• For split ratios: They must sum to 1.0\n"
                error_msg += "  Example: --train_split 0.8 --val_split 0.1 --test_split 0.1\n\n"
            
            error_msg += f"\nFor more help, see documentation or run: python main.py --help\n"
            error_msg += "="*80 + "\n"
            
            raise ValidationError(error_msg)
        
        if warnings_list:
            print(f"✅ Configuration valid with {len(warnings_list)} warning(s)")
        else:
            print("✅ Configuration valid")
        
        return True
        
    except (ValidationError, ConfigurationError):
        raise
    except Exception as e:
        raise ConfigurationError(
            f"Unexpected error during validation: {e}\n\n"
            f"This is likely a bug. Please report it with the full error message."
        )


def _validate_data_config(args, errors: List[str], warnings: List[str]) -> None:
    """Validate data configuration with enhanced error checking."""
    try:
        # Check data loading paths
        if args.data_path:
            if args.train_data or args.val_data or args.test_data:
                errors.append("Cannot specify both --data_path and individual train/val/test dataset paths.")
            
            # Validate data_path exists
            if not os.path.exists(args.data_path):
                errors.append(f"Data file not found: {args.data_path}")
            elif not args.data_path.endswith('.csv'):
                warnings.append(f"Data file should be CSV format: {args.data_path}")
            
            # Check split ratios
            total_split = args.train_split + args.val_split + args.test_split
            if not (0.999 <= total_split <= 1.001):  # Allow for floating point imprecision
                errors.append(f"Split ratios must sum to 1.0, got {total_split:.3f}")
                
            # Check individual split values
            if args.train_split <= 0:
                errors.append(f"Train split must be positive, got {args.train_split}")
            if args.val_split <= 0:
                warnings.append(f"Validation split is {args.val_split}, consider using a positive value")
            if args.test_split <= 0:
                warnings.append(f"Test split is {args.test_split}, consider using a positive value")
                
        elif not (args.train_data and args.val_data and args.test_data) and not (args.inference_csv or args.inference_hdf5):
            errors.append("Must specify either --data_path or all of --train_data, --val_data, and --test_data when not in inference mode.")
        
        # Validate individual data files if specified
        for data_file, name in [(args.train_data, "train"), (args.val_data, "val"), (args.test_data, "test")]:
            if data_file:
                if not os.path.exists(data_file):
                    errors.append(f"{name.capitalize()} data file not found: {data_file}")
                elif not data_file.endswith('.csv'):
                    warnings.append(f"{name.capitalize()} data file should be CSV format: {data_file}")
        
        # Check HDF5 configuration
        if args.iterable_dataset:
            if not (args.train_hdf5 and args.val_hdf5 and args.test_hdf5):
                errors.append("When using --iterable_dataset, must specify --train_hdf5, --val_hdf5, and --test_hdf5.")
            
            # Check HDF5 paths are valid
            for hdf5_file, name in [(args.train_hdf5, "train"), (args.val_hdf5, "val"), (args.test_hdf5, "test")]:
                if hdf5_file:
                    if not hdf5_file.endswith(('.h5', '.hdf5')):
                        warnings.append(f"{name.capitalize()} HDF5 file should have .h5 or .hdf5 extension: {hdf5_file}")
                    
                    # Create parent directory if it doesn't exist
                    parent_dir = Path(hdf5_file).parent
                    if not parent_dir.exists():
                        warnings.append(f"Parent directory for {name} HDF5 will be created: {parent_dir}")
        
        # Validate column names
        if args.smiles_column and not args.smiles_column.strip():
            errors.append("SMILES column name cannot be empty")
        if args.target_column and not args.target_column.strip():
            errors.append("Target column name cannot be empty")
            
    except Exception as e:
        errors.append(f"Error validating data configuration: {e}")


def _validate_task_config(args, errors: List[str], warnings: List[str]) -> None:
    """Validate task configuration with enhanced checking."""
    try:
        if args.task_type == 'multitask':
            if not args.multi_target_columns:
                errors.append("For --task_type=multitask, must specify --multi_target_columns.")
            else:
                # Parse and validate multi target columns
                try:
                    multi_cols = [c.strip() for c in args.multi_target_columns.split(',')]
                    if len(multi_cols) < 2:
                        errors.append("Multitask requires at least 2 target columns")
                    
                    # Check for duplicate columns
                    if len(set(multi_cols)) != len(multi_cols):
                        errors.append("Duplicate target columns found in --multi_target_columns")
                    
                    # Check for empty column names
                    if any(not col for col in multi_cols):
                        errors.append("Empty column names found in --multi_target_columns")
                        
                except Exception as e:
                    errors.append(f"Error parsing --multi_target_columns: {e}")
            
            # Validate multitask weights
            if args.multitask_weights:
                try:
                    weights = [float(w.strip()) for w in args.multitask_weights.split(',')]
                    multi_cols = [c.strip() for c in args.multi_target_columns.split(',')]
                    
                    if len(weights) != len(multi_cols):
                        errors.append(f"Number of multitask weights ({len(weights)}) must match number of target columns ({len(multi_cols)}).")
                    
                    # Check for negative weights
                    if any(w < 0 for w in weights):
                        errors.append("Multitask weights cannot be negative")
                    
                    # Check for all zero weights
                    if all(w == 0 for w in weights):
                        errors.append("All multitask weights cannot be zero")
                        
                except ValueError as e:
                    errors.append(f"Error parsing multitask weights: {e}")
                except Exception as e:
                    errors.append(f"Unexpected error with multitask weights: {e}")
        
        # Check SAE configuration
        if args.calculate_sae:
            if args.task_type == 'multitask' and not args.sae_subtasks:
                warnings.append("SAE calculation enabled for multitask, but no --sae_subtasks specified. SAE will be skipped.")
            elif args.sae_subtasks:
                try:
                    subtasks = [int(x.strip()) for x in args.sae_subtasks.split(',')]
                    
                    if any(st < 0 for st in subtasks):
                        errors.append("SAE subtask indices cannot be negative")
                    
                    # Check for duplicates
                    if len(set(subtasks)) != len(subtasks):
                        warnings.append("Duplicate SAE subtask indices found")
                        
                    if args.task_type == 'multitask' and args.multi_target_columns:
                        multi_cols = [c.strip() for c in args.multi_target_columns.split(',')]
                        max_idx = len(multi_cols) - 1
                        if any(st > max_idx for st in subtasks):
                            errors.append(f"SAE subtask indices exceed available targets (max index: {max_idx})")
                            
                except ValueError as e:
                    errors.append(f"Error parsing SAE subtasks: {e}")
                except Exception as e:
                    errors.append(f"Unexpected error with SAE subtasks: {e}")
                    
    except Exception as e:
        errors.append(f"Error validating task configuration: {e}")


def _validate_model_config(args, errors: List[str], warnings: List[str]) -> None:
    """Validate model configuration with enhanced checking."""
    try:
        # Check transfer learning configuration
        if args.transfer_learning:
            if not os.path.exists(args.transfer_learning):
                errors.append(f"Transfer learning model file not found: {args.transfer_learning}")
            elif not args.transfer_learning.endswith('.pth'):
                warnings.append(f"Transfer learning model should be a .pth file: {args.transfer_learning}")
            
            if args.freeze_layers and args.unfreeze_layers:
                warnings.append("Both --freeze_layers and --unfreeze_layers specified. --unfreeze_layers will take precedence for overlapping patterns.")
        
        # Validate model dimensions
        if args.hidden_dim <= 0:
            errors.append(f"Hidden dimension must be positive, got {args.hidden_dim}")
        elif args.hidden_dim < 32:
            warnings.append(f"Hidden dimension is quite small ({args.hidden_dim}), consider increasing it")
        elif args.hidden_dim > 2048:
            warnings.append(f"Hidden dimension is very large ({args.hidden_dim}), this may cause memory issues")
        
        if args.embedding_dim <= 0:
            errors.append(f"Embedding dimension must be positive, got {args.embedding_dim}")
        elif args.embedding_dim > args.hidden_dim:
            warnings.append(f"Embedding dimension ({args.embedding_dim}) is larger than hidden dimension ({args.hidden_dim})")
        
        # Validate layer counts
        if args.num_message_passing_layers <= 0:
            errors.append(f"Number of message passing layers must be positive, got {args.num_message_passing_layers}")
        elif args.num_message_passing_layers > 10:
            warnings.append(f"Many message passing layers ({args.num_message_passing_layers}) may cause gradient issues")
        
        if args.ffn_num_layers <= 0:
            errors.append(f"Number of FFN layers must be positive, got {args.ffn_num_layers}")
        
        # Validate attention parameters
        if args.attention_num_heads <= 0:
            errors.append(f"Number of attention heads must be positive, got {args.attention_num_heads}")
        elif args.hidden_dim % args.attention_num_heads != 0:
            warnings.append(f"Hidden dimension ({args.hidden_dim}) should be divisible by number of attention heads ({args.attention_num_heads})")
        
        if args.attention_temperature <= 0:
            errors.append(f"Attention temperature must be positive, got {args.attention_temperature}")
            
    except Exception as e:
        errors.append(f"Error validating model configuration: {e}")


def _validate_system_config(args, errors: List[str], warnings: List[str]) -> None:
    """Validate system configuration with enhanced checking."""
    try:
        # Check hardware configuration
        if args.num_workers < 0:
            errors.append(f"Number of workers cannot be negative, got {args.num_workers}")
        elif args.num_workers > 16:
            warnings.append(f"Large number of workers ({args.num_workers}) may cause memory issues")
        
        if args.num_gpu_devices < 0:
            errors.append(f"Number of GPU devices cannot be negative, got {args.num_gpu_devices}")
        elif args.num_gpu_devices > 8:
            warnings.append(f"Very large number of GPUs ({args.num_gpu_devices}), ensure you have them available")
        
        # Check DDP configuration
        if args.num_gpu_devices > 1:
            if args.num_workers > 4:
                warnings.append("Using multiple GPUs with many workers may cause issues with some PyTorch versions")
            
            if args.batch_size < args.num_gpu_devices:
                warnings.append(f"Batch size ({args.batch_size}) is smaller than number of GPUs ({args.num_gpu_devices})")
        
        # Validate model save path
        if args.model_save_path:
            save_dir = os.path.dirname(args.model_save_path)
            if save_dir and not os.path.exists(save_dir):
                warnings.append(f"Model save directory will be created: {save_dir}")
            
            if not args.model_save_path.endswith('.pth'):
                warnings.append(f"Model save path should have .pth extension: {args.model_save_path}")
        
        # Check precompute workers
        if args.precompute_num_workers is not None:
            if args.precompute_num_workers <= 0:
                errors.append(f"Precompute workers must be positive, got {args.precompute_num_workers}")
            elif args.precompute_num_workers > 32:
                warnings.append(f"Very large number of precompute workers ({args.precompute_num_workers})")
                
    except Exception as e:
        errors.append(f"Error validating system configuration: {e}")


def _validate_training_config(args, errors: List[str], warnings: List[str]) -> None:
    """Validate training configuration with enhanced checking."""
    try:
        # Basic training parameters
        if args.learning_rate <= 0:
            errors.append(f"Learning rate must be positive, got {args.learning_rate}")
        elif args.learning_rate > 0.1:
            warnings.append(f"Learning rate is quite high ({args.learning_rate}), consider reducing it")
        elif args.learning_rate < 1e-6:
            warnings.append(f"Learning rate is very small ({args.learning_rate}), training may be slow")
        
        if args.epochs < 0:
            errors.append(f"Number of epochs must be positive, got {args.epochs}")
        elif args.epochs > 1000:
            warnings.append(f"Very large number of epochs ({args.epochs}), consider using early stopping")
        
        if args.batch_size <= 0:
            errors.append(f"Batch size must be positive, got {args.batch_size}")
        elif args.batch_size > 1024:
            warnings.append(f"Very large batch size ({args.batch_size}) may cause memory issues")
        elif args.batch_size < 8:
            warnings.append(f"Small batch size ({args.batch_size}) may lead to unstable training")
        
        # Early stopping configuration
        if args.early_stopping:
            if args.patience <= 0:
                errors.append(f"Early stopping patience must be positive, got {args.patience}")
            elif args.patience >= args.epochs:
                warnings.append(f"Early stopping patience ({args.patience}) is >= epochs ({args.epochs})")
        
        # Learning rate scheduler validation
        if args.lr_scheduler:
            if args.lr_scheduler == "ReduceLROnPlateau":
                if args.lr_reduce_factor <= 0 or args.lr_reduce_factor >= 1:
                    errors.append(f"LR reduce factor must be between 0 and 1, got {args.lr_reduce_factor}")
                if args.lr_patience <= 0:
                    errors.append(f"LR patience must be positive, got {args.lr_patience}")
            
            elif args.lr_scheduler == "StepLR":
                if args.lr_step_size <= 0:
                    errors.append(f"LR step size must be positive, got {args.lr_step_size}")
                if args.lr_step_gamma <= 0 or args.lr_step_gamma >= 1:
                    errors.append(f"LR step gamma must be between 0 and 1, got {args.lr_step_gamma}")
            
            elif args.lr_scheduler == "CosineAnnealingLR":
                if args.lr_cosine_t_max <= 0:
                    errors.append(f"LR cosine T_max must be positive, got {args.lr_cosine_t_max}")
            
            elif args.lr_scheduler == "ExponentialLR":
                if args.lr_exp_gamma <= 0 or args.lr_exp_gamma >= 1:
                    errors.append(f"LR exponential gamma must be between 0 and 1, got {args.lr_exp_gamma}")
        
        # Layer-wise learning rate decay
        if args.layer_wise_lr_decay:
            if args.lr_decay_factor <= 0 or args.lr_decay_factor >= 1:
                errors.append(f"LR decay factor must be between 0 and 1, got {args.lr_decay_factor}")
                
    except Exception as e:
        errors.append(f"Error validating training configuration: {e}")


def _validate_inference_config(args, errors: List[str], warnings: List[str]) -> None:
    """Validate inference configuration with enhanced checking."""
    try:
        # Check inference mode
        if args.inference_csv or args.inference_hdf5:
            if not args.model_save_path:
                errors.append("Must specify --model_save_path when running in inference mode")
            elif not os.path.exists(args.model_save_path):
                errors.append(f"Model file not found for inference: {args.model_save_path}")
            
            # Validate input files
            if args.inference_csv:
                if not os.path.exists(args.inference_csv):
                    errors.append(f"Inference CSV file not found: {args.inference_csv}")
                elif not args.inference_csv.endswith('.csv'):
                    warnings.append(f"Inference input should be CSV format: {args.inference_csv}")
            
            if args.inference_hdf5:
                if not os.path.exists(args.inference_hdf5):
                    errors.append(f"Inference HDF5 file not found: {args.inference_hdf5}")
                elif not args.inference_hdf5.endswith(('.h5', '.hdf5')):
                    warnings.append(f"Inference input should be HDF5 format: {args.inference_hdf5}")
            
            # Validate output path
            if not args.inference_output:
                warnings.append("No --inference_output specified. Results will be saved to 'predictions.csv'")
            else:
                output_dir = os.path.dirname(args.inference_output)
                if output_dir and not os.path.exists(output_dir):
                    warnings.append(f"Inference output directory will be created: {output_dir}")
        
        # Monte Carlo dropout validation
        if args.mc_samples < 0:
            errors.append(f"MC samples cannot be negative, got {args.mc_samples}")
        elif args.mc_samples > 100:
            warnings.append(f"Large number of MC samples ({args.mc_samples}) will be slow")
        
        # Streaming configuration
        if args.stream_chunk_size <= 0:
            errors.append(f"Stream chunk size must be positive, got {args.stream_chunk_size}")
        elif args.stream_chunk_size > 10000:
            warnings.append(f"Large stream chunk size ({args.stream_chunk_size}) may cause memory issues")
        
        # Embeddings validation
        if args.save_embeddings:
            if not args.embeddings_output_path:
                warnings.append("No --embeddings_output_path specified. Embeddings will be saved to 'molecular_embeddings.h5'")
            else:
                emb_dir = os.path.dirname(args.embeddings_output_path)
                if emb_dir and not os.path.exists(emb_dir):
                    warnings.append(f"Embeddings output directory will be created: {emb_dir}")
                
                if not args.embeddings_output_path.endswith(('.h5', '.hdf5')):
                    warnings.append(f"Embeddings output should be HDF5 format: {args.embeddings_output_path}")
        
        # Hyperparameter tuning validation
        if args.hyperparameter_file:
            if not os.path.exists(args.hyperparameter_file):
                errors.append(f"Hyperparameter file not found: {args.hyperparameter_file}")
            elif not args.hyperparameter_file.endswith(('.yaml', '.yml')):
                warnings.append(f"Hyperparameter file should be YAML format: {args.hyperparameter_file}")
            
            if args.num_trials <= 1:
                warnings.append("Hyperparameter file specified but --num_trials is 1. Only one set will be used")
        
        # Wandb configuration
        if args.enable_wandb:
            if not args.wandb_project:
                warnings.append("Wandb enabled but no --wandb_project specified. Using default project name")
            elif not args.wandb_project.strip():
                errors.append("Wandb project name cannot be empty")
                
    except Exception as e:
        errors.append(f"Error validating inference configuration: {e}")


def validate_file_permissions(file_path: str) -> bool:
    """
    Validate that we can read/write to a file path.
    
    Args:
        file_path: Path to validate
        
    Returns:
        bool: True if permissions are valid
    """
    try:
        # Check if we can create the parent directory
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        # Try to create/write to the file
        with open(file_path, 'a'):
            pass
        return True
        
    except (OSError, IOError, PermissionError):
        return False


def check_dependencies() -> List[str]:
    """
    Check if required dependencies are available.
    
    Returns:
        List of missing dependencies
    """
    missing_deps = []
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn', 'rdkit', 
        'h5py', 'yaml', 'tqdm', 'numba'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_deps.append(package)
    
    return missing_deps