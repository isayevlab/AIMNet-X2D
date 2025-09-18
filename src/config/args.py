# config/args.py
"""
Command-line argument parsing for AIMNet-X2D with robust error handling.
"""

import argparse
import sys
from typing import Optional


class ArgumentError(Exception):
    """Raised when argument parsing fails."""
    pass


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with all options."""
    try:
        parser = argparse.ArgumentParser(
            description="GNN Molecular Property Predictor",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog="For more information, see the documentation."
        )

        # Add all argument groups
        _add_data_arguments(parser)
        _add_model_arguments(parser)
        _add_training_arguments(parser)
        _add_inference_arguments(parser)
        _add_system_arguments(parser)
        _add_logging_arguments(parser)
        
        return parser
        
    except Exception as e:
        raise ArgumentError(f"Failed to create argument parser: {e}")


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add data-related arguments."""
    data_group = parser.add_argument_group('Data Options')
    
    # Data paths
    data_group.add_argument("--data_path", type=str, 
                           help="Path to single CSV file")
    data_group.add_argument("--train_data", type=str, 
                           help="CSV file for train set")
    data_group.add_argument("--val_data", type=str, 
                           help="CSV file for val set")
    data_group.add_argument("--test_data", type=str, 
                           help="CSV file for test set")
    
    # Data splits
    data_group.add_argument("--train_split", type=float, default=0.8,
                           help="Fraction for training set")
    data_group.add_argument("--val_split", type=float, default=0.1,
                           help="Fraction for validation set")
    data_group.add_argument("--test_split", type=float, default=0.1,
                           help="Fraction for test set")
    
    # Column configuration
    data_group.add_argument("--smiles_column", type=str, default="smiles",
                           help="Column name for SMILES strings")
    data_group.add_argument("--target_column", type=str, default="target",
                           help="Column name for target values")
    data_group.add_argument("--multi_target_columns", type=str, default=None,
                           help="Comma-separated list of target columns for multitask")
    
    # Dataset configuration
    data_group.add_argument("--iterable_dataset", action="store_true",
                           help="Use HDF5 iterable dataset for large data")
    data_group.add_argument("--shuffle_buffer_size", type=int, default=1000,
                           help="Buffer size for shuffling iterable datasets")
    data_group.add_argument("--train_hdf5", type=str, default=None,
                           help="Path to train HDF5 file")
    data_group.add_argument("--val_hdf5", type=str, default=None,
                           help="Path to validation HDF5 file")
    data_group.add_argument("--test_hdf5", type=str, default="test.h5",
                           help="Path to test HDF5 file")


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model architecture arguments."""
    model_group = parser.add_argument_group('Model Architecture')
    
    # Core architecture
    model_group.add_argument("--hidden_dim", type=int, default=512,
                            help="Hidden dimension for the model")
    model_group.add_argument("--num_shells", type=int, default=3,
                            help="Number of shells/hops for message passing")
    model_group.add_argument("--num_message_passing_layers", type=int, default=3,
                            help="Number of message passing layers")
    model_group.add_argument("--embedding_dim", type=int, default=64,
                            help="Embedding dimension for atom features")
    
    # Feed-forward network
    model_group.add_argument("--ffn_hidden_dim", type=int, default=None,
                            help="Feed-forward network hidden dimension")
    model_group.add_argument("--ffn_num_layers", type=int, default=3,
                            help="Number of feed-forward layers")
    model_group.add_argument("--ffn_dropout", type=float, default=0.05,
                            help="Dropout rate for feed-forward layers")
    
    # Pooling and attention
    model_group.add_argument("--pooling_type", type=str, default="attention",
                            choices=["attention", "mean", "max", "sum"],
                            help="Type of graph pooling")
    model_group.add_argument("--attention_num_heads", type=int, default=4,
                            help="Number of attention heads")
    model_group.add_argument("--attention_temperature", type=float, default=1.0,
                            help="Initial temperature for attention pooling")
    
    # Layer configuration
    model_group.add_argument("--shell_conv_num_mlp_layers", type=int, default=2,
                            help="Number of MLP layers in shell convolution")
    model_group.add_argument("--shell_conv_dropout", type=float, default=0.05,
                            help="Dropout rate for shell convolution")
    model_group.add_argument("--activation_type", type=str, default="silu",
                            choices=["relu", "leakyrelu", "elu", "silu", "gelu"],
                            help="Type of activation function")
    
    # Features
    model_group.add_argument("--use_partial_charges", action="store_true",
                            help="Enable partial charge calculations")
    model_group.add_argument("--use_stereochemistry", action="store_true",
                            help="Enable stereochemical feature calculations")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add training-related arguments."""
    training_group = parser.add_argument_group('Training Options')
    
    # Basic training
    training_group.add_argument("--learning_rate", type=float, default=0.00025,
                               help="Learning rate for optimization")
    training_group.add_argument("--epochs", type=int, default=50,
                               help="Number of training epochs")
    training_group.add_argument("--batch_size", type=int, default=64,
                               help="Batch size for training")
    training_group.add_argument("--early_stopping", action="store_true",
                               help="Enable early stopping")
    training_group.add_argument("--patience", type=int, default=25,
                               help="Early stopping patience")
    
    # Task configuration
    training_group.add_argument("--task_type", type=str, default="regression",
                               choices=["regression", "multitask"],
                               help="Type of prediction task")
    training_group.add_argument("--loss_function", type=str, default="l1", 
                               choices=["l1", "mse"],
                               help="Loss function to use")
    training_group.add_argument("--multitask_weights", type=str, default=None,
                               help="Comma-separated weights for multitask learning")
    
    # Optimization
    training_group.add_argument("--lr_scheduler", type=str, default="ReduceLROnPlateau",
                               choices=[None, "ReduceLROnPlateau", "CosineAnnealingLR", 
                                       "StepLR", "ExponentialLR"],
                               help="Learning rate scheduler")
    training_group.add_argument("--lr_reduce_factor", type=float, default=0.5,
                               help="Factor for ReduceLROnPlateau scheduler")
    training_group.add_argument("--lr_patience", type=int, default=10,
                               help="Patience for ReduceLROnPlateau scheduler")
    training_group.add_argument("--lr_cosine_t_max", type=int, default=10,
                               help="T_max for CosineAnnealingLR scheduler")
    training_group.add_argument("--lr_step_size", type=int, default=10,
                               help="Step size for StepLR scheduler")
    training_group.add_argument("--lr_step_gamma", type=float, default=0.1,
                               help="Gamma for StepLR scheduler")
    training_group.add_argument("--lr_exp_gamma", type=float, default=0.95,
                               help="Gamma for ExponentialLR scheduler")
    
    # Transfer learning
    training_group.add_argument("--transfer_learning", type=str, default=None,
                               help="Path to pretrained model")
    training_group.add_argument("--freeze_pretrained", action="store_true",
                               help="Freeze pretrained layers except output")
    training_group.add_argument("--freeze_layers", type=str, default=None,
                               help="Comma-separated layer patterns to freeze")
    training_group.add_argument("--unfreeze_layers", type=str, default=None,
                               help="Comma-separated layer patterns to unfreeze")
    training_group.add_argument("--layer_wise_lr_decay", action="store_true",
                               help="Enable layer-wise learning rate decay")
    training_group.add_argument("--lr_decay_factor", type=float, default=0.8,
                               help="Decay factor for layer-wise learning rate")
    
    # SAE
    training_group.add_argument("--calculate_sae", action="store_true",
                               help="Calculate Size-Extensive Additive normalization")
    training_group.add_argument("--sae_subtasks", type=str, default=None,
                               help="Comma-separated subtask indices for SAE")


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    """Add inference-related arguments."""
    inference_group = parser.add_argument_group('Inference Options')
    
    # Input/output
    inference_group.add_argument("--inference_csv", type=str, default=None,
                                help="Path to CSV file for inference")
    inference_group.add_argument("--inference_hdf5", type=str, default=None,
                                help="Path to HDF5 file for inference")
    inference_group.add_argument("--inference_output", type=str, default="predictions.csv",
                                help="Output path for predictions")
    inference_group.add_argument("--inference_mode", type=str, 
                                choices=["streaming", "inmemory", "iterable"], default=None,
                                help="Mode for inference processing")
    
    # Uncertainty estimation
    inference_group.add_argument("--mc_samples", type=int, default=0,
                                help="Number of Monte Carlo dropout samples")
    inference_group.add_argument("--stream_chunk_size", type=int, default=1000,
                                help="Chunk size for streaming inference")
    inference_group.add_argument("--stream_batch_size", type=int, default=None,
                                help="Batch size for streaming inference")
    
    # Embeddings
    inference_group.add_argument("--save_embeddings", action="store_true",
                                help="Extract and save molecular embeddings")
    inference_group.add_argument("--embeddings_output_path", type=str, 
                                default="molecular_embeddings.h5",
                                help="Path to save extracted embeddings")
    inference_group.add_argument("--include_atom_embeddings", action="store_true",
                                help="Include atom-level embeddings")
    
    # Partial charges
    inference_group.add_argument("--output_partial_charges", type=str, default=None,
                                help="Path to save partial charges CSV")


def _add_system_arguments(parser: argparse.ArgumentParser) -> None:
    """Add system and performance arguments."""
    system_group = parser.add_argument_group('System Options')
    
    # Hardware
    system_group.add_argument("--num_workers", type=int, default=10,
                             help="Number of data loading workers")
    system_group.add_argument("--num_gpu_devices", type=int, default=1,
                             help="Number of GPU devices for training")
    system_group.add_argument("--mixed_precision", action="store_true",
                             help="Enable mixed precision training")
    system_group.add_argument("--precompute_num_workers", type=int, default=None,
                             help="Number of workers for feature precomputation")
    
    # Model management
    system_group.add_argument("--model_save_path", type=str, default="gnn_model.pth",
                             help="Path to save/load model")


def _add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add logging and experiment tracking arguments."""
    logging_group = parser.add_argument_group('Logging & Tracking')
    
    # Experiment management
    logging_group.add_argument("--hyperparameter_file", type=str, default=None,
                              help="YAML file with hyperparameter configurations")
    logging_group.add_argument("--num_trials", type=int, default=1,
                              help="Number of hyperparameter tuning trials")
    
    # Wandb
    logging_group.add_argument("--enable_wandb", action="store_true",
                              help="Enable Weights & Biases logging")
    logging_group.add_argument("--wandb_project", type=str, 
                              default="gnn-molecular-property-prediction",
                              help="Wandb project name")


def parse_arguments(args: Optional[list] = None) -> argparse.Namespace:
    """
    Parse command-line arguments with robust error handling.
    
    Args:
        args: Optional list of arguments (for testing)
        
    Returns:
        Parsed arguments namespace
        
    Raises:
        ArgumentError: If argument parsing fails
    """
    try:
        parser = create_argument_parser()
        
        if args is None:
            parsed_args = parser.parse_args()
        else:
            parsed_args = parser.parse_args(args)
            
        return parsed_args
        
    except SystemExit as e:
        # argparse calls sys.exit() on error, catch and convert to exception
        if e.code != 0:
            raise ArgumentError(f"Argument parsing failed with exit code {e.code}")
        raise
        
    except Exception as e:
        raise ArgumentError(f"Unexpected error during argument parsing: {e}")


def print_arguments(args: argparse.Namespace) -> None:
    """Print arguments in a formatted way for debugging."""
    try:
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        
        # Group arguments by their parser groups
        arg_dict = vars(args)
        
        # Data arguments
        data_args = {k: v for k, v in arg_dict.items() 
                    if k in ['data_path', 'train_data', 'val_data', 'test_data', 
                            'train_split', 'val_split', 'test_split',
                            'smiles_column', 'target_column', 'multi_target_columns',
                            'iterable_dataset', 'shuffle_buffer_size']}
        if data_args:
            print("Data Configuration:")
            for k, v in data_args.items():
                print(f"  {k}: {v}")
            print()
        
        # Model arguments  
        model_args = {k: v for k, v in arg_dict.items()
                     if k in ['hidden_dim', 'num_shells', 'num_message_passing_layers',
                             'pooling_type', 'activation_type', 'use_partial_charges',
                             'use_stereochemistry']}
        if model_args:
            print("Model Configuration:")
            for k, v in model_args.items():
                print(f"  {k}: {v}")
            print()
        
        # Training arguments
        training_args = {k: v for k, v in arg_dict.items()
                        if k in ['learning_rate', 'epochs', 'batch_size', 'task_type',
                                'loss_function', 'early_stopping', 'calculate_sae']}
        if training_args:
            print("Training Configuration:")
            for k, v in training_args.items():
                print(f"  {k}: {v}")
            print()
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Warning: Could not print arguments: {e}")


def validate_argument_types(args: argparse.Namespace) -> None:
    """
    Validate argument types and ranges with robust error handling.
    
    Args:
        args: Parsed arguments
        
    Raises:
        ArgumentError: If validation fails
    """
    try:
        errors = []
        
        # Validate positive integers
        positive_int_args = ['epochs', 'batch_size', 'hidden_dim', 'num_shells',
                            'num_message_passing_layers', 'embedding_dim', 'num_workers']
        for arg_name in positive_int_args:
            value = getattr(args, arg_name, None)
            if value is not None and value <= 0:
                errors.append(f"{arg_name} must be positive, got {value}")
        
        # Validate probabilities (0-1 range)
        prob_args = ['train_split', 'val_split', 'test_split', 'ffn_dropout', 
                    'shell_conv_dropout', 'lr_reduce_factor']
        for arg_name in prob_args:
            value = getattr(args, arg_name, None)
            if value is not None and not (0 <= value <= 1):
                errors.append(f"{arg_name} must be between 0 and 1, got {value}")
        
        # Validate positive floats
        positive_float_args = ['learning_rate', 'attention_temperature']
        for arg_name in positive_float_args:
            value = getattr(args, arg_name, None)
            if value is not None and value <= 0:
                errors.append(f"{arg_name} must be positive, got {value}")
        
        if errors:
            raise ArgumentError("Argument validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
            
    except Exception as e:
        if isinstance(e, ArgumentError):
            raise
        else:
            raise ArgumentError(f"Unexpected error during argument validation: {e}")