"""
Command-line interface for AIMNet-X2D.

Handles argument parsing and validation for the main execution.
"""

import argparse
from typing import Optional


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all options."""
    parser = argparse.ArgumentParser(
        description="AIMNet-X2D: Advanced Graph Neural Network for Molecular Property Prediction",  # FIXED: Clear name
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="For detailed documentation, visit the project repository."
    )


    # Add argument groups
    _add_data_arguments(parser)
    _add_model_arguments(parser)
    _add_training_arguments(parser)
    _add_inference_arguments(parser)
    _add_system_arguments(parser)
    _add_hyperopt_arguments(parser)
    _add_logging_arguments(parser)

    return parser


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add data-related arguments."""
    data_group = parser.add_argument_group('Data Configuration')
    
    # Data paths
    data_group.add_argument("--data_path", type=str, 
                           help="Path to single CSV file for train/val/test split")
    data_group.add_argument("--train_data", type=str, 
                           help="Path to training CSV file")
    data_group.add_argument("--val_data", type=str, 
                           help="Path to validation CSV file")
    data_group.add_argument("--test_data", type=str, 
                           help="Path to test CSV file")
    
    # Data splits
    data_group.add_argument("--train_split", type=float, default=0.8,
                           help="Fraction for training set (when using --data_path)")
    data_group.add_argument("--val_split", type=float, default=0.1,
                           help="Fraction for validation set (when using --data_path)")
    data_group.add_argument("--test_split", type=float, default=0.1,
                           help="Fraction for test set (when using --data_path)")
    
    # Column configuration
    data_group.add_argument("--smiles_column", type=str, default="smiles",
                           help="Column name for SMILES strings")
    data_group.add_argument("--target_column", type=str, default="target",
                           help="Column name for target values (single-task)")
    data_group.add_argument("--multi_target_columns", type=str, default=None,
                           help="Comma-separated target columns for multi-task")
    
    # HDF5 configuration
    data_group.add_argument("--iterable_dataset", action="store_true",
                           help="Use HDF5 iterable dataset for large data")
    data_group.add_argument("--shuffle_buffer_size", type=int, default=1000,
                           help="Buffer size for shuffling iterable datasets")
    data_group.add_argument("--train_hdf5", type=str, default=None,
                           help="Path to training HDF5 file")
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
    training_group = parser.add_argument_group('Training Configuration')
    
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
                               choices=["l1", "mse", "evidential"],
                               help="Loss function to use")
    training_group.add_argument("--multitask_weights", type=str, default=None,
                               help="Comma-separated weights for multitask learning")
    
    # Evidential loss parameters
    training_group.add_argument("--evidential_lambda", type=float, default=1.0,
                               help="Regularization strength for evidential loss")
    
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
    inference_group = parser.add_argument_group('Inference Configuration')
    
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
    system_group = parser.add_argument_group('System Configuration')
    
    # Hardware
    system_group.add_argument("--num_workers", type=int, default=4,
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


def _add_hyperopt_arguments(parser: argparse.ArgumentParser) -> None:
    """Add hyperparameter optimization arguments."""
    hyperopt_group = parser.add_argument_group('Hyperparameter Optimization')
    
    # Legacy hyperparameter support
    hyperopt_group.add_argument("--hyperparameter_file", type=str, default=None,
                               help="YAML file with hyperparameter configurations")
    hyperopt_group.add_argument("--num_trials", type=int, default=1,
                               help="Number of hyperparameter trials")


# Fix 2a: In main/cli.py - fix project name
def _add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add logging and experiment tracking arguments."""
    logging_group = parser.add_argument_group('Logging & Tracking')
    
    # Weights & Biases
    logging_group.add_argument("--enable_wandb", action="store_true",
                              help="Enable Weights & Biases logging")
    logging_group.add_argument("--wandb_project", type=str, 
                              default="aimnet-x2d-molecular-prediction",  # FIXED: was "gnn-molecular-property-prediction"
                              help="Weights & Biases project name")
    logging_group.add_argument("--wandb_entity", type=str, default=None,
                              help="Weights & Biases entity/team name")
    logging_group.add_argument("--wandb_tags", type=str, default=None,
                              help="Comma-separated tags for Weights & Biases")



def parse_main_arguments(args: Optional[list] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the main execution.
    
    Args:
        args: Optional list of arguments (for testing)
        
    Returns:
        Parsed arguments namespace
    """
    parser = create_main_parser()
    
    if args is None:
        parsed_args = parser.parse_args()
    else:
        parsed_args = parser.parse_args(args)
    
    # Post-process arguments
    _postprocess_arguments(parsed_args)
    
    return parsed_args


def _postprocess_arguments(args: argparse.Namespace) -> None:
    """Post-process parsed arguments for consistency and validation."""
    # Set ffn_hidden_dim to hidden_dim if not specified
    if args.ffn_hidden_dim is None:
        args.ffn_hidden_dim = args.hidden_dim
    
    # Set precompute_num_workers to num_workers if not specified
    if args.precompute_num_workers is None:
        args.precompute_num_workers = args.num_workers
    
    # Set stream_batch_size to batch_size if not specified
    if args.stream_batch_size is None:
        args.stream_batch_size = args.batch_size
    
    # Process multi-target columns
    if args.multi_target_columns:
        args.multi_target_columns_list = [c.strip() for c in args.multi_target_columns.split(',')]
    else:
        args.multi_target_columns_list = None
    
    # Process multitask weights
    if args.multitask_weights:
        args.multitask_weights_list = [float(w.strip()) for w in args.multitask_weights.split(',')]
    else:
        args.multitask_weights_list = None
    
    # Process SAE subtasks
    if args.sae_subtasks:
        args.sae_subtasks_list = [int(x.strip()) for x in args.sae_subtasks.split(',')]
    else:
        args.sae_subtasks_list = None
    
    # Process wandb tags
    if args.wandb_tags:
        args.wandb_tags_list = [tag.strip() for tag in args.wandb_tags.split(',')]
    else:
        args.wandb_tags_list = None
    
    # Set inference mode based on inputs if not specified
    if args.inference_mode is None:
        if args.inference_csv:
            args.inference_mode = "streaming"
        elif args.inference_hdf5:
            args.inference_mode = "iterable"


def print_configuration(args: argparse.Namespace) -> None:
    """Print the current configuration in a formatted way."""
    print("=" * 80)
    print("AIMNET-X2D CONFIGURATION")
    print("=" * 80)
    
    # Data configuration
    print("Data Configuration:")
    if args.data_path:
        print(f"  Single file: {args.data_path}")
        print(f"  Train/Val/Test splits: {args.train_split}/{args.val_split}/{args.test_split}")
    else:
        print(f"  Train: {args.train_data}")
        print(f"  Validation: {args.val_data}")
        print(f"  Test: {args.test_data}")
    
    print(f"  SMILES column: {args.smiles_column}")
    if args.task_type == "multitask":
        print(f"  Target columns: {args.multi_target_columns}")
    else:
        print(f"  Target column: {args.target_column}")
    print()
    
    # Model configuration
    print("Model Configuration:")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of shells: {args.num_shells}")
    print(f"  Message passing layers: {args.num_message_passing_layers}")
    print(f"  Pooling type: {args.pooling_type}")
    print(f"  Activation: {args.activation_type}")
    print(f"  Partial charges: {args.use_partial_charges}")
    print(f"  Stereochemistry: {args.use_stereochemistry}")
    print()
    
    # Training configuration
    print("Training Configuration:")
    print(f"  Task type: {args.task_type}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Loss function: {args.loss_function}")
    if args.loss_function == "evidential":
        print(f"  Evidential lambda: {args.evidential_lambda}")
    print(f"  Early stopping: {args.early_stopping}")
    if args.early_stopping:
        print(f"  Patience: {args.patience}")
    print()
    
    # Hyperparameter optimization
    if args.hyperparameter_file and args.num_trials > 1:
        print("Hyperparameter Optimization:")
        print(f"  Configuration file: {args.hyperparameter_file}")
        print(f"  Number of trials: {args.num_trials}")
        print()
    
    # System configuration
    print("System Configuration:")
    print(f"  GPU devices: {args.num_gpu_devices}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Iterable dataset: {args.iterable_dataset}")
    
    print("=" * 80)