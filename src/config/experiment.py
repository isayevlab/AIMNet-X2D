# config/experiment.py
"""
Experiment configuration management with robust error handling.
"""

import yaml
import argparse
import os
import json
from datetime import datetime
from typing import Any
from pathlib import Path

from utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentError(Exception):
    """Raised when experiment configuration operations fail."""
    pass


def save_experiment_config(args: argparse.Namespace, filepath: str | None = None) -> str:
    """
    Save experiment configuration to a YAML file with robust error handling.

    Args:
        args: Experiment arguments to save
        filepath: Optional custom filepath

    Returns:
        str: Path to saved configuration file

    Raises:
        ExperimentError: If saving fails
    """
    try:
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = getattr(args, 'experiment_name', "experiment")
            filepath = f"{exp_name}_{timestamp}_config.yaml"

        # Ensure parent directory exists
        parent_dir = Path(filepath).parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Convert args to dictionary, handling special types
        config = _serialize_args(args)

        # Add metadata
        config['_metadata'] = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            'type': 'experiment_config'
        }

        # Write to file with error handling
        try:
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=True)
        except IOError as e:
            raise ExperimentError(f"Failed to write config file {filepath}: {e}") from e
        except yaml.YAMLError as e:
            raise ExperimentError(f"Failed to serialize config to YAML: {e}") from e

        logger.info(f"Experiment configuration saved to {filepath}")
        return filepath

    except ExperimentError:
        raise
    except Exception as e:
        raise ExperimentError(f"Unexpected error saving experiment config: {e}") from e


def load_experiment_config(filepath: str) -> argparse.Namespace:
    """
    Load experiment configuration from a YAML file with robust error handling.

    Args:
        filepath: Path to configuration file

    Returns:
        argparse.Namespace: Loaded configuration

    Raises:
        ExperimentError: If loading fails
    """
    try:
        if not os.path.exists(filepath):
            raise ExperimentError(f"Configuration file not found: {filepath}")

        # Load and validate file
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
        except IOError as e:
            raise ExperimentError(f"Failed to read config file {filepath}: {e}") from e
        except yaml.YAMLError as e:
            raise ExperimentError(f"Failed to parse YAML config: {e}") from e

        if not isinstance(config, dict):
            raise ExperimentError(f"Invalid config format in {filepath}: expected dictionary")

        # Remove metadata if present
        config.pop('_metadata', None)

        # Convert dictionary to argparse Namespace with type restoration
        args = _deserialize_args(config)

        logger.info(f"Experiment configuration loaded from {filepath}")
        return args

    except ExperimentError:
        raise
    except Exception as e:
        raise ExperimentError(f"Unexpected error loading experiment config: {e}") from e


def create_experiment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """
    Create comprehensive experiment metadata dictionary.

    Args:
        args: Experiment arguments

    Returns:
        Dictionary of experiment metadata
    """
    try:
        metadata = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'task_type': getattr(args, 'task_type', 'unknown'),
                'model_save_path': getattr(args, 'model_save_path', None),
            },
            'model_architecture': {
                'hidden_dim': getattr(args, 'hidden_dim', None),
                'num_shells': getattr(args, 'num_shells', None),
                'num_message_passing_layers': getattr(args, 'num_message_passing_layers', None),
                'pooling_type': getattr(args, 'pooling_type', None),
                'activation_type': getattr(args, 'activation_type', None),
                'embedding_dim': getattr(args, 'embedding_dim', None),
            },
            'training_config': {
                'learning_rate': getattr(args, 'learning_rate', None),
                'epochs': getattr(args, 'epochs', None),
                'batch_size': getattr(args, 'batch_size', None),
                'loss_function': getattr(args, 'loss_function', None),
                'early_stopping': getattr(args, 'early_stopping', False),
                'patience': getattr(args, 'patience', None),
            },
            'data_config': {
                'data_path': getattr(args, 'data_path', None),
                'train_data': getattr(args, 'train_data', None),
                'val_data': getattr(args, 'val_data', None),
                'test_data': getattr(args, 'test_data', None),
                'smiles_column': getattr(args, 'smiles_column', None),
                'target_column': getattr(args, 'target_column', None),
                'multi_target_columns': getattr(args, 'multi_target_columns', None),
            },
            'features': {
                'use_partial_charges': getattr(args, 'use_partial_charges', False),
                'use_stereochemistry': getattr(args, 'use_stereochemistry', False),
                'calculate_sae': getattr(args, 'calculate_sae', False),
                'sae_subtasks': getattr(args, 'sae_subtasks', None),
            },
            'system': {
                'num_workers': getattr(args, 'num_workers', None),
                'num_gpu_devices': getattr(args, 'num_gpu_devices', None),
                'mixed_precision': getattr(args, 'mixed_precision', False),
            }
        }

        return metadata

    except Exception as e:
        raise ExperimentError(f"Failed to create experiment metadata: {e}") from e


def save_experiment_results(results: dict[str, Any], filepath: str) -> None:
    """
    Save experiment results with metadata.

    Args:
        results: Dictionary of results to save
        filepath: Path to save results

    Raises:
        ExperimentError: If saving fails
    """
    try:
        # Ensure parent directory exists
        parent_dir = Path(filepath).parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Add metadata to results
        results_with_metadata = {
            'results': results,
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'version': '1.0',
                'type': 'experiment_results'
            }
        }

        # Determine format based on file extension
        if filepath.endswith('.json'):
            try:
                with open(filepath, 'w') as f:
                    json.dump(results_with_metadata, f, indent=2, default=str)
            except (IOError, TypeError) as e:
                raise ExperimentError(f"Failed to save results as JSON: {e}") from e
        else:
            # Default to YAML
            try:
                with open(filepath, 'w') as f:
                    yaml.dump(results_with_metadata, f, default_flow_style=False)
            except (IOError, yaml.YAMLError) as e:
                raise ExperimentError(f"Failed to save results as YAML: {e}") from e

        logger.info(f"Experiment results saved to {filepath}")

    except ExperimentError:
        raise
    except Exception as e:
        raise ExperimentError(f"Unexpected error saving experiment results: {e}") from e


def load_experiment_results(filepath: str) -> dict[str, Any]:
    """
    Load experiment results from file.

    Args:
        filepath: Path to results file

    Returns:
        Dictionary of loaded results

    Raises:
        ExperimentError: If loading fails
    """
    try:
        if not os.path.exists(filepath):
            raise ExperimentError(f"Results file not found: {filepath}")

        # Load based on file extension
        if filepath.endswith('.json'):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                raise ExperimentError(f"Failed to load JSON results: {e}") from e
        else:
            # Default to YAML
            try:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
            except (IOError, yaml.YAMLError) as e:
                raise ExperimentError(f"Failed to load YAML results: {e}") from e

        # Extract results, handling both old and new formats
        if isinstance(data, dict) and 'results' in data:
            return data['results']
        else:
            return data

    except ExperimentError:
        raise
    except Exception as e:
        raise ExperimentError(f"Unexpected error loading experiment results: {e}") from e


def compare_experiment_configs(config1_path: str, config2_path: str) -> dict[str, Any]:
    """
    Compare two experiment configurations and return differences.

    Args:
        config1_path: Path to first configuration
        config2_path: Path to second configuration

    Returns:
        Dictionary describing differences

    Raises:
        ExperimentError: If comparison fails
    """
    try:
        config1 = load_experiment_config(config1_path)
        config2 = load_experiment_config(config2_path)

        dict1 = vars(config1)
        dict2 = vars(config2)

        differences = {
            'only_in_config1': {},
            'only_in_config2': {},
            'different_values': {},
            'same_values': {}
        }

        all_keys = set(dict1.keys()) | set(dict2.keys())

        for key in all_keys:
            if key not in dict1:
                differences['only_in_config2'][key] = dict2[key]
            elif key not in dict2:
                differences['only_in_config1'][key] = dict1[key]
            elif dict1[key] != dict2[key]:
                differences['different_values'][key] = {
                    'config1': dict1[key],
                    'config2': dict2[key]
                }
            else:
                differences['same_values'][key] = dict1[key]

        return differences

    except Exception as e:
        raise ExperimentError(f"Failed to compare experiment configs: {e}") from e


def _serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert argparse Namespace to serializable dictionary."""
    config = {}

    for key, value in vars(args).items():
        # Handle special types that YAML can't serialize
        if value is None:
            config[key] = None
        elif isinstance(value, (str, int, float, bool, list)):
            config[key] = value
        else:
            # Convert other types to string representation
            config[key] = str(value)

    return config


def _deserialize_args(config: dict[str, Any]) -> argparse.Namespace:
    """Convert dictionary back to argparse Namespace with type restoration."""
    # Known boolean arguments
    bool_args = {
        'early_stopping', 'calculate_sae', 'use_partial_charges',
        'use_stereochemistry', 'mixed_precision', 'iterable_dataset',
        'freeze_pretrained', 'layer_wise_lr_decay', 'save_embeddings',
        'include_atom_embeddings', 'enable_wandb'
    }

    # Known integer arguments
    int_args = {
        'epochs', 'batch_size', 'hidden_dim', 'num_shells', 'embedding_dim',
        'num_message_passing_layers', 'ffn_num_layers', 'patience',
        'num_workers', 'num_gpu_devices', 'attention_num_heads',
        'shell_conv_num_mlp_layers', 'num_trials', 'mc_samples'
    }

    # Known float arguments
    float_args = {
        'learning_rate', 'train_split', 'val_split', 'test_split',
        'ffn_dropout', 'shell_conv_dropout', 'attention_temperature',
        'lr_reduce_factor', 'lr_decay_factor'
    }

    restored_config = {}

    for key, value in config.items():
        if value is None:
            restored_config[key] = None
        elif key in bool_args:
            if isinstance(value, str):
                restored_config[key] = value.lower() in ('true', '1', 'yes', 'on')
            else:
                restored_config[key] = bool(value)
        elif key in int_args:
            try:
                restored_config[key] = int(value)
            except (ValueError, TypeError):
                restored_config[key] = value
        elif key in float_args:
            try:
                restored_config[key] = float(value)
            except (ValueError, TypeError):
                restored_config[key] = value
        else:
            restored_config[key] = value

    return argparse.Namespace(**restored_config)
