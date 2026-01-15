"""
Legacy hyperparameter optimization for AIMNet-X2D.

This module provides hyperparameter optimization using YAML configuration files
and simple random/grid search without external dependencies.
"""

import os
import json
import yaml
import random
import copy
import math
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class HyperparameterError(Exception):
    """Raised when hyperparameter operations fail."""
    pass


def run_hyperparameter_optimization(args) -> dict[str, Any]:
    """
    Run hyperparameter optimization using the legacy YAML-based system.

    Args:
        args: Command line arguments

    Returns:
        Dictionary containing optimization results

    Raises:
        HyperparameterError: If hyperparameter optimization fails
    """
    logger.info("=" * 80)
    logger.info("STARTING HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)

    # Import here to avoid circular imports
    from main.runner import run_single_trial

    # Load hyperparameter configuration
    if not args.hyperparameter_file or not os.path.exists(args.hyperparameter_file):
        raise HyperparameterError(f"Hyperparameter file not found: {args.hyperparameter_file}")

    with open(args.hyperparameter_file, 'r') as f:
        hparam_config = yaml.safe_load(f)

    results = {
        "trials": [],
        "best_trial": None,
        "best_metrics": None
    }

    best_loss = float('inf')
    best_trial_data = None

    for trial_idx in range(args.num_trials):
        logger.info(f"\n=== Trial {trial_idx + 1}/{args.num_trials} ===")

        # Sample hyperparameters
        trial_config = {}
        for param_name, param_config in hparam_config.items():
            trial_config[param_name] = _sample_hparam_value(param_config)

        logger.info(f"Hyperparameters: {trial_config}")

        # Create trial arguments
        trial_args = copy.deepcopy(args)
        for param_name, param_value in trial_config.items():
            setattr(trial_args, param_name, param_value)

        # Mark this as a hyperopt trial to prevent individual model saving
        trial_args._is_hyperopt_trial = True

        # Run trial
        try:
            trial_results = run_single_trial(trial_args)

            trial_info = {
                "trial_id": trial_idx,
                "config": trial_config,
                "metrics": {k: v for k, v in trial_results.items() if not k.startswith('_')},
                "status": "completed"
            }

            # Check if this is the best trial
            val_loss = trial_results.get("val_loss", float('inf'))
            if val_loss < best_loss:
                best_loss = val_loss
                results["best_trial"] = trial_info
                results["best_metrics"] = trial_info["metrics"]
                # Store the full trial data including model state
                best_trial_data = {
                    "args": trial_args,
                    "results": trial_results,
                    "trial_id": trial_idx
                }
                logger.info(f"New best trial! Loss: {val_loss:.6f}")

        except Exception as e:
            logger.error(f"Trial {trial_idx + 1} failed: {e}")
            trial_info = {
                "trial_id": trial_idx,
                "config": trial_config,
                "metrics": {"error": str(e)},
                "status": "failed"
            }

        results["trials"].append(trial_info)

    # Save the best model ONCE at the end
    if best_trial_data is not None:
        logger.info(f"\nSaving best model from trial {results['best_trial']['trial_id'] + 1}...")
        _save_best_hyperopt_model(best_trial_data, args)

        # Verify the saved model
        _verify_saved_model(args, results["best_trial"])

    # Print summary
    _print_hyperopt_summary(results, args)

    # Save results
    _save_optimization_results(results, args)

    # Log to wandb if enabled
    if args.enable_wandb and WANDB_AVAILABLE:
        _log_optimization_to_wandb(results, args)

    return results


def _sample_hparam_value(param_config) -> Any:
    """Sample a hyperparameter value from configuration."""
    if isinstance(param_config, list):
        # Grid search: return a random element from the list
        return random.choice(param_config)
    elif isinstance(param_config, dict):
        # Random search: sample based on type and range
        if param_config.get("type") == "int":
            return random.randint(param_config["min"], param_config["max"])
        elif param_config.get("type") == "float":
            if param_config.get("log", False):
                # Sample on a logarithmic scale
                log_min = math.log(param_config["min"])
                log_max = math.log(param_config["max"])
                log_value = random.uniform(log_min, log_max)
                return math.exp(log_value)
            else:
                return random.uniform(param_config["min"], param_config["max"])
        elif param_config.get("type") == "choice":
            return random.choice(param_config["values"])
        else:
            # If it's a dict without type field, just return it as is
            return param_config
    else:
        # Fixed value (not a list or dict)
        return param_config


def _save_best_hyperopt_model(best_trial_data: dict[str, Any], args) -> None:
    """Save the best model from hyperparameter optimization."""
    trial_results = best_trial_data["results"]
    trial_args = best_trial_data["args"]

    # Extract model components
    model = trial_results["_model_state"]
    preprocessing_pipeline = trial_results["_preprocessing_pipeline"]
    test_metrics = trial_results["_test_metrics"]

    # Get model state dict
    if hasattr(model, 'module'):  # DDP model
        model_state_dict = {k: v.cpu() for k, v in model.module.state_dict().items()}
        model_for_config = model.module
    else:
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        model_for_config = model

    # Create model artifact with hyperopt config
    model_artifact = {
        "hyperparams": {
            "task_type": trial_args.task_type,
            "num_shells": trial_args.num_shells,
            "hidden_dim": trial_args.hidden_dim,
            "num_message_passing_layers": trial_args.num_message_passing_layers,
            "ffn_hidden_dim": trial_args.ffn_hidden_dim,
            "ffn_num_layers": trial_args.ffn_num_layers,
            "pooling_type": trial_args.pooling_type,
            "embedding_dim": trial_args.embedding_dim,
            "use_partial_charges": trial_args.use_partial_charges,
            "use_stereochemistry": trial_args.use_stereochemistry,
            "ffn_dropout": trial_args.ffn_dropout,
            "learning_rate": trial_args.learning_rate,
            "activation_type": trial_args.activation_type,
            "shell_conv_num_mlp_layers": trial_args.shell_conv_num_mlp_layers,
            "shell_conv_dropout": trial_args.shell_conv_dropout,
            "attention_num_heads": trial_args.attention_num_heads,
            "attention_temperature": trial_args.attention_temperature,
            "loss_function": trial_args.loss_function,
            "evidential_lambda": getattr(trial_args, 'evidential_lambda', 1.0),
            "best_val_loss": test_metrics.get("loss", float('inf')),

            # Hyperopt metadata
            "hyperopt_best_trial": True,
            "hyperopt_trial_id": best_trial_data.get("trial_id", -1),

            # Preprocessing pipeline information
            "preprocessing_config": {
                "apply_sae": preprocessing_pipeline.config.apply_sae,
                "sae_subtasks": preprocessing_pipeline.config.sae_subtasks,
                "apply_standard_scaling": preprocessing_pipeline.config.apply_standard_scaling,
                "task_type": preprocessing_pipeline.config.task_type,
                "sae_percentile_cutoff": preprocessing_pipeline.config.sae_percentile_cutoff
            } if preprocessing_pipeline else None,

            # Scaler parameters
            "scaler_means": preprocessing_pipeline.standard_scaler.means.tolist() if (
                preprocessing_pipeline and preprocessing_pipeline.standard_scaler
            ) else None,
            "scaler_stds": preprocessing_pipeline.standard_scaler.stds.tolist() if (
                preprocessing_pipeline and preprocessing_pipeline.standard_scaler
            ) else None,

            # SAE statistics
            "sae_statistics": preprocessing_pipeline.sae_normalizer.sae_statistics if (
                preprocessing_pipeline and preprocessing_pipeline.sae_normalizer
            ) else None,
        },
        "state_dict": model_state_dict
    }

    # Ensure output directory exists
    model_dir = os.path.dirname(os.path.abspath(args.model_save_path))
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    # Save model
    import torch
    torch.save(model_artifact, args.model_save_path)
    logger.info(f"Best model saved to: {args.model_save_path}")


def _verify_saved_model(args, best_trial_info: dict[str, Any]) -> bool:
    """Verify that the saved model contains the expected configuration."""
    try:
        import torch

        # Load the saved model
        model_artifact = torch.load(args.model_save_path, map_location='cpu')

        # Verify it contains expected fields
        assert "hyperparams" in model_artifact, "Model missing hyperparameters"
        assert "state_dict" in model_artifact, "Model missing state dict"

        # Verify best trial metadata
        hyperparams = model_artifact["hyperparams"]
        assert hyperparams.get("hyperopt_best_trial") == True, "Model not marked as best trial"

        # Verify key hyperparameters match
        saved_config = {k: v for k, v in hyperparams.items()
                       if k in best_trial_info["config"]}

        logger.info("Model verification passed")
        logger.info(f"   - Contains {len(model_artifact['state_dict'])} parameter tensors")
        logger.info(f"   - Best trial loss: {hyperparams.get('best_val_loss', 'N/A')}")
        logger.info(f"   - Key config: hidden_dim={hyperparams.get('hidden_dim')}, "
              f"lr={hyperparams.get('learning_rate'):.2e}")

        return True

    except Exception as e:
        logger.warning(f"Model verification failed: {e}")
        return False


def _print_hyperopt_summary(results: dict[str, Any], args) -> None:
    """Print hyperparameter optimization summary."""
    logger.info("\n" + "=" * 80)
    logger.info("HYPERPARAMETER OPTIMIZATION SUMMARY")
    logger.info("=" * 80)

    completed_trials = [t for t in results["trials"] if t["status"] == "completed"]
    failed_trials = [t for t in results["trials"] if t["status"] == "failed"]

    logger.info(f"Total trials: {len(results['trials'])}")
    logger.info(f"Completed trials: {len(completed_trials)}")
    logger.info(f"Failed trials: {len(failed_trials)}")

    if results["best_trial"]:
        best = results["best_trial"]
        logger.info(f"\nBest trial: {best['trial_id'] + 1}")
        logger.info(f"Best validation loss: {best['metrics'].get('val_loss', 'N/A'):.6f}")
        logger.info("Best metrics:")
        metrics = best['metrics']
        logger.info(f"   - MAE: {metrics.get('val_mae', 'N/A'):.6f}")
        logger.info(f"   - RMSE: {metrics.get('val_rmse', 'N/A'):.6f}")
        logger.info(f"   - R2: {metrics.get('val_r2', 'N/A'):.6f}")

        logger.info("\nBest hyperparameters:")
        for k, v in best["config"].items():
            if isinstance(v, float):
                logger.info(f"   - {k}: {v:.6f}")
            else:
                logger.info(f"   - {k}: {v}")

        logger.info(f"\nModel saved to: {args.model_save_path}")
    else:
        logger.warning("No successful trials found.")

    logger.info("=" * 80)


def _save_optimization_results(results: dict[str, Any], args) -> None:
    """Save optimization results to file."""
    # Create results directory if it doesn't exist
    results_dir = "hyperopt_results"
    os.makedirs(results_dir, exist_ok=True)

    # Generate filename
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"hyperopt_results_{timestamp}.json")

    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to: {results_file}")


def _log_optimization_to_wandb(results: dict[str, Any], args) -> None:
    """Log optimization results to Weights & Biases."""
    if not WANDB_AVAILABLE:
        logger.warning("Weights & Biases not available, skipping logging")
        return

    try:
        # Initialize a new run for the optimization summary
        wandb_config = {
            "optimization_type": "legacy_hyperopt",
            "num_trials": args.num_trials,
            "hyperparameter_file": args.hyperparameter_file,
        }

        # Add tags if specified
        tags = ["hyperopt", "legacy"]
        if args.wandb_tags_list:
            tags.extend(args.wandb_tags_list)

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"hyperopt_summary_{len(results['trials'])}_trials",
            config=wandb_config,
            tags=tags
        )

        # Log optimization statistics
        completed_trials = [t for t in results["trials"] if t["status"] == "completed"]
        failed_trials = [t for t in results["trials"] if t["status"] == "failed"]

        wandb.log({
            "optimization/num_trials": len(results["trials"]),
            "optimization/num_successful": len(completed_trials),
            "optimization/num_failed": len(failed_trials),
        })

        # Log best results if available
        if results["best_trial"]:
            best_metrics = results["best_trial"]["metrics"]
            wandb.log({
                "optimization/best_loss": best_metrics.get("val_loss", float('inf')),
                "optimization/best_mae": best_metrics.get("val_mae", float('inf')),
                "optimization/best_rmse": best_metrics.get("val_rmse", float('inf')),
                "optimization/best_r2": best_metrics.get("val_r2", -float('inf')),
            })

            # Log best hyperparameters
            best_config = results["best_trial"]["config"]
            wandb.log({f"best_config/{k}": v for k, v in best_config.items()})

        # Create summary table of all trials
        trial_data = []
        for trial in results["trials"]:
            if trial["status"] == "completed":
                row = {
                    "trial_id": trial["trial_id"],
                    "status": trial["status"],
                    **trial["metrics"],
                    **{f"config_{k}": v for k, v in trial["config"].items()}
                }
                trial_data.append(row)

        # Log trial table
        if trial_data:
            wandb.log({"optimization/trials_table": wandb.Table(data=trial_data)})

        wandb.finish()

    except Exception as e:
        logger.error(f"Failed to log to Weights & Biases: {e}")


def create_example_hyperparameter_config() -> dict[str, Any]:
    """
    Create an example hyperparameter configuration file.

    Returns:
        Dictionary with example hyperparameter configurations
    """
    return {
        # Model architecture parameters
        "hidden_dim": {
            "type": "choice",
            "values": [256, 512, 768, 1024]
        },
        "num_shells": {
            "type": "int",
            "min": 2,
            "max": 5
        },
        "num_message_passing_layers": {
            "type": "int",
            "min": 2,
            "max": 5
        },
        "embedding_dim": [32, 64, 128],  # Grid search
        "ffn_num_layers": [2, 3, 4],
        "attention_num_heads": [2, 4, 8],

        # Training parameters
        "learning_rate": {
            "type": "float",
            "min": 1e-5,
            "max": 1e-2,
            "log": True  # Sample on log scale
        },
        "batch_size": {
            "type": "choice",
            "values": [32, 64, 128, 256]
        },
        "ffn_dropout": {
            "type": "float",
            "min": 0.0,
            "max": 0.3
        },
        "shell_conv_dropout": {
            "type": "float",
            "min": 0.0,
            "max": 0.3
        },

        # Optimization parameters
        "lr_scheduler": ["ReduceLROnPlateau", "CosineAnnealingLR", "StepLR"],
        "lr_reduce_factor": {
            "type": "float",
            "min": 0.1,
            "max": 0.8
        },
        "lr_patience": {
            "type": "int",
            "min": 5,
            "max": 20
        },

        # Architecture choices
        "pooling_type": ["attention", "mean", "max", "sum"],
        "activation_type": ["relu", "leakyrelu", "elu", "silu", "gelu"],
    }


def save_example_config(output_path: str = "example_hyperparameters.yaml") -> None:
    """
    Save an example hyperparameter configuration to a YAML file.

    Args:
        output_path: Path to save the example configuration
    """
    config = create_example_hyperparameter_config()

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)

    logger.info(f"Example hyperparameter configuration saved to: {output_path}")
    logger.info("To use this configuration:")
    logger.info(f"  python main.py --hyperparameter_file {output_path} --num_trials 10")


# Legacy compatibility function
def run_legacy_hyperparameter_optimization(args) -> dict[str, Any]:
    """
    Legacy compatibility function that just calls the main optimization function.

    Args:
        args: Command line arguments

    Returns:
        Dictionary containing optimization results
    """
    return run_hyperparameter_optimization(args)
