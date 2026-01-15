"""
Model loading and validation for inference pipeline.
"""

import os
from typing import Any

import torch
import h5py

from .config import InferenceConfig
from .preprocessing import PreprocessingReconstructor
from .hdf5_validation import (
    _check_hdf5_max_hops_compatibility,
    _check_hdf5_task_type_compatibility,
    _check_hdf5_inference_data_compatibility,
    _format_compatibility_error,
)
from models import GNN
from data.preprocessing import PreprocessingPipeline
from utils.distributed import is_main_process
from utils.logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Handles model loading and validation for inference."""

    def __init__(self, config: InferenceConfig) -> None:
        """
        Initialize ModelLoader with inference configuration.

        Args:
            config: InferenceConfig containing model path and settings.
        """
        self.config = config
        self.device: torch.device | None = None

    def load(self, device: torch.device) -> tuple[GNN, PreprocessingPipeline | None, dict[str, Any]]:
        """
        Load model, preprocessing pipeline, and hyperparameters.

        Args:
            device: Device to load the model onto.

        Returns:
            Tuple of (model, preprocessing_pipeline, hyperparams).
        """
        self.device = device

        # Load model and preprocessing
        model, preprocessing_pipeline, hyperparams = self._load_model_and_preprocessing()

        return model, preprocessing_pipeline, hyperparams

    def _load_model_and_preprocessing(self) -> tuple[GNN, PreprocessingPipeline | None, dict[str, Any]]:
        """Load model and reconstruct preprocessing pipeline."""
        if not self.config.model_path:
            raise ValueError("Model path not specified in configuration")
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")

        # Load model artifact
        model_artifact = torch.load(self.config.model_path, map_location=self.device)

        if "hyperparams" not in model_artifact:
            raise ValueError("Model file missing hyperparams - incompatible model format")

        hyperparams = model_artifact["hyperparams"]
        state_dict = model_artifact["state_dict"]

        # CRITICAL FIX: Verify critical hyperparameters match expectations
        self._verify_model_compatibility(hyperparams)

        # Reconstruct preprocessing pipeline
        preprocessing_pipeline = PreprocessingReconstructor.load_preprocessing_pipeline(model_artifact)

        # Build model with EXACT hyperparameters from saved model
        model = self._build_model_from_hyperparams(hyperparams, state_dict)

        if is_main_process():
            logger.info(f"Model loaded from {self.config.model_path}")
            logger.info("Model hyperparameters verified and restored")
            loss_function = hyperparams.get('loss_function', 'l1')
            logger.info(f"Loss function: {loss_function}")
            logger.info(f"Hidden dim: {hyperparams.get('hidden_dim')}")
            logger.info(f"Num shells: {hyperparams.get('num_shells')}")
            logger.info(f"Task type: {hyperparams.get('task_type')}")
            logger.info(f"FFN dropout: {hyperparams.get('ffn_dropout')}")
            logger.info(f"Shell conv dropout: {hyperparams.get('shell_conv_dropout')}")
            logger.info(f"Attention heads: {hyperparams.get('attention_num_heads')}")
            if preprocessing_pipeline:
                logger.info("Preprocessing pipeline loaded successfully")

        return model, preprocessing_pipeline, hyperparams

    def _verify_model_compatibility(self, hyperparams: dict[str, Any]) -> None:
        """Verify that loaded model parameters are compatible with inference requirements."""
        required_params = [
            'hidden_dim', 'num_shells', 'num_message_passing_layers',
            'task_type', 'loss_function'
        ]

        missing_params = []
        for param in required_params:
            if param not in hyperparams:
                missing_params.append(param)

        if missing_params:
            raise ValueError(f"Model missing critical hyperparameters: {missing_params}")

        # Verify inference configuration compatibility
        if hasattr(self.config, 'max_hops') and self.config.max_hops and self.config.max_hops != hyperparams.get('num_shells'):
            logger.warning(f"Config max_hops ({self.config.max_hops}) != model num_shells ({hyperparams.get('num_shells')})")
            logger.warning(f"Using model's num_shells value: {hyperparams.get('num_shells')}")
            self.config.max_hops = hyperparams.get('num_shells')

    def _build_model_from_hyperparams(self, hyperparams: dict[str, Any], state_dict: dict[str, Any]) -> GNN:
        """Build model using EXACT hyperparameters from saved model."""

        # CRITICAL FIX: Use feature sizes from saved model, not hardcoded values
        if "feature_sizes" in hyperparams:
            feature_sizes = hyperparams["feature_sizes"]
            logger.info(f"Using saved feature sizes: {feature_sizes}")
        else:
            # Fallback with warning
            logger.warning("No feature_sizes in saved model, using defaults")
            feature_sizes = {
                'atom_type': 119,
                'hydrogen_count': 9,
                'degree': 7,
                'hybridization': 7,
            }

        # Get exact output dimension from state dict
        output_dim = self._get_output_dim_from_state_dict(state_dict, hyperparams)
        logger.info(f"Building model with output_dim={output_dim}")

        # CRITICAL FIX: Use ALL hyperparameters from saved model with proper defaults
        model = GNN(
            feature_sizes=feature_sizes,
            hidden_dim=hyperparams["hidden_dim"],
            output_dim=output_dim,
            num_shells=hyperparams["num_shells"],
            num_message_passing_layers=hyperparams["num_message_passing_layers"],
            ffn_hidden_dim=hyperparams.get("ffn_hidden_dim", hyperparams["hidden_dim"]),
            ffn_num_layers=hyperparams.get("ffn_num_layers", 3),
            pooling_type=hyperparams.get("pooling_type", "attention"),
            task_type=hyperparams["task_type"],
            embedding_dim=hyperparams.get("embedding_dim", 64),
            use_partial_charges=hyperparams.get("use_partial_charges", False),
            use_stereochemistry=hyperparams.get("use_stereochemistry", False),
            ffn_dropout=hyperparams.get("ffn_dropout", 0.05),
            activation_type=hyperparams.get("activation_type", "silu"),
            shell_conv_num_mlp_layers=hyperparams.get("shell_conv_num_mlp_layers", 2),
            shell_conv_dropout=hyperparams.get("shell_conv_dropout", 0.05),
            attention_num_heads=hyperparams.get("attention_num_heads", 4),
            attention_temperature=hyperparams.get("attention_temperature", 1.0),
            loss_function=hyperparams.get("loss_function", "l1")
        ).to(self.device)

        # CRITICAL FIX: Strict loading with proper error handling
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise ValueError(f"Failed to load model state dict: {e}") from e

        model.eval()

        logger.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")

        # CRITICAL FIX: Validate that loaded model matches expected configuration
        self._validate_loaded_model(model, hyperparams)

        return model

    def _validate_loaded_model(self, model: GNN, hyperparams: dict[str, Any]) -> None:
        """Validate that loaded model matches saved hyperparameters."""
        try:
            # Check critical architecture parameters
            assert model.hidden_dim == hyperparams["hidden_dim"], f"Hidden dim mismatch: {model.hidden_dim} != {hyperparams['hidden_dim']}"
            assert model.num_shells == hyperparams["num_shells"], f"Num shells mismatch: {model.num_shells} != {hyperparams['num_shells']}"
            assert model.task_type == hyperparams["task_type"], f"Task type mismatch: {model.task_type} != {hyperparams['task_type']}"
            assert model.loss_function == hyperparams.get("loss_function", "l1"), f"Loss function mismatch"

            logger.info("Model validation passed")

        except AssertionError as e:
            raise ValueError(f"Model validation failed: {e}") from e

    def _get_output_dim_from_state_dict(self, state_dict: dict[str, Any], hyperparams: dict[str, Any]) -> int:
        """Determine output dimension from state dict."""
        output_keys = [
            "output_layer.weight", "module.output_layer.weight",
            "classifier.weight", "module.classifier.weight"
        ]

        for key in output_keys:
            if key in state_dict:
                output_layer_size = state_dict[key].shape[0]

                # For evidential loss, the actual number of tasks is output_size / 4
                loss_function = hyperparams.get('loss_function', 'l1')
                if loss_function == 'evidential' and output_layer_size % 4 == 0:
                    return output_layer_size // 4
                else:
                    return output_layer_size

        # Fallback to hyperparams
        return hyperparams.get('output_dim', 1)

    def verify_hdf5_model_compatibility(self, model: GNN) -> None:
        """Verify HDF5 file is compatible with loaded model."""
        if not hasattr(self.config, 'input_path'):
            return

        if not self.config.input_path.endswith(('.h5', '.hdf5')):
            return

        logger.info("")
        logger.info("=" * 60)
        logger.info("VERIFYING HDF5 COMPATIBILITY WITH MODEL")
        logger.info("=" * 60)

        try:
            with h5py.File(self.config.input_path, 'r') as f:
                if 'metadata' not in f:
                    logger.warning("HDF5 file has no metadata")
                    logger.warning("Cannot verify compatibility - proceeding with caution")
                    return

                metadata = f['metadata']
                model_max_hops, model_task_type = self._get_model_parameters(model)

                if 'model_compatibility' in metadata:
                    compat = metadata['model_compatibility']
                    errors = self._check_compatibility_metadata(compat, model_max_hops, model_task_type)

                    if errors:
                        error_msg = _format_compatibility_error(
                            errors, self.config.model_path, self.config.input_path
                        )
                        raise ValueError(error_msg)

                    self._log_compatibility_success(compat)
                else:
                    # Old format - basic checks only
                    logger.warning("HDF5 missing model_compatibility metadata")
                    logger.warning("Performing basic validation only...")

                    preprocessing_applied = metadata.attrs.get('preprocessing_applied', False)
                    if preprocessing_applied:
                        raise ValueError(
                            "HDF5 contains preprocessed data but inference requires raw data.\n"
                            "Please recreate using create_inference_hdf5.py"
                        )
                    logger.info("Basic validation passed")

        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Could not verify HDF5 compatibility: {e}")
            logger.warning("Proceeding with caution...")

        logger.info("=" * 60)
        logger.info("")

    def _get_model_parameters(self, model: GNN) -> tuple[int | None, str | None]:
        """Extract model parameters handling DDP wrapper."""
        model_max_hops = model.num_shells if hasattr(model, 'num_shells') else None
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_max_hops = model.module.num_shells if hasattr(model.module, 'num_shells') else None

        model_task_type = model.task_type if hasattr(model, 'task_type') else None
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_task_type = model.module.task_type if hasattr(model.module, 'task_type') else None

        return model_max_hops, model_task_type

    def _check_compatibility_metadata(
        self,
        compat: Any,
        model_max_hops: int | None,
        model_task_type: str | None
    ) -> list[str]:
        """Check compatibility metadata and return list of errors."""
        errors = []

        # Check max_hops compatibility
        if 'max_hops' in compat.attrs and model_max_hops is not None:
            hdf5_max_hops = int(compat.attrs['max_hops'])
            error = _check_hdf5_max_hops_compatibility(hdf5_max_hops, model_max_hops)
            if error:
                errors.append(error)

        # Check task type compatibility
        if 'task_type' in compat.attrs and model_task_type is not None:
            hdf5_task_type = str(compat.attrs['task_type'])
            error = _check_hdf5_task_type_compatibility(hdf5_task_type, model_task_type)
            if error:
                errors.append(error)

        # Check preprocessing status
        if 'preprocessing_applied' in compat.attrs:
            error = _check_hdf5_inference_data_compatibility(compat.attrs['preprocessing_applied'])
            if error:
                errors.append(error)

        # Check if marked for inference
        if 'for_inference' in compat.attrs:
            if not compat.attrs['for_inference']:
                logger.warning("HDF5 not marked for inference")
                logger.warning("This file may have been created for training")

        return errors

    def _log_compatibility_success(self, compat: Any) -> None:
        """Log successful compatibility check."""
        logger.info("HDF5 file is COMPATIBLE with model")
        logger.info(f"   Max hops: {compat.attrs.get('max_hops', 'N/A')}")
        logger.info(f"   Task type: {compat.attrs.get('task_type', 'N/A')}")
        preprocessing_status = 'Applied' if compat.attrs.get('preprocessing_applied', False) else 'RAW (will apply during inference)'
        logger.info(f"   Preprocessing: {preprocessing_status}")
        logger.info(f"   For inference: {compat.attrs.get('for_inference', 'N/A')}")
