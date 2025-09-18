"""
Preprocessing pipeline reconstruction for inference.
"""

import numpy as np
from typing import Optional, Dict, Any

from data.preprocessing import PreprocessingPipeline, PreprocessingConfig, SAENormalizer, StandardScaler


class PreprocessingReconstructor:

    @staticmethod
    def load_preprocessing_pipeline(model_artifact: Dict[str, Any]) -> Optional[PreprocessingPipeline]:
        """
        CRITICAL FIX: Properly reconstruct ALL preprocessing pipeline parameters from saved model.
        """
        if "hyperparams" not in model_artifact:
            raise ValueError("Model artifact missing hyperparams - cannot perform inference")
        
        hyperparams = model_artifact["hyperparams"]
        
        # CRITICAL FIX: Comprehensive preprocessing reconstruction
        preprocessing_info = hyperparams.get("preprocessing_config")
        if not preprocessing_info:
            # Try legacy format with explicit error for missing info
            return PreprocessingReconstructor._load_legacy_format(hyperparams)
        
        # Reconstruct full preprocessing config with ALL parameters
        config = PreprocessingConfig(
            apply_sae=preprocessing_info.get("apply_sae", False),
            sae_subtasks=preprocessing_info.get("sae_subtasks"),
            apply_standard_scaling=preprocessing_info.get("apply_standard_scaling", True),
            task_type=preprocessing_info.get("task_type", "regression"),
            sae_percentile_cutoff=preprocessing_info.get("sae_percentile_cutoff", 2.0)
        )
        
        print(f"[Preprocessing] Loaded config: SAE={config.apply_sae}, Scaling={config.apply_standard_scaling}, Task={config.task_type}")
        
        # Create pipeline
        pipeline = PreprocessingPipeline(config)
        
        # CRITICAL FIX: Restore SAE normalizer with proper error handling
        if config.apply_sae:
            if "sae_statistics" not in hyperparams or not hyperparams["sae_statistics"]:
                raise ValueError(
                    "Model was trained with SAE normalization but no SAE statistics found in model file. "
                    "Cannot perform inference without proper SAE statistics. "
                    "Please retrain the model or check if the model file is corrupted."
                )
            
            pipeline.sae_normalizer = SAENormalizer(
                task_type=config.task_type,
                percentile_cutoff=config.sae_percentile_cutoff
            )
            pipeline.sae_normalizer.sae_statistics = hyperparams["sae_statistics"]
            pipeline.sae_normalizer.is_fitted = True
            print(f"[Preprocessing] ✅ Restored SAE normalizer with {len(hyperparams['sae_statistics'])} task(s)")
        
        # CRITICAL FIX: Restore standard scaler with proper error handling
        if config.apply_standard_scaling:
            if "scaler_means" not in hyperparams or hyperparams["scaler_means"] is None:
                raise ValueError(
                    "Model was trained with standard scaling but no scaler statistics found in model file. "
                    "Cannot perform inference without proper scaling statistics. "
                    "Please retrain the model or check if the model file is corrupted."
                )
            
            pipeline.standard_scaler = StandardScaler()
            pipeline.standard_scaler.means = np.array(hyperparams["scaler_means"])
            pipeline.standard_scaler.stds = np.array(hyperparams["scaler_stds"])
            pipeline.standard_scaler.is_fitted = True
            print(f"[Preprocessing] ✅ Restored standard scaler: means={pipeline.standard_scaler.means}, stds={pipeline.standard_scaler.stds}")
        
        pipeline.is_fitted = True
        
        # CRITICAL FIX: Validate preprocessing pipeline
        PreprocessingReconstructor._validate_preprocessing_pipeline(pipeline, hyperparams)
        
        return pipeline

    @staticmethod
    def _validate_preprocessing_pipeline(pipeline: PreprocessingPipeline, hyperparams: Dict[str, Any]) -> None:
        """Validate that preprocessing pipeline is correctly reconstructed."""
        config = pipeline.config
        
        # Validate SAE configuration
        if config.apply_sae:
            if not pipeline.sae_normalizer or not pipeline.sae_normalizer.is_fitted:
                raise ValueError("SAE normalization enabled but SAE normalizer not properly restored")
            
            # Check task consistency
            if config.task_type != pipeline.sae_normalizer.task_type:
                raise ValueError(f"Task type mismatch in SAE: {config.task_type} != {pipeline.sae_normalizer.task_type}")
        
        # Validate scaling configuration
        if config.apply_standard_scaling:
            if not pipeline.standard_scaler or not pipeline.standard_scaler.is_fitted:
                raise ValueError("Standard scaling enabled but scaler not properly restored")
        
        print(f"[Preprocessing] ✅ Preprocessing pipeline validation passed")

    @staticmethod
    def _load_legacy_format(hyperparams: Dict[str, Any]) -> Optional[PreprocessingPipeline]:
        """Load legacy preprocessing format with error handling."""
        if "scaler_means" in hyperparams and hyperparams["scaler_means"] is not None:
            print("[Preprocessing] Detected legacy model format, reconstructing standard scaler only")
            
            config = PreprocessingConfig(
                apply_sae=False,
                sae_subtasks=None,
                apply_standard_scaling=True,
                task_type=hyperparams.get("task_type", "regression")
            )
            
            pipeline = PreprocessingPipeline(config)
            
            # Restore standard scaler
            pipeline.standard_scaler = StandardScaler()
            pipeline.standard_scaler.means = np.array(hyperparams["scaler_means"])
            pipeline.standard_scaler.stds = np.array(hyperparams["scaler_stds"])
            pipeline.standard_scaler.is_fitted = True
            pipeline.is_fitted = True
            
            print("[Preprocessing] Legacy preprocessing pipeline restored")
            return pipeline
        
        # CRITICAL FIX: If no preprocessing information found, raise clear error
        raise ValueError(
            "No preprocessing information found in model file. "
            "Cannot perform inference without knowing how the training data was preprocessed. "
            "Please retrain the model with the current version of the code."
        )