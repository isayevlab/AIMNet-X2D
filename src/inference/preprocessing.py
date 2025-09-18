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
        
        This fixes the critical bug where hyperparameters were not loaded during inference.
        """
        if "hyperparams" not in model_artifact:
            raise ValueError("Model artifact missing hyperparams - cannot perform inference")
        
        hyperparams = model_artifact["hyperparams"]
        
        # CRITICAL FIX: Load preprocessing config from model
        preprocessing_info = hyperparams.get("preprocessing_config")
        if not preprocessing_info:
            # Try legacy format
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
        
        # CRITICAL FIX: Restore SAE normalizer if it was used
        if config.apply_sae and "sae_statistics" in hyperparams and hyperparams["sae_statistics"]:
            pipeline.sae_normalizer = SAENormalizer(
                task_type=config.task_type,
                percentile_cutoff=config.sae_percentile_cutoff
            )
            pipeline.sae_normalizer.sae_statistics = hyperparams["sae_statistics"]
            pipeline.sae_normalizer.is_fitted = True
            print(f"[Preprocessing] Restored SAE normalizer with {len(hyperparams['sae_statistics'])} task(s)")
        
        # CRITICAL FIX: Restore standard scaler if it was used
        if config.apply_standard_scaling and "scaler_means" in hyperparams and hyperparams["scaler_means"]:
            pipeline.standard_scaler = StandardScaler()
            pipeline.standard_scaler.means = np.array(hyperparams["scaler_means"])
            pipeline.standard_scaler.stds = np.array(hyperparams["scaler_stds"])
            pipeline.standard_scaler.is_fitted = True
            print(f"[Preprocessing] Restored standard scaler: means={pipeline.standard_scaler.means}, stds={pipeline.standard_scaler.stds}")
        elif config.apply_standard_scaling:
            # CRITICAL FIX: Raise error instead of using dummy values
            raise ValueError(
                "Model expects standard scaling but no scaler statistics found in model file. "
                "Cannot perform inference with proper scaling. "
                "This model may have been saved with an older version of the code."
            )
        
        pipeline.is_fitted = True
        return pipeline



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