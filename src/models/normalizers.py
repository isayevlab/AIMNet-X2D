"""
Normalization for molecular property prediction.

This module simply re-exports the preprocessing system from data.preprocessing
and provides convenient factory functions.
"""


# Import the preprocessing system from data.preprocessing
from data.preprocessing import (
    PreprocessingPipeline,
    PreprocessingConfig, 
    SAENormalizer,
    StandardScaler
)

# Export everything from the preprocessing system
__all__ = [
    'PreprocessingPipeline',
    'PreprocessingConfig', 
    'SAENormalizer',
    'StandardScaler',
    'create_preprocessing_pipeline',
    'create_sae_pipeline',
    'create_standard_pipeline',
]


def create_preprocessing_pipeline(apply_sae: bool = False, 
                                sae_subtasks: list[int] | None = None,
                                apply_standard_scaling: bool = True,
                                task_type: str = "regression",
                                sae_percentile_cutoff: float = 2.0) -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline with common settings.
    
    Args:
        apply_sae: Whether to apply SAE normalization
        sae_subtasks: Subtasks for SAE (for multitask only)
        apply_standard_scaling: Whether to apply standard scaling
        task_type: Task type ('regression' or 'multitask')
        sae_percentile_cutoff: Percentile cutoff for SAE outlier filtering
        
    Returns:
        Configured preprocessing pipeline
        
    Examples:
        # Single-task with SAE + scaling
        pipeline = create_preprocessing_pipeline(
            apply_sae=True,
            task_type="regression"
        )
        
        # Multi-task with SAE on specific subtasks
        pipeline = create_preprocessing_pipeline(
            apply_sae=True,
            sae_subtasks=[0, 2],
            task_type="multitask"
        )
        
        # Just standard scaling (no SAE)
        pipeline = create_preprocessing_pipeline(
            apply_sae=False,
            task_type="regression"
        )
    """
    config = PreprocessingConfig(
        apply_sae=apply_sae,
        sae_subtasks=sae_subtasks,
        apply_standard_scaling=apply_standard_scaling,
        task_type=task_type,
        sae_percentile_cutoff=sae_percentile_cutoff
    )
    return PreprocessingPipeline(config)


def create_sae_pipeline(task_type: str = "regression", 
                       subtasks: list[int] | None = None) -> PreprocessingPipeline:
    """
    Create a pipeline with SAE normalization enabled.
    
    Args:
        task_type: 'regression' or 'multitask'
        subtasks: Subtasks for multitask SAE
        
    Returns:
        Pipeline with SAE + standard scaling
    """
    return create_preprocessing_pipeline(
        apply_sae=True,
        sae_subtasks=subtasks,
        task_type=task_type
    )


def create_standard_pipeline(task_type: str = "regression") -> PreprocessingPipeline:
    """
    Create a pipeline with only standard scaling (no SAE).
    
    Args:
        task_type: 'regression' or 'multitask'
        
    Returns:
        Pipeline with only standard scaling
    """
    return create_preprocessing_pipeline(
        apply_sae=False,
        task_type=task_type
    )