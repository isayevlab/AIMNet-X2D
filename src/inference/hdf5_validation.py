"""
HDF5 validation functions for model compatibility checks.

This module provides shared validation functions used by both model_loader.py
and pipeline.py to verify HDF5 file compatibility with loaded models.
"""


def _check_hdf5_max_hops_compatibility(
    hdf5_max_hops: int,
    model_num_shells: int
) -> str | None:
    """Check max_hops compatibility. Returns error message or None if compatible."""
    if hdf5_max_hops != model_num_shells:
        return (
            f"CRITICAL: Max hops mismatch!\n"
            f"   HDF5 file: {hdf5_max_hops} hops\n"
            f"   Model expects: {model_num_shells} hops\n"
            f"   \n"
            f"   This is a FATAL incompatibility - the molecular features\n"
            f"   in the HDF5 file do not match what the model was trained on.\n"
            f"   \n"
            f"   The HDF5 file contains BFS features computed to depth {hdf5_max_hops},\n"
            f"   but the model expects features computed to depth {model_num_shells}.\n"
            f"   These are fundamentally different graph representations.\n"
            f"   \n"
            f"   You MUST recreate the HDF5 file with --num_shells={model_num_shells}"
        )
    return None


def _check_hdf5_preprocessing_compatibility(
    hdf5_has_preprocessing: bool,
    model_has_preprocessing: bool
) -> str | None:
    """Check preprocessing compatibility. Returns error message or None if compatible."""
    if hdf5_has_preprocessing and not model_has_preprocessing:
        return (
            "HDF5 file has preprocessing applied but model has no preprocessing pipeline. "
            "This will produce incorrect results. Regenerate HDF5 without preprocessing."
        )
    return None


def _check_hdf5_task_type_compatibility(
    hdf5_task_type: str,
    model_task_type: str
) -> str | None:
    """Check task type compatibility. Returns error message or None if compatible."""
    if hdf5_task_type != model_task_type:
        return (
            f"Task type mismatch:\n"
            f"   HDF5: {hdf5_task_type}\n"
            f"   Model: {model_task_type}"
        )
    return None


def _check_hdf5_inference_data_compatibility(
    preprocessing_applied: bool
) -> str | None:
    """Check if HDF5 contains raw data suitable for inference. Returns error message or None if compatible."""
    if preprocessing_applied:
        return (
            "HDF5 contains PREPROCESSED data\n"
            "   Inference requires RAW data\n"
            "   Use create_inference_hdf5.py to create proper HDF5"
        )
    return None


def _format_compatibility_error(
    errors: list[str],
    model_path: str,
    input_path: str
) -> str:
    """Format compatibility errors into a detailed error message."""
    error_msg = "\n" + "=" * 60 + "\n"
    error_msg += "HDF5 FILE IS INCOMPATIBLE WITH MODEL\n"
    error_msg += "=" * 60 + "\n\n"
    for i, e in enumerate(errors, 1):
        error_msg += f"{i}. {e}\n\n"
    error_msg += "=" * 60 + "\n"
    error_msg += "SOLUTION:\n"
    error_msg += "=" * 60 + "\n"
    error_msg += f"Recreate the HDF5 file with matching parameters:\n\n"
    error_msg += f"  python create_inference_hdf5.py \\\n"
    error_msg += f"    --model_path {model_path} \\\n"
    error_msg += f"    --input_csv YOUR_DATA.csv \\\n"
    error_msg += f"    --output_hdf5 {input_path} \\\n"
    error_msg += f"    --smiles_column smiles\n"
    error_msg += "=" * 60 + "\n"
    return error_msg
