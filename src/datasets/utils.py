# datasets/utils.py
"""
Utility functions for dataset management.
"""

import os
from .molecular import PyGSMILESDataset


def subset_in_memory_dataset(dataset: PyGSMILESDataset, indices: list[int]) -> PyGSMILESDataset:
    """
    Creates a new PyGSMILESDataset containing only the data objects at the specified indices.
    
    Args:
        dataset: Original dataset
        indices: Indices to include in subset
        
    Returns:
        New dataset containing only the specified indices
    """
    subset_smiles = [dataset.smiles_list[i] for i in indices]
    subset_targets = [dataset.targets[i] for i in indices]
    subset_precomputed = [dataset.precomputed_data[i] for i in indices]
    return PyGSMILESDataset(subset_smiles, subset_targets, subset_precomputed)


def check_and_create_hdf5_directories(args) -> None:
    """Create directories for HDF5 files if they don't exist yet."""
    # Handle separate HDF5 files (default behavior)
    for path in [args.train_hdf5, args.val_hdf5, args.test_hdf5]:
        if path:
            dirname = os.path.dirname(path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)