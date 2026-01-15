# datasets/__init__.py
"""
Datasets package for AIMNet-X2D.

This package contains all dataset-related functionality organized by purpose.
Handles molecular data loading, preprocessing, feature computation, and batching.
"""

# Core data structures
from .molecular import PyGSMILESDataset, HDF5MolecularIterableDataset, MolecularBatch
from .loaders import (
    create_pyg_dataloader,
    create_iterable_pyg_dataloader,
    iterable_collate_fn
)

# Data loading functions
from .io import (
    load_dataset_simple,
    load_dataset_multitask,
    split_dataset
)

# Feature computation
from .features import (
    compute_all,
    precompute_all_and_filter,
    precompute_and_write_hdf5_parallel_chunked,
    partial_parse_atomic_numbers,
    compute_sae_dict_from_atomic_numbers_list,
    _worker_bfs,
    _worker_process_smiles
)

# Constants
from .constants import ATOM_TYPES, DEGREES, HYBRIDIZATIONS

# Utilities
from .utils import subset_in_memory_dataset, check_and_create_hdf5_directories

__all__ = [
    # Core datasets
    "PyGSMILESDataset",
    "HDF5MolecularIterableDataset", 
    "MolecularBatch",
    
    # Data loaders
    "create_pyg_dataloader",
    "create_iterable_pyg_dataloader",
    "iterable_collate_fn",
    
    # I/O functions
    "load_dataset_simple",
    "load_dataset_multitask",
    "split_dataset",
    
    # Feature computation
    "compute_all",
    "precompute_all_and_filter",
    "precompute_and_write_hdf5_parallel_chunked",
    "partial_parse_atomic_numbers",
    "compute_sae_dict_from_atomic_numbers_list",
    
    # Constants
    "ATOM_TYPES",
    "DEGREES", 
    "HYBRIDIZATIONS",
    
    # Utilities
    "subset_in_memory_dataset",
    "check_and_create_hdf5_directories",
]