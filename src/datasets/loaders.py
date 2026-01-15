# datasets/loaders.py
"""
Data loader creation and collate functions for molecular datasets.
"""

from torch.utils.data import DataLoader
from .molecular import PyGSMILESDataset, HDF5MolecularIterableDataset, MolecularBatch


def iterable_collate_fn(batch_list):
    """Collate function for iterable datasets, filtering out None values."""
    filtered = [b for b in batch_list if b is not None]
    if len(filtered) == 0:
        return None
    return MolecularBatch.from_data_list(filtered)


def create_pyg_dataloader(
    dataset: PyGSMILESDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler=None
):
    """
    Creates a PyTorch DataLoader for an InMemoryDataset.
    
    Args:
        dataset: PyG dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        sampler: Optional sampler (e.g., for distributed training)
        
    Returns:
        DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        collate_fn=MolecularBatch.from_data_list,
        sampler=sampler
    )

def create_iterable_pyg_dataloader(
    hdf5_path: str, 
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    shuffle_buffer_size: int,
    ddp_enabled: bool = False,
    rank: int = 0,
    world_size: int = 1,
    preprocessing_pipeline = None
):
    """
    Creates a DataLoader for an HDF5MolecularIterableDataset.
    
    CRITICAL: This function checks if HDF5 contains preprocessed or raw data
    and ONLY applies preprocessing_pipeline if data is RAW.
    """
    import h5py
    
    # Check if HDF5 contains preprocessed data
    data_is_preprocessed = None  # Explicitly None to catch missing metadata
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'metadata' in f:
                metadata = f['metadata']
                data_is_preprocessed = metadata.attrs.get('preprocessing_applied', None)
                
                if data_is_preprocessed is None:
                    # Check legacy format
                    data_is_preprocessed = metadata.attrs.get('sae_applied', False)
                    
        if data_is_preprocessed is None:
            raise ValueError(
                f"Cannot determine preprocessing status of HDF5 file: {hdf5_path}\n\n"
                "The file is missing 'preprocessing_applied' metadata.\n"
                "This usually means the file was created with an old version.\n\n"
                "SOLUTION: Recreate the HDF5 file using current code:\n"
                f"  python create_inference_hdf5.py --model_path MODEL.pth \\\n"
                f"    --input_csv data.csv --output_hdf5 {hdf5_path}"
            )
    except OSError as e:
        raise ValueError(
            f"Cannot read HDF5 file: {hdf5_path}\n"
            f"Error: {e}\n\n"
            "The file may be corrupted or still being written.\n"
            "If training just started, wait a moment and try again."
        )
    
    # CRITICAL: Only pass preprocessing_pipeline if data is RAW
    pipeline_to_use = None
    if data_is_preprocessed:
        if preprocessing_pipeline is not None and rank == 0:
            print(f"[DataLoader] ✓ HDF5 contains PREPROCESSED data")
            print(f"[DataLoader]   → Preprocessing will NOT be applied (would corrupt data!)")
        pipeline_to_use = None  # Don't apply preprocessing again!
    else:
        if rank == 0:
            print(f"[DataLoader] ✓ HDF5 contains RAW data")
            print(f"[DataLoader]   → Preprocessing will be applied during loading")
        pipeline_to_use = preprocessing_pipeline
    
    dataset = HDF5MolecularIterableDataset(
        hdf5_path=hdf5_path, 
        shuffle=shuffle, 
        buffer_size=shuffle_buffer_size,
        ddp_enabled=ddp_enabled,
        rank=rank,
        world_size=world_size,
        preprocessing_pipeline=pipeline_to_use  # Only if data is raw
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=iterable_collate_fn
    )