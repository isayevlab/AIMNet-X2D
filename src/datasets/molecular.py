# datasets/molecular.py
"""
Core molecular dataset classes and batch handling.
"""

import os
import pickle
import random
import math
import torch
import torch.utils.data
import h5py
import numpy as np
from typing import Any
from torch_geometric.data import InMemoryDataset, Data, Batch

from .constants import DEFAULT_SHUFFLE_BUFFER_SIZE, DEFAULT_RANDOM_SEED
from utils.logging import get_logger

logger = get_logger(__name__)


class PyGSMILESDataset(InMemoryDataset):
    """
    InMemoryDataset that stores each molecule as a Data object.
    
    Args:
        smiles_list: List of SMILES strings
        targets: List of target values (single values or lists for multi-task)
        precomputed_data: List of dictionaries with precomputed molecular features
        transform: PyG transform to apply
        pre_transform: PyG pre-transform to apply
    """

    def __init__(
        self,
        smiles_list: list[str],
        targets: list[Any],
        precomputed_data: list[dict[str, Any]],
        transform=None,
        pre_transform=None,
        **kwargs
    ) -> None:
        self.smiles_list = smiles_list
        self.targets = targets
        self.precomputed_data = precomputed_data
        super().__init__(None, transform, pre_transform)
        self.process()
        self.data = None
        self.slices = None

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return []

    def download(self) -> None:
        pass

    def process(self) -> None:
        self.data_list = []
        for i, smiles in enumerate(self.smiles_list):
            precomp = self.precomputed_data[i]

            num_atoms = precomp['atom_features']['atom_type'].shape[0]
            x_dummy = torch.ones((num_atoms, 1), dtype=torch.float)

            data = Data()
            data.x = x_dummy
            data.smiles = precomp["processed_smiles"]

            t = self.targets[i]
            if isinstance(t, list) or isinstance(t, np.ndarray):
                data.target = torch.tensor(t, dtype=torch.float)
            else:
                data.target = torch.tensor([t], dtype=torch.float)

            # Multi_hop edges
            data.multi_hop_edges = [torch.from_numpy(e).long() for e in precomp["multi_hop_edges"]]

            # Atom features map
            atom_feats_map = {}
            for k, arr in precomp["atom_features"].items():
                atom_feats_map[k] = torch.from_numpy(arr).long()
            data.atom_features_map = atom_feats_map

            # Chirality / cis / trans
            # chiral_tensors now have shape (5,): [center_idx, n1, n2, n3, n4]
            data.chiral_tensors = [torch.from_numpy(x).long() for x in precomp["chiral_tensors"]]
            data.chiral_signs = torch.from_numpy(precomp["chiral_signs"]).float()
            data.chiral_is_virtual_lp = torch.from_numpy(precomp["chiral_is_virtual_lp"]).bool()
            data.chiral_elements = torch.from_numpy(precomp["chiral_elements"]).long()
            data.cis_bonds_tensors = [torch.from_numpy(x).long() for x in precomp["cis_bonds_tensors"]]
            data.trans_bonds_tensors = [torch.from_numpy(x).long() for x in precomp["trans_bonds_tensors"]]

            # Allene (axial) chirality
            data.allene_centers = torch.from_numpy(precomp["allene_centers"]).long()
            data.allene_subs = torch.from_numpy(precomp["allene_subs"]).long()

            data.total_charge = torch.tensor([precomp["total_charge"]], dtype=torch.float)
            data.atomic_numbers = torch.from_numpy(precomp["atomic_numbers"]).long()

            self.data_list.append(data)

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


class HDF5MolecularIterableDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset for large-scale molecular data stored in HDF5 format.
    
    Allows efficient streaming of molecular data without loading everything into memory.
    Handles both raw and preprocessed data depending on what's stored in the HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        shuffle: Whether to shuffle data during iteration
        buffer_size: Size of shuffle buffer
        ddp_enabled: Whether distributed data parallel is enabled
        rank: Process rank for DDP
        world_size: Total number of processes for DDP
        fold_indices: Indices to use for cross-validation
        cv_fold: Current cross-validation fold
        seed: Random seed for shuffling (default: 42)
        preprocessing_pipeline: Optional preprocessing pipeline for raw data
    """
    def __init__(self,
                 hdf5_path: str,
                 shuffle: bool = False,
                 buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
                 ddp_enabled: bool = False,
                 rank: int = 0,
                 world_size: int = 1,
                 fold_indices: list[int] | None = None,
                 cv_fold: int | None = None,
                 seed: int = DEFAULT_RANDOM_SEED,
                 preprocessing_pipeline: Any | None = None) -> None:
        super().__init__()
        self.hdf5_path = hdf5_path
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.ddp_enabled = ddp_enabled
        self.rank = rank
        self.world_size = world_size
        self.fold_indices = fold_indices
        self.cv_fold = cv_fold
        self.seed = seed
        self.preprocessing_pipeline = preprocessing_pipeline
        self._length = None
        self.index_map = None
        self.cv_splits = None
        self.data_is_preprocessed = True  # Assume preprocessed by default
        
        # Pre-load metadata outside the iterator
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load HDF5 metadata only once to avoid repeated file access."""
        with h5py.File(self.hdf5_path, "r") as f:
            self._length = f["data"].shape[0]
            
            # Load index_map if available
            if "index_map" in f:
                self.index_map = f["index_map"][:]
            else:
                self.index_map = np.arange(self._length)
            
            # Load cv_splits if they exist
            if "cv_splits" in f:
                self.cv_splits = {}
                cv_splits_group = f["cv_splits"]
                for fold_name in cv_splits_group:
                    fold_group = cv_splits_group[fold_name]
                    self.cv_splits[fold_name] = {
                        "train_indices": fold_group["train_indices"][:],
                        "val_indices": fold_group["val_indices"][:]
                    }
            
            # Check if data is preprocessed by looking at metadata
            if "metadata" in f:
                metadata = f["metadata"]
                if "preprocessing_applied" in metadata.attrs:
                    self.data_is_preprocessed = metadata.attrs["preprocessing_applied"]
                elif "sae" in metadata:
                    # Check SAE metadata to determine if preprocessing was applied
                    sae_group = metadata["sae"]
                    self.data_is_preprocessed = sae_group.attrs.get("applied", False)
                else:
                    # If no clear indication, assume data needs preprocessing
                    self.data_is_preprocessed = False
            else:
                # No metadata, assume raw data
                self.data_is_preprocessed = False

        # If cv_fold is specified, use indices from that fold
        if self.cv_fold is not None and self.cv_splits is not None:
            fold_key = f"fold_{self.cv_fold}"
            if fold_key in self.cv_splits:
                self.fold_indices = self.cv_splits[fold_key]["train_indices"]

    def __len__(self) -> int:
        # Calculate length based on fold_indices if available
        if self.fold_indices is not None:
            return len(self.fold_indices)
        elif self.ddp_enabled:
            # Ensure even distribution of samples across processes
            return self._length // self.world_size + (1 if self._length % self.world_size > self.rank else 0)
        else:
            return self._length

    def __iter__(self):
        # Get worker info
        worker_info = torch.utils.data.get_worker_info()
        
        # Set deterministic seed for this rank/worker combination
        if self.shuffle:
            epoch_seed = torch.initial_seed()
            process_seed = self.seed + self.rank * 10000
            combined_seed = (epoch_seed + process_seed) % (2**32 - 1)
            rng = random.Random(combined_seed)
        
        # Determine indices to process
        if self.fold_indices is not None:
            total_indices = self.fold_indices
        else:
            total_indices = list(range(len(self.index_map)))

        # Shuffle indices once at the dataset level (if needed)
        if self.shuffle:
            indices_copy = total_indices.copy()
            rng.shuffle(indices_copy)
            total_indices = indices_copy

        # Distribute indices across processes (DDP)
        if self.ddp_enabled:
            num_samples = len(total_indices)
            chunk_size = int(math.ceil(num_samples / float(self.world_size)))
            
            start_idx = self.rank * chunk_size
            end_idx = min(start_idx + chunk_size, num_samples)
            process_indices = total_indices[start_idx:end_idx]
        else:
            process_indices = total_indices

        # Further distribute across workers within this process
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            per_worker = int(math.ceil(len(process_indices) / float(num_workers)))
            worker_start = worker_id * per_worker
            worker_end = min(worker_start + per_worker, len(process_indices))
            
            final_indices = process_indices[worker_start:worker_end]
        else:
            final_indices = process_indices

        # Open the HDF5 file for reading
        f = h5py.File(self.hdf5_path, "r")
        dset = f["data"]
        
        try:
            # Generate samples
            for idx in final_indices:
                mapped_idx = self.index_map[idx]
                raw = dset[mapped_idx]
                try:
                    decoded = pickle.loads(raw.tobytes())
                    data_obj = self._build_data_object(decoded)
                    if data_obj is not None:
                        yield data_obj
                except Exception as e:
                    logger.warning(f"Error processing index {idx} (mapped to {mapped_idx}): {str(e)}")
                    continue
        finally:
            f.close()

    def _build_data_object(self, item: dict[str, Any] | None) -> Data | None:
        """Convert raw HDF5 result into a PyG Data object."""
        if item is None or (item.get('precomputed') is None):
            return None

        smi = item['smiles']
        tgt = item['target']
        precomp = item['precomputed']

        # Apply preprocessing to target if needed and pipeline is provided
        if not self.data_is_preprocessed and self.preprocessing_pipeline is not None:
            # Apply preprocessing to this single target
            # Note: This is not ideal as preprocessing should be done on the full dataset
            # for proper statistics, but this is a fallback for compatibility
            if isinstance(tgt, list):
                processed_tgt = self.preprocessing_pipeline.transform([smi], [tgt])
                if len(processed_tgt) > 0:
                    tgt = processed_tgt[0]
            else:
                processed_tgt = self.preprocessing_pipeline.transform([smi], [tgt])
                if len(processed_tgt) > 0:
                    tgt = processed_tgt[0]

        num_atoms = precomp['atom_features']['atom_type'].shape[0]
        x_dummy = torch.ones((num_atoms, 1), dtype=torch.float)

        data_obj = Data()
        data_obj.x = x_dummy
        # Use processed SMILES (canonical with explicit H) for consistency with in-memory dataset
        data_obj.smiles = precomp.get("processed_smiles", smi)

        if isinstance(tgt, list):
            data_obj.target = torch.tensor(tgt, dtype=torch.float)
        else:
            data_obj.target = torch.tensor([tgt], dtype=torch.float)

        data_obj.multi_hop_edges = [
            torch.from_numpy(e).long() for e in precomp["multi_hop_edges"]
        ]

        atom_feats_map = {}
        for k, arr in precomp["atom_features"].items():
            atom_feats_map[k] = torch.from_numpy(arr).long()
        data_obj.atom_features_map = atom_feats_map

        # chiral_tensors now have shape (5,): [center_idx, n1, n2, n3, n4]
        data_obj.chiral_tensors = [
            torch.from_numpy(x).long() for x in precomp["chiral_tensors"]
        ]
        data_obj.chiral_signs = torch.from_numpy(precomp["chiral_signs"]).float()
        data_obj.chiral_is_virtual_lp = torch.from_numpy(precomp["chiral_is_virtual_lp"]).bool()
        data_obj.chiral_elements = torch.from_numpy(precomp["chiral_elements"]).long()
        data_obj.cis_bonds_tensors = [
            torch.from_numpy(x).long() for x in precomp["cis_bonds_tensors"]
        ]
        data_obj.trans_bonds_tensors = [
            torch.from_numpy(x).long() for x in precomp["trans_bonds_tensors"]
        ]

        # Allene (axial) chirality
        data_obj.allene_centers = torch.from_numpy(precomp["allene_centers"]).long()
        data_obj.allene_subs = torch.from_numpy(precomp["allene_subs"]).long()

        data_obj.total_charge = torch.tensor([precomp["total_charge"]], dtype=torch.float)
        data_obj.atomic_numbers = torch.from_numpy(precomp["atomic_numbers"]).long()

        return data_obj


class MolecularBatch(Batch):
    """
    Custom Batch class for molecular graphs.
    
    Handles proper batching of molecular graphs with BFS-based features
    and stereo/chemical information.
    """
    @staticmethod
    def from_data_list(data_list: list[Data]) -> Batch:
        """
        Collates multiple Data objects into a single batch
        with BFS offset/padding logic + offset chirality/cis/trans indices.
        """
        batch_size = len(data_list)
        if batch_size == 0:
            return Batch()

        num_hops = len(data_list[0].multi_hop_edges)

        # 1) Number of atoms per sample + offsets
        num_atoms_per_sample = [d.x.size(0) for d in data_list]
        atom_offsets = torch.cat([
            torch.tensor([0], dtype=torch.long),
            torch.cumsum(torch.tensor(num_atoms_per_sample[:-1], dtype=torch.long), dim=0)
        ])

        # 2) Collect + offset chirality/cis/trans
        # chiral_tensors now have shape (5,): [center_idx, n1, n2, n3, n4]
        shifted_chiral_centers = []
        chiral_signs_list = []
        chiral_is_virtual_lp_list = []
        chiral_elements_list = []
        tet_batch_list = []  # Track molecule index for each chiral center
        shifted_cis_bonds = []
        cis_batch_list = []  # Track molecule index for each cis bond
        shifted_trans_bonds = []
        trans_batch_list = []  # Track molecule index for each trans bond
        shifted_allene_centers = []
        shifted_allene_subs = []
        allene_batch_list = []  # Track molecule index for each allene center
        for i, d in enumerate(data_list):
            offset = atom_offsets[i]
            # Chirality - now expects shape (5,)
            num_chiral_in_mol = 0
            for ch in d.chiral_tensors:
                if ch.size(0) == 5:  # [center, n1, n2, n3, n4]
                    shifted_chiral_centers.append(ch + offset)
                    num_chiral_in_mol += 1
                elif ch.size(0) != 0:
                    logger.warning(f"Unexpected chiral tensor size {ch.size(0)}, expected 5. Skipping.")
            tet_batch_list.extend([i] * num_chiral_in_mol)
            # Collect chiral signs for this molecule
            if hasattr(d, 'chiral_signs') and d.chiral_signs.numel() > 0:
                chiral_signs_list.extend(d.chiral_signs.tolist())
            # Collect virtual LP mask - validate shape is (N, 4)
            if hasattr(d, 'chiral_is_virtual_lp') and d.chiral_is_virtual_lp.numel() > 0:
                if d.chiral_is_virtual_lp.dim() == 2 and d.chiral_is_virtual_lp.shape[1] == 4:
                    chiral_is_virtual_lp_list.append(d.chiral_is_virtual_lp)
                else:
                    logger.warning(f"Unexpected chiral_is_virtual_lp shape {d.chiral_is_virtual_lp.shape}, expected (N, 4). Skipping.")
            # Collect chiral center elements
            if hasattr(d, 'chiral_elements') and d.chiral_elements.numel() > 0:
                chiral_elements_list.append(d.chiral_elements)
            # Cis
            num_cis = len(d.cis_bonds_tensors)
            shifted_cis_bonds.extend(bond + offset for bond in d.cis_bonds_tensors)
            cis_batch_list.extend([i] * num_cis)
            # Trans
            num_trans = len(d.trans_bonds_tensors)
            shifted_trans_bonds.extend(bond + offset for bond in d.trans_bonds_tensors)
            trans_batch_list.extend([i] * num_trans)
            # Allene centers and substituents
            if hasattr(d, 'allene_centers') and d.allene_centers.numel() > 0:
                num_allenes = d.allene_centers.numel()
                shifted_allene_centers.append(d.allene_centers + offset)
                allene_batch_list.extend([i] * num_allenes)
            if hasattr(d, 'allene_subs') and d.allene_subs.numel() > 0:
                shifted_allene_subs.append(d.allene_subs + offset)

        final_tetrahedral_chiral_tensor = (
            torch.stack(shifted_chiral_centers, dim=0)
            if shifted_chiral_centers
            else torch.empty((0, 5), dtype=torch.long)
        )
        chiral_signs_tensor = (
            torch.tensor(chiral_signs_list, dtype=torch.float)
            if chiral_signs_list
            else torch.empty((0,), dtype=torch.float)
        )
        chiral_is_virtual_lp_tensor = (
            torch.cat(chiral_is_virtual_lp_list, dim=0)
            if chiral_is_virtual_lp_list
            else torch.empty((0, 4), dtype=torch.bool)
        )
        chiral_elements_tensor = (
            torch.cat(chiral_elements_list, dim=0)
            if chiral_elements_list
            else torch.empty((0,), dtype=torch.long)
        )
        cis_bonds_tensor = (
            torch.stack(shifted_cis_bonds, dim=0)
            if shifted_cis_bonds
            else torch.empty((0, 2), dtype=torch.long)
        )
        trans_bonds_tensor = (
            torch.stack(shifted_trans_bonds, dim=0)
            if shifted_trans_bonds
            else torch.empty((0, 2), dtype=torch.long)
        )
        allene_centers_tensor = (
            torch.cat(shifted_allene_centers, dim=0)
            if shifted_allene_centers
            else torch.empty((0,), dtype=torch.long)
        )
        allene_subs_tensor = (
            torch.cat(shifted_allene_subs, dim=0)
            if shifted_allene_subs
            else torch.empty((0, 4), dtype=torch.long)
        )

        # Create batch index tensors for stereochemistry elements
        # These track which molecule in the batch each stereo element belongs to
        tet_batch_tensor = torch.tensor(tet_batch_list, dtype=torch.long) if tet_batch_list else torch.empty((0,), dtype=torch.long)
        cis_batch_tensor = torch.tensor(cis_batch_list, dtype=torch.long) if cis_batch_list else torch.empty((0,), dtype=torch.long)
        trans_batch_tensor = torch.tensor(trans_batch_list, dtype=torch.long) if trans_batch_list else torch.empty((0,), dtype=torch.long)
        allene_batch_tensor = torch.tensor(allene_batch_list, dtype=torch.long) if allene_batch_list else torch.empty((0,), dtype=torch.long)

        # Note: features.py already adds both directions for cis/trans pairs
        # so no reversal needed here
        final_cis_tensor = cis_bonds_tensor if cis_bonds_tensor.numel() > 0 else torch.empty((0, 2), dtype=torch.long)
        final_trans_tensor = trans_bonds_tensor if trans_bonds_tensor.numel() > 0 else torch.empty((0, 2), dtype=torch.long)

        # Validate cis/trans tensor shapes
        if final_cis_tensor.numel() > 0 and (final_cis_tensor.dim() != 2 or final_cis_tensor.shape[1] != 2):
            logger.warning(f"Unexpected cis_tensor shape {final_cis_tensor.shape}, expected (N, 2). Resetting to empty.")
            final_cis_tensor = torch.empty((0, 2), dtype=torch.long)
        if final_trans_tensor.numel() > 0 and (final_trans_tensor.dim() != 2 or final_trans_tensor.shape[1] != 2):
            logger.warning(f"Unexpected trans_tensor shape {final_trans_tensor.shape}, expected (N, 2). Resetting to empty.")
            final_trans_tensor = torch.empty((0, 2), dtype=torch.long)

        # Validate allene center/subs correspondence
        if allene_centers_tensor.numel() > 0 or allene_subs_tensor.numel() > 0:
            num_allene_centers = allene_centers_tensor.shape[0] if allene_centers_tensor.dim() >= 1 else 0
            num_allene_subs = allene_subs_tensor.shape[0] if allene_subs_tensor.dim() >= 1 else 0
            if num_allene_centers != num_allene_subs:
                logger.warning(
                    f"Allene centers ({num_allene_centers}) and subs ({num_allene_subs}) count mismatch. "
                    "Resetting both to empty."
                )
                allene_centers_tensor = torch.empty((0,), dtype=torch.long)
                allene_subs_tensor = torch.empty((0, 4), dtype=torch.long)
                allene_batch_tensor = torch.empty((0,), dtype=torch.long)

        # 4) Concatenate atom features
        feature_keys = list(data_list[0].atom_features_map.keys())
        all_atom_features_map = {
            key: torch.cat([d.atom_features_map[key] for d in data_list], dim=0)
            for key in feature_keys
        }

        # 5) Create batch_indices
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long),
            torch.tensor(num_atoms_per_sample, dtype=torch.long)
        )

        # 6) Concatenate targets and total_charges
        first_target_dim = data_list[0].target.shape[0]
        is_all_same_dim = all(d.target.shape[0] == first_target_dim for d in data_list)
        if is_all_same_dim:
            targets = torch.stack([d.target for d in data_list], dim=0)
        else:
            targets = [d.target for d in data_list]

        total_charges = torch.cat([d.total_charge for d in data_list], dim=0)

        # 7) Collect smiles and atomic_numbers
        smiles_list = [d.smiles for d in data_list]
        batched_atomic_numbers = torch.cat([d.atomic_numbers for d in data_list], dim=0)

        # 8) Build final BFS edge indices
        shell_edge_indices_list = []
        for i, d in enumerate(data_list):
            offset_val = atom_offsets[i]
            for h_idx in range(num_hops):
                e = d.multi_hop_edges[h_idx]
                if e.numel() > 0:
                    shell_edge_indices_list.append(e + offset_val)

        if len(shell_edge_indices_list) > 0:
            shell_edge_indices = torch.cat(shell_edge_indices_list, dim=1).t()
        else:
            shell_edge_indices = torch.empty((0, 2), dtype=torch.long)

        # 9) Construct new Batch
        batch = Batch()
        batch.x = torch.cat([d.x for d in data_list], dim=0)
        batch.multi_hop_edge_indices = shell_edge_indices
        batch.batch_indices = batch_indices
        batch.atom_features_map = all_atom_features_map
        batch.targets = targets
        batch.total_charges = total_charges

        batch.final_tetrahedral_chiral_tensor = final_tetrahedral_chiral_tensor
        batch.chiral_signs = chiral_signs_tensor
        batch.chiral_is_virtual_lp = chiral_is_virtual_lp_tensor
        batch.chiral_elements = chiral_elements_tensor
        batch.tet_batch = tet_batch_tensor  # Molecule index for each chiral center
        batch.final_cis_tensor = final_cis_tensor
        batch.cis_batch = cis_batch_tensor  # Molecule index for each cis bond
        batch.final_trans_tensor = final_trans_tensor
        batch.trans_batch = trans_batch_tensor  # Molecule index for each trans bond
        batch.allene_centers = allene_centers_tensor
        batch.allene_subs = allene_subs_tensor
        batch.allene_batch = allene_batch_tensor  # Molecule index for each allene
        batch.smiles_list = smiles_list

        # For PyG usage
        batch.batch = batch_indices
        batch.atomic_numbers = batched_atomic_numbers

        return batch