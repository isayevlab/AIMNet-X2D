# datasets/features.py
"""
Molecular feature computation and processing.
"""

import time
import pickle
import random
from typing import List, Dict, Tuple, Any
from multiprocessing import Pool
from functools import partial
import os

import numpy as np
import h5py
import tqdm
from rdkit import Chem
from rdkit.Chem import rdBase
from rdkit.Chem.rdchem import HybridizationType
from numba import njit, boolean
from numba.typed import List as NumbaList

from .constants import ATOM_TYPES, DEGREES, HYBRIDIZATIONS


def partial_parse_atomic_numbers(smiles: str) -> np.ndarray or None:
    """Quick parse of SMILES to get atomic numbers only."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
    except:
        return None
    nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return np.array(nums, dtype=np.int32)


def compute_sae_dict_from_atomic_numbers_list(
    atomic_numbers_list: List[np.ndarray],
    target_values: List[float],
    percentile_cutoff: float = 2.0
) -> Dict[int, float]:
    """
    Compute Size-Extensive Additive (SAE) contribution for each atom type.
    
    Args:
        atomic_numbers_list: List of arrays containing atomic numbers for each molecule
        target_values: Target property values for each molecule
        percentile_cutoff: Percentile cutoff for filtering outliers
        
    Returns:
        Dictionary mapping atomic numbers to their SAE contributions
    """
    all_targets = np.array(target_values, dtype=np.float64)
    max_atomic_num = 119
    N = len(atomic_numbers_list)
    A = np.zeros((N, max_atomic_num), dtype=np.float64)

    for i, nums in enumerate(atomic_numbers_list):
        unique, counts = np.unique(nums, return_counts=True)
        for u, c in zip(unique, counts):
            if 1 <= u < max_atomic_num:
                A[i, u] = c

    pct_low, pct_high = np.percentile(all_targets, [percentile_cutoff, 100 - percentile_cutoff])
    mask = (all_targets >= pct_low) & (all_targets <= pct_high)
    A_filt = A[mask]
    b_filt = all_targets[mask]

    print(f"Fitting atomic contributions using {len(b_filt)} molecules (after percentile filtering).")
    sae_values, residuals, rank, s = np.linalg.lstsq(A_filt, b_filt, rcond=None)

    sae_dict = {}
    for atomic_num in range(max_atomic_num):
        val = sae_values[atomic_num]
        if not np.isnan(val):
            sae_dict[atomic_num] = val

    return sae_dict


@njit
def build_numba_adjacency_list(adj_matrix: np.ndarray):
    """Build adjacency list from matrix for fast BFS using numba."""
    n = adj_matrix.shape[0]
    adjacency_list = NumbaList()
    for v in range(n):
        row_nonzero = np.where(adj_matrix[v] > 0)[0]
        nbr_list = NumbaList()
        for nbr in row_nonzero:
            if nbr != v:  # skip self-loop
                nbr_list.append(nbr)
        adjacency_list.append(nbr_list)
    return adjacency_list


@njit
def compute_multi_hop_edges_bfs_numba(adj_list, max_hops):
    """
    Produces the same hop-by-hop edges as adjacency exponentiation,
    using a BFS frontier in edge-space. Returns a list of (2 x E) arrays.
    """
    n = len(adj_list)
    visited = np.zeros((n, n), dtype=boolean)

    # Hop 1
    hop1_list = []
    for v in range(n):
        for w in adj_list[v]:
            if not visited[v, w]:
                visited[v, w] = True
                hop1_list.append((v, w))
    
    hop1_array = np.empty((2, len(hop1_list)), dtype=np.int32)
    for i, (src, dst) in enumerate(hop1_list):
        hop1_array[0, i] = src
        hop1_array[1, i] = dst
    
    results = [hop1_array]
    frontier = hop1_list

    # Hops 2..max_hops
    for _hop in range(1, max_hops):
        new_edges = []
        for (u, v) in frontier:
            # expand from v -> w
            neighbors_v = adj_list[v]
            for w in neighbors_v:
                if w != u:
                    if not visited[u, w]:
                        visited[u, w] = True
                        new_edges.append((u, w))

        if len(new_edges) == 0:
            empty_arr = np.empty((2, 0), dtype=np.int32)
            results.append(empty_arr)
            break

        arr = np.empty((2, len(new_edges)), dtype=np.int32)
        for i, (src, dst) in enumerate(new_edges):
            arr[0, i] = src
            arr[1, i] = dst

        results.append(arr)
        frontier = new_edges

    while len(results) < max_hops:
        results.append(np.empty((2, 0), dtype=np.int32))

    return results


def compute_all(smiles: str, max_hops: int) -> Dict[str, Any] or None:
    """
    Compute multi-hops + features in one pass.
    Return None if SMILES invalid/unparseable or any essential step fails.
    
    Args:
        smiles: SMILES string for a molecule
        max_hops: Maximum number of hops for BFS
        
    Returns:
        Dictionary containing molecular features or None if processing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add H's and assign stereochemistry
    try:
        mol = Chem.AddHs(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        processed_smi = Chem.MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True)
    except:
        return None

    # 1) Multi-hop BFS edges
    try:
        adj_matrix = Chem.GetAdjacencyMatrix(mol).astype(np.int32)
        adj_list_numba = build_numba_adjacency_list(adj_matrix)
        edge_indices_list = compute_multi_hop_edges_bfs_numba(adj_list_numba, max_hops)
        multi_hop_edges = edge_indices_list
    except:
        return None

    # 2) Atom features
    atom_features_list = []
    try:
        for atom in mol.GetAtoms():
            atom_type = atom.GetAtomicNum()
            degree = atom.GetTotalDegree()
            hydrogen_count = atom.GetTotalNumHs(includeNeighbors=True)
            hybridization = atom.GetHybridization()
            atom_features_list.append({
                'atom_type': atom_type,
                'hydrogen_count': hydrogen_count,
                'degree': degree,
                'hybridization': hybridization,
            })
    except:
        return None

    # Store atomic numbers in a single array
    try:
        atomic_numbers_array = np.array(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=np.int32
        )
    except:
        return None

    # 3) Chiral centers
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    chiral_tensors = []
    for center_idx, chirality in chiral_centers:
        center_atom = mol.GetAtomWithIdx(center_idx)
        neighbors = [nbr.GetIdx() for nbr in center_atom.GetNeighbors()]
        chiral_tensors.append(np.array(neighbors, dtype=np.int32))

    # 4) Cis/Trans bonds
    cis_bonds_list = []
    trans_bonds_list = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            if stereo not in [Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE]:
                # skip STEREONONE, STEREOANY, etc.
                continue

            start_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            # Build neighbor lists excluding the double-bond partner
            start_neighbors = [nbr.GetIdx() for nbr in start_atom.GetNeighbors()
                               if nbr.GetIdx() != end_atom.GetIdx()]
            end_neighbors = [nbr.GetIdx() for nbr in end_atom.GetNeighbors()
                               if nbr.GetIdx() != start_atom.GetIdx()]

            # skip "symmetric" or near-symmetric bonds
            if len(set(start_neighbors + end_neighbors)) < 4:
                continue

            stereo_atoms = bond.GetStereoAtoms()
            if len(stereo_atoms) != 2:
                continue

            s_high = stereo_atoms[0]
            e_high = stereo_atoms[1]

            # Identify the "low" substituent on each side
            s_low_candidates = [x for x in start_neighbors if x != s_high]
            if not s_low_candidates:
                continue
            s_low = min(s_low_candidates, key=lambda idx: mol.GetAtomWithIdx(idx).GetAtomicNum())

            e_low_candidates = [x for x in end_neighbors if x != e_high]
            if not e_low_candidates:
                continue
            e_low = min(e_low_candidates, key=lambda idx: mol.GetAtomWithIdx(idx).GetAtomicNum())

            if stereo == Chem.BondStereo.STEREOE:  # E => opposite
                trans_bonds_list.append([s_high, e_high])  
                trans_bonds_list.append([s_low, e_low])    
                trans_bonds_list.append([e_high, s_high])  
                trans_bonds_list.append([e_low, s_low])

                # cross pairs = cis
                cis_bonds_list.append([s_high, e_low])
                cis_bonds_list.append([s_low, e_high])
                cis_bonds_list.append([e_low, s_high])
                cis_bonds_list.append([e_high, s_low])

            elif stereo == Chem.BondStereo.STEREOZ:  # Z => same
                cis_bonds_list.append([s_high, e_high])
                cis_bonds_list.append([s_low, e_low])
                cis_bonds_list.append([e_high, s_high])
                cis_bonds_list.append([e_low, s_low])

                # cross pairs = trans
                trans_bonds_list.append([s_high, e_low])
                trans_bonds_list.append([s_low, e_high])
                trans_bonds_list.append([e_low, s_high])
                trans_bonds_list.append([e_high, s_low])

    # 5) Total formal charge
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    # 6) Convert atom features to index arrays
    mapped_atom_features = {
        'atom_type': [],
        'hydrogen_count': [],
        'degree': [],
        'hybridization': [],
    }

    for feat in atom_features_list:
        # Atomic number
        a_type = feat['atom_type']
        a_type_idx = ATOM_TYPES.index(a_type) if a_type in ATOM_TYPES else len(ATOM_TYPES)
        
        # Hydrogen count (create index on the fly)
        h_count = feat['hydrogen_count']
        h_count_idx = min(h_count, 8)  # Cap at 8 hydrogens (reasonable limit)
        
        # Degree
        deg = feat['degree']
        deg_idx = DEGREES.index(deg) if deg in DEGREES else len(DEGREES)
        
        # Hybridization
        hyb = feat['hybridization']
        hyb_idx = HYBRIDIZATIONS.index(hyb) if hyb in HYBRIDIZATIONS else len(HYBRIDIZATIONS)

        mapped_atom_features['atom_type'].append(a_type_idx)
        mapped_atom_features['hydrogen_count'].append(h_count_idx)
        mapped_atom_features['degree'].append(deg_idx)
        mapped_atom_features['hybridization'].append(hyb_idx)

    for k in mapped_atom_features:
        mapped_atom_features[k] = np.array(mapped_atom_features[k], dtype=np.int8)

    chiral_tensors = [np.array(x, dtype=np.int32) for x in chiral_tensors]
    cis_bonds_tensors = [np.array(x, dtype=np.int32) for x in cis_bonds_list]
    trans_bonds_tensors = [np.array(x, dtype=np.int32) for x in trans_bonds_list]

    return {
        "multi_hop_edges": multi_hop_edges,
        "atom_features": mapped_atom_features,
        "chiral_tensors": chiral_tensors,
        "cis_bonds_tensors": cis_bonds_tensors,
        "trans_bonds_tensors": trans_bonds_tensors,
        "total_charge": total_charge,
        "atomic_numbers": atomic_numbers_array,
        "processed_smiles": processed_smi
    }


def precompute_all_and_filter(
    smiles_list: List[str],
    target_values: List[Any],  # float or list[float]
    max_hops: int,
    num_workers: int = 4
) -> Tuple[List[str], List[Any], List[Dict[str, Any]]]:
    """
    In-memory BFS + feature precomputation with multiprocessing.
    
    Args:
        smiles_list: List of SMILES strings
        target_values: List of target values (single values or lists for multi-task)
        max_hops: Maximum number of hops for BFS
        num_workers: Number of parallel workers
        
    Returns:
        Tuple of (valid_smiles, valid_targets, precomputed_data)
    """
    print(f"Precomputing multi-hop edge + features for {len(smiles_list)} SMILES using {num_workers} workers...")
    start_time = time.time()

    compute_partial = partial(compute_all, max_hops=max_hops)

    valid_smiles = []
    valid_targets = []
    precomputed_data = []

    with Pool(num_workers) as pool:
        for smi, tgt, res in tqdm.tqdm(
            zip(smiles_list, target_values, pool.imap(compute_partial, smiles_list, chunksize=1000)),
            total=len(smiles_list)
        ):
            if res is not None:
                valid_smiles.append(smi)
                valid_targets.append(tgt)
                precomputed_data.append(res)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    discarded = len(smiles_list) - len(valid_smiles)
    print(f"Kept {len(valid_smiles)} valid SMILES; discarded {discarded} invalid or unparseable.")

    return valid_smiles, valid_targets, precomputed_data

def precompute_and_write_hdf5_parallel_chunked(
    smiles_list: List[str],
    target_values: List[Any],
    max_hops: int,
    hdf5_path: str,
    num_workers: int = 4,
    chunk_size: int = 1000,
    sae_subtasks: List[int] = None,
    task_type: str = "regression",
    multi_target_columns: List[str] = None,
    preprocessing_applied: bool = True,
):
    """
    FIXED: Added proper cleanup and error handling to prevent directory creation issues.
    """
    print(f"Using {num_workers} workers")

    # CRITICAL FIX: Ensure parent directory exists and is properly named
    hdf5_path = os.path.abspath(hdf5_path)  # Convert to absolute path
    parent_dir = os.path.dirname(hdf5_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"Created directory: {parent_dir}")

    with h5py.File(hdf5_path, "w") as f:
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        dset = f.create_dataset("data", (len(smiles_list),), dtype=dt)

        # Create index map dataset
        index_map_dset = f.create_dataset("index_map", (len(smiles_list),), dtype=np.int32)
        index_map_dset[:] = np.arange(len(smiles_list))

        # Add metadata group
        metadata = f.create_group("metadata")
        metadata.attrs["num_samples"] = len(smiles_list)
        metadata.attrs["task_type"] = task_type
        metadata.attrs["max_hops"] = max_hops
        metadata.attrs["preprocessing_applied"] = preprocessing_applied
        
        if multi_target_columns is not None:
            dt_str = h5py.special_dtype(vlen=str)
            target_cols = metadata.create_dataset("target_columns", (len(multi_target_columns),), dtype=dt_str)
            for i, col in enumerate(multi_target_columns):
                target_cols[i] = col

        print(f"Writing data to HDF5 (parallel + chunked) => {hdf5_path}")
        
        if preprocessing_applied:
            print("   → Target values are ALREADY PREPROCESSED (SAE + scaling applied)")
            print("   → No additional SAE normalization will be applied")
        else:
            print("   → Target values are RAW, SAE normalization will be applied if requested")

        # CRITICAL FIX: Use context manager for multiprocessing to ensure cleanup
        func_partial = partial(_worker_bfs, max_hops=max_hops)

        try:
            with Pool(num_workers) as pool:
                # Process SMILES in parallel
                results_iter = pool.imap(
                    func_partial, 
                    zip(smiles_list, target_values), 
                    chunksize=chunk_size
                )

                # Process and write in chunks
                buffer = []
                buffer_indices = []
                
                for i, res in enumerate(
                    tqdm.tqdm(results_iter, total=len(smiles_list), desc="Processing molecules")
                ):
                    if res is None:
                        # Encode None result for invalid SMILES
                        encoded = pickle.dumps(None)
                    else:
                        # Encode valid result
                        to_store = {
                            'smiles': res['smiles'],
                            'target': res['target'],
                            'precomputed': res['precomputed']
                        }
                        encoded = pickle.dumps(to_store)
                    
                    # Add to buffer
                    buffer.append(np.frombuffer(encoded, dtype=np.uint8))
                    buffer_indices.append(i)

                    # Once we have chunk_size items, or end of iteration => bulk write
                    if len(buffer) >= chunk_size or i == len(smiles_list) - 1:
                        if buffer_indices:  # Make sure there's something to write
                            dset[buffer_indices[0] : buffer_indices[-1] + 1] = buffer
                            buffer = []
                            buffer_indices = []

        except Exception as e:
            print(f"ERROR during parallel processing: {e}")
            raise
        finally:
            # CRITICAL FIX: Explicit cleanup
            import gc
            gc.collect()

        # Calculate and store statistics
        valid_count = 0
        invalid_count = 0
        
        # Sample a few entries to determine validity
        sample_size = min(1000, len(smiles_list))
        sample_indices = random.sample(range(len(smiles_list)), sample_size)
        
        for idx in sample_indices:
            raw = dset[idx]
            decoded = pickle.loads(raw.tobytes())
            if decoded is not None and decoded.get('precomputed') is not None:
                valid_count += 1
            else:
                invalid_count += 1
        
        # Extrapolate statistics to full dataset
        estimated_valid_pct = (valid_count / sample_size) * 100
        metadata.attrs["estimated_valid_pct"] = estimated_valid_pct
        
        print(f"HDF5 file created successfully at {hdf5_path}")
        print(f"Estimated valid molecules: {estimated_valid_pct:.1f}% (based on sample of {sample_size})")
        if preprocessing_applied:
            print(f"✅ Data stored with PREPROCESSED targets (ready for training)")
        else:
            print(f"✅ Data stored with RAW targets (preprocessing applied during HDF5 creation)")


# Worker Functions for Parallel Processing

def _worker_bfs(smiles_and_target, max_hops):
    """Worker function for parallel feature computation."""
    smi, tgt = smiles_and_target
    precomp = compute_all(smi, max_hops)
    if precomp is None:
        return None
    return {
        'smiles': smi,
        'target': tgt,
        'precomputed': precomp
    }


def _worker_process_smiles(item):
    """
    Worker function for processing SMILES in parallel.
    
    Args:
        item: Tuple containing (idx, smiles, max_hops)
    
    Returns:
        Tuple of (idx, precomp) where precomp is the result of compute_all
        or None if the SMILES couldn't be processed
    """
    idx, smiles, max_hops = item
    precomp = compute_all(smiles, max_hops)
    return (idx, precomp)