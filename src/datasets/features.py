# datasets/features.py
"""
Molecular feature computation and processing.
"""

import time
import pickle
import random
from typing import Any
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
from utils.logging import get_logger

logger = get_logger(__name__)

# Mapping from CIP designation to signed scalar value
# R/S are standard designations; r/s are pseudo-asymmetric
CHIRALITY_SIGNS = {'R': 1.0, 'S': -1.0, 'r': 0.5, 's': -0.5}


def get_canonical_ranks(mol) -> list[int]:
    """
    Get canonical (CIP-like) ranks for all atoms in the molecule.

    Uses Chem.CanonicalRankAtoms() which is more reliable than accessing
    the _CIPRank property directly. Lower rank = higher priority.

    Note: This is NOT true CIP ranking - it uses RDKit's canonical ordering
    which may differ from CIP for isotopes, nested chirality, etc.
    For ML purposes, consistency is more important than strict CIP compliance.

    Args:
        mol: RDKit molecule object with stereochemistry assigned

    Returns:
        List of canonical ranks, one per atom (lower = higher priority)

    Raises:
        ValueError: If mol is None
    """
    if mol is None:
        raise ValueError("Cannot compute canonical ranks for None molecule")
    if mol.GetNumAtoms() == 0:
        return []
    return list(Chem.CanonicalRankAtoms(mol, breakTies=True))


def classify_pyramidal_hetero(mol, atom_idx: int) -> str | None:
    """
    Classify pyramidal heteroatom centers that can be chiral.

    Pyramidal centers have 3 explicit neighbors + 1 lone pair, which acts
    as a virtual 4th substituent for chirality determination.

    Args:
        mol: RDKit molecule object
        atom_idx: Index of the atom to classify

    Returns:
        'sulfoxide' - R₂S=O (stable ~40 kcal/mol inversion barrier)
        'sulfonium' - R₃S⁺ (configurationally stable, e.g., SAM derivatives)
        'selenoxide' - R₂Se=O (stable ~35 kcal/mol inversion barrier)
        'phosphine_oxide' - R₃P=O (configurationally stable)
        'phosphine_pyramidal' - R₃P (may have low barrier, include with caution)
        'arsine_pyramidal' - R₃As (higher barrier than P, ~40-45 kcal/mol)
        'quaternary_N' - R₄N⁺ (truly tetrahedral, no lone pair)
        'aziridine' - N in 3-membered ring (high inversion barrier ~15 kcal/mol)
        None - not a stable chiral center (e.g., tertiary amines)
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    Z = atom.GetAtomicNum()
    neighbors = list(atom.GetNeighbors())
    charge = atom.GetFormalCharge()

    # Sulfur (Z=16)
    if Z == 16:
        # Sulfonium ion R₃S⁺ - configurationally stable (e.g., SAM)
        if charge == 1 and len(neighbors) == 3:
            return 'sulfonium'
        # Sulfoxide R₂S=O - check for S=O double bond
        if len(neighbors) == 3 and charge == 0:
            for n in neighbors:
                bond = mol.GetBondBetweenAtoms(atom_idx, n.GetIdx())
                if bond is not None:
                    if (bond.GetBondType() == Chem.BondType.DOUBLE and
                        n.GetAtomicNum() == 8):
                        return 'sulfoxide'

    # Selenium (Z=34) - selenoxides R₂Se=O
    if Z == 34 and len(neighbors) == 3 and charge == 0:
        for n in neighbors:
            bond = mol.GetBondBetweenAtoms(atom_idx, n.GetIdx())
            if bond is not None:
                if (bond.GetBondType() == Chem.BondType.DOUBLE and
                    n.GetAtomicNum() == 8):
                    return 'selenoxide'

    # Arsenic (Z=33) - pyramidal arsines R₃As
    # Higher inversion barrier than phosphorus (~40-45 kcal/mol)
    if Z == 33 and len(neighbors) == 3 and charge == 0:
        return 'arsine_pyramidal'

    # Phosphorus (Z=15)
    if Z == 15:
        if len(neighbors) == 4 and charge == 0:
            # P(V) with 4 neighbors - check specifically for P=O (not P=C or P=N)
            for n in neighbors:
                bond = mol.GetBondBetweenAtoms(atom_idx, n.GetIdx())
                if bond is not None:
                    if (bond.GetBondType() == Chem.BondType.DOUBLE and
                        n.GetAtomicNum() == 8):  # Must be oxygen!
                        return 'phosphine_oxide'
        elif len(neighbors) == 3 and charge == 0:
            # P(III) pyramidal phosphine
            # Note: May have low inversion barrier - include with caution
            return 'phosphine_pyramidal'

    # Nitrogen (Z=7)
    if Z == 7:
        if charge == 1 and len(neighbors) == 4:
            # Quaternary N⁺ - truly tetrahedral, no lone pair
            # NOTE: Uniqueness check (all 4 substituents different) is done in
            # compute_stereochemistry_features() since canonical_ranks are needed
            return 'quaternary_N'
        # Check for aziridine (N in 3-membered ring) - high inversion barrier
        if charge == 0 and len(neighbors) == 3:
            ring_info = mol.GetRingInfo()
            for ring in ring_info.AtomRings():
                if len(ring) == 3 and atom_idx in ring:
                    return 'aziridine'
        # Regular tertiary amine - fast inversion, NOT chiral

    return None


def _trace_cumulene_chain(mol, start_idx: int, visited: set) -> list[int] | None:
    """
    Trace a cumulene chain starting from a given atom.

    Cumulenes are chains of sp-hybridized carbons connected by cumulated
    double bonds: C=C=C (allene), C=C=C=C (butatriene), C=C=C=C=C, etc.

    Args:
        mol: RDKit molecule object
        start_idx: Starting atom index (must be part of cumulated double bond)
        visited: Set of already-visited atom indices (to avoid double-counting)

    Returns:
        List of atom indices forming the cumulene chain (ordered), or None if invalid
    """
    if start_idx in visited:
        return None

    chain = [start_idx]
    visited.add(start_idx)

    # Extend chain in both directions
    for direction in [0, 1]:
        current_idx = start_idx
        while True:
            atom = mol.GetAtomWithIdx(current_idx)
            neighbors = list(atom.GetNeighbors())

            # Find next atom in chain (connected by double bond, not already in chain)
            next_idx = None
            for neighbor in neighbors:
                n_idx = neighbor.GetIdx()
                if n_idx in visited or n_idx in chain:
                    continue

                bond = mol.GetBondBetweenAtoms(current_idx, n_idx)
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    # Check if neighbor is sp-hybridized (2 neighbors, both double bonds)
                    # or a terminal carbon (will have 2+ neighbors with only 1 double bond)
                    n_neighbors = list(neighbor.GetNeighbors())
                    n_double_count = sum(
                        1 for nn in n_neighbors
                        if mol.GetBondBetweenAtoms(n_idx, nn.GetIdx()) and
                           mol.GetBondBetweenAtoms(n_idx, nn.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                    )
                    if n_double_count >= 1:  # At least one double bond (terminal or internal)
                        next_idx = n_idx
                        break

            if next_idx is None:
                break

            # Add to chain
            if direction == 0:
                chain.append(next_idx)
            else:
                chain.insert(0, next_idx)
            visited.add(next_idx)
            current_idx = next_idx

            # Stop if we've reached a terminal carbon (only 1 double bond neighbor)
            next_atom = mol.GetAtomWithIdx(next_idx)
            next_neighbors = list(next_atom.GetNeighbors())
            next_double_count = sum(
                1 for nn in next_neighbors
                if mol.GetBondBetweenAtoms(next_idx, nn.GetIdx()) and
                   mol.GetBondBetweenAtoms(next_idx, nn.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            )
            if next_double_count == 1:  # Terminal - only 1 double bond
                break

    return chain if len(chain) >= 3 else None


def extract_allenes(mol, canonical_ranks: list[int]) -> dict:
    """
    Extract allene/cumulene axial chirality using double-bond pattern.

    Detects cumulated double bond chains (C=C=C, C=C=C=C=C, etc.) and
    extracts only odd-length chains which exhibit axial chirality.

    Chirality condition:
    - Odd-length chains (3, 5, 7...): chiral if terminal substituents differ
    - Even-length chains (4, 6, 8...): achiral (planar)

    For odd chains, the center is the middle carbon. Terminal substituents
    are the non-chain neighbors of the end carbons.

    Args:
        mol: RDKit molecule object with stereochemistry assigned
        canonical_ranks: Precomputed canonical ranks for all atoms

    Returns:
        Dictionary with:
        - allene_centers: List of central atom indices
        - allene_subs: List of [R1, R2, R3, R4] substituent indices
    """
    centers = []
    subs = []
    visited = set()

    # Find all cumulene chains
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom_idx in visited:
            continue

        neighbors = list(atom.GetNeighbors())
        # Look for atoms with at least 1 double bond (could be start of chain)
        double_bond_count = sum(
            1 for n in neighbors
            if mol.GetBondBetweenAtoms(atom_idx, n.GetIdx()) and
               mol.GetBondBetweenAtoms(atom_idx, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        )
        if double_bond_count == 0:
            continue

        # Try to trace a cumulene chain starting from this atom
        chain = _trace_cumulene_chain(mol, atom_idx, visited)
        if chain is None:
            continue

        chain_length = len(chain)

        # Only odd-length chains are chiral
        if chain_length % 2 == 0:
            continue

        # Get the center (middle atom of odd-length chain)
        center_idx = chain[chain_length // 2]

        # Get terminal carbons
        end1_idx = chain[0]
        end2_idx = chain[-1]

        # Get substituents on each end (excluding chain atoms)
        chain_set = set(chain)
        end1_subs = [n.GetIdx() for n in mol.GetAtomWithIdx(end1_idx).GetNeighbors()
                     if n.GetIdx() not in chain_set]
        end2_subs = [n.GetIdx() for n in mol.GetAtomWithIdx(end2_idx).GetNeighbors()
                     if n.GetIdx() not in chain_set]

        # Need exactly 2 substituents per end for chirality
        # (if less, it's achiral; if more, it's not a standard cumulene)
        if len(end1_subs) != 2 or len(end2_subs) != 2:
            continue

        # Sort substituents by canonical rank (lower = higher priority)
        end1_subs.sort(key=lambda n: canonical_ranks[n])
        end2_subs.sort(key=lambda n: canonical_ranks[n])

        centers.append(center_idx)
        subs.append(end1_subs + end2_subs)  # [R1, R2, R3, R4]

    return {
        'allene_centers': centers,
        'allene_subs': subs,
    }


def partial_parse_atomic_numbers(smiles: str) -> np.ndarray | None:
    """Quick parse of SMILES to get atomic numbers only."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
    except (ValueError, RuntimeError):
        return None
    nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return np.array(nums, dtype=np.int32)


def compute_sae_dict_from_atomic_numbers_list(
    atomic_numbers_list: list[np.ndarray],
    target_values: list[float],
    percentile_cutoff: float = 2.0
) -> dict[int, float]:
    """
    Compute Self-Atomic Energy (SAE) contribution for each atom type.

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

    logger.info(f"Fitting atomic contributions using {len(b_filt)} molecules (after percentile filtering)")
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


def compute_all(smiles: str, max_hops: int) -> dict[str, Any] | None:
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
        logger.warning(f"Invalid SMILES could not be parsed: '{smiles}'")
        return None

    # Add H's and assign stereochemistry
    try:
        mol = Chem.AddHs(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        processed_smi = Chem.MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True)
        # Compute canonical ranks for all atoms (used for consistent CIP-like ordering)
        canonical_ranks = get_canonical_ranks(mol)
    except Exception as e:
        logger.warning(f"Failed to process SMILES '{smiles}': {e}")
        return None

    # 1) Multi-hop BFS edges
    try:
        adj_matrix = Chem.GetAdjacencyMatrix(mol).astype(np.int32)
        adj_list_numba = build_numba_adjacency_list(adj_matrix)
        edge_indices_list = compute_multi_hop_edges_bfs_numba(adj_list_numba, max_hops)
        multi_hop_edges = edge_indices_list
    except Exception as e:
        logger.warning(f"Failed to compute multi-hop edges for SMILES '{smiles}': {e}")
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
    except Exception as e:
        logger.warning(f"Failed to compute atom features for SMILES '{smiles}': {e}")
        return None

    # Store atomic numbers in a single array
    try:
        atomic_numbers_array = np.array(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=np.int32
        )
    except Exception as e:
        logger.warning(f"Failed to extract atomic numbers for SMILES '{smiles}': {e}")
        return None

    # 3) Chiral centers - now storing [center, n1, n2, n3, n4] with CIP ordering
    # Supports both tetrahedral (4 neighbors) and pyramidal (3 neighbors + virtual LP)
    chiral_tensors = []     # List of (5,) arrays: [center_idx, n1, n2, n3, n4]
    chiral_signs = []       # List of floats: R=+1, S=-1, r=+0.5, s=-0.5, ?=0
    chiral_is_virtual_lp = []  # List of (4,) bool arrays: True if neighbor is virtual LP
    chiral_elements = []    # List of atomic numbers for the center atoms
    try:
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        processed_centers = set()  # Track processed center indices

        for center_idx, chirality in chiral_centers:
            center_atom = mol.GetAtomWithIdx(center_idx)
            neighbors = list(center_atom.GetNeighbors())
            element = center_atom.GetAtomicNum()

            # Standard tetrahedral centers (4 neighbors)
            if len(neighbors) == 4:
                # N+ uniqueness check: quaternary N+ is only chiral if all 4
                # substituents have different canonical ranks (e.g., tetramethylammonium
                # [N+(CH3)4] is NOT chiral because all methyls are equivalent)
                if element == 7 and center_atom.GetFormalCharge() == 1:
                    neighbor_ranks = [canonical_ranks[n.GetIdx()] for n in neighbors]
                    if len(set(neighbor_ranks)) != 4:
                        # Skip - symmetric N+ is not chiral
                        continue

                # Sort neighbors by canonical rank (lowest rank = highest priority first)
                neighbors.sort(key=lambda n: canonical_ranks[n.GetIdx()])
                neighbor_indices = [n.GetIdx() for n in neighbors]

                chiral_tensors.append(np.array(
                    [center_idx] + neighbor_indices, dtype=np.int32
                ))
                chiral_signs.append(CHIRALITY_SIGNS.get(chirality, 0.0))
                chiral_is_virtual_lp.append([False, False, False, False])
                chiral_elements.append(element)
                processed_centers.add(center_idx)

            # Pyramidal heteroatom centers (3 neighbors + virtual lone pair)
            elif len(neighbors) == 3:
                hetero_type = classify_pyramidal_hetero(mol, center_idx)
                if hetero_type is not None:
                    # Sort neighbors by canonical rank
                    neighbors.sort(key=lambda n: canonical_ranks[n.GetIdx()])
                    neighbor_indices = [n.GetIdx() for n in neighbors]

                    # Use center index as placeholder for virtual LP (4th position)
                    # The LP has lowest priority (highest rank) by convention
                    neighbor_indices.append(center_idx)  # Virtual LP placeholder

                    chiral_tensors.append(np.array(
                        [center_idx] + neighbor_indices, dtype=np.int32
                    ))
                    chiral_signs.append(CHIRALITY_SIGNS.get(chirality, 0.0))
                    chiral_is_virtual_lp.append([False, False, False, True])  # 4th is LP
                    chiral_elements.append(element)
                    processed_centers.add(center_idx)

        # Also check for potential stereocenters using FindPotentialStereo
        # This catches some cases FindMolChiralCenters may miss
        try:
            potential_stereo = list(Chem.FindPotentialStereo(mol))
            for si in potential_stereo:
                if si.type != Chem.StereoType.Atom_Tetrahedral:
                    continue
                center_idx = si.centeredOn
                if center_idx in processed_centers:
                    continue  # Already processed

                center_atom = mol.GetAtomWithIdx(center_idx)
                neighbors = list(center_atom.GetNeighbors())
                element = center_atom.GetAtomicNum()

                # Handle pyramidal heteroatoms detected by FindPotentialStereo
                if len(neighbors) == 3:
                    hetero_type = classify_pyramidal_hetero(mol, center_idx)
                    if hetero_type is not None:
                        neighbors.sort(key=lambda n: canonical_ranks[n.GetIdx()])
                        neighbor_indices = [n.GetIdx() for n in neighbors]
                        neighbor_indices.append(center_idx)  # Virtual LP

                        chiral_tensors.append(np.array(
                            [center_idx] + neighbor_indices, dtype=np.int32
                        ))
                        # Use descriptor to determine sign if available
                        if hasattr(si, 'descriptor') and si.descriptor:
                            desc_str = str(si.descriptor)
                            if 'CW' in desc_str or desc_str == 'Tet_CW':
                                chiral_signs.append(1.0)  # Clockwise ~ R
                            elif 'CCW' in desc_str or desc_str == 'Tet_CCW':
                                chiral_signs.append(-1.0)  # Counter-clockwise ~ S
                            else:
                                chiral_signs.append(0.0)  # Unassigned
                        else:
                            chiral_signs.append(0.0)
                        chiral_is_virtual_lp.append([False, False, False, True])
                        chiral_elements.append(element)
        except Exception as e:
            logger.debug(f"FindPotentialStereo failed for SMILES '{smiles}': {e}")

    except Exception as e:
        logger.warning(f"Failed to compute chiral centers for SMILES '{smiles}': {e}")
        chiral_tensors = []
        chiral_signs = []
        chiral_is_virtual_lp = []
        chiral_elements = []

    # 4) Cis/Trans bonds
    cis_bonds_list = []
    trans_bonds_list = []
    try:
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                stereo = bond.GetStereo()
                if stereo not in [Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE]:
                    # skip STEREONONE, STEREOANY, etc.
                    continue

                # Skip double bonds in rings smaller than 8 members
                # trans-Cyclooctene (8-membered) is the smallest cycloalkene where E isomer exists
                # Rings 3-7 are geometrically constrained to cis configuration only
                if bond.IsInRing():
                    ring_info = mol.GetRingInfo()
                    ring_sizes = [len(r) for r in ring_info.AtomRings()
                                  if bond.GetBeginAtomIdx() in r and bond.GetEndAtomIdx() in r]
                    if any(size < 8 for size in ring_sizes):
                        continue  # Skip rings where E isomer cannot exist

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

                # Identify the "low" substituent on each side using canonical ranks
                # Lower rank = higher priority, so we want the max rank for "low" priority
                s_low_candidates = [x for x in start_neighbors if x != s_high]
                if not s_low_candidates:
                    continue
                s_low = max(s_low_candidates, key=lambda idx: canonical_ranks[idx])

                e_low_candidates = [x for x in end_neighbors if x != e_high]
                if not e_low_candidates:
                    continue
                e_low = max(e_low_candidates, key=lambda idx: canonical_ranks[idx])

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
    except Exception as e:
        logger.warning(f"Failed to compute cis/trans bonds for SMILES '{smiles}': {e}")

    # 4b) Allene (axial) chirality - C=C=C pattern
    allene_data = {'allene_centers': [], 'allene_subs': []}
    try:
        allene_data = extract_allenes(mol, canonical_ranks)
    except Exception as e:
        logger.warning(f"Failed to compute allene chirality for SMILES '{smiles}': {e}")

    # 5) Total formal charge
    try:
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    except Exception as e:
        logger.warning(f"Failed to compute total charge for SMILES '{smiles}': {e}")
        total_charge = 0

    # 6) Convert atom features to index arrays
    mapped_atom_features = {
        'atom_type': [],
        'hydrogen_count': [],
        'degree': [],
        'hybridization': [],
    }

    try:
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
    except Exception as e:
        logger.warning(f"Failed to map atom features for SMILES '{smiles}': {e}")
        return None

    # chiral_tensors are already np.int32 arrays from the loop above
    cis_bonds_tensors = [np.array(x, dtype=np.int32) for x in cis_bonds_list]
    trans_bonds_tensors = [np.array(x, dtype=np.int32) for x in trans_bonds_list]

    # Convert allene data to numpy arrays
    allene_centers = np.array(allene_data['allene_centers'], dtype=np.int32)
    allene_subs = np.array(allene_data['allene_subs'], dtype=np.int32) if allene_data['allene_subs'] else np.array([], dtype=np.int32).reshape(0, 4)

    return {
        "multi_hop_edges": multi_hop_edges,
        "atom_features": mapped_atom_features,
        "chiral_tensors": chiral_tensors,  # Shape (M, 5): [center, n1, n2, n3, n4]
        "chiral_signs": np.array(chiral_signs, dtype=np.float32),  # R=+1, S=-1, etc.
        "chiral_is_virtual_lp": np.array(chiral_is_virtual_lp, dtype=np.bool_) if chiral_is_virtual_lp else np.array([], dtype=np.bool_).reshape(0, 4),  # (M, 4) mask for virtual LPs
        "chiral_elements": np.array(chiral_elements, dtype=np.int32),  # (M,) atomic numbers of centers
        "cis_bonds_tensors": cis_bonds_tensors,
        "trans_bonds_tensors": trans_bonds_tensors,
        "allene_centers": allene_centers,  # (M_all,) center atom indices
        "allene_subs": allene_subs,  # (M_all, 4) [R1, R2, R3, R4] substituent indices
        "total_charge": total_charge,
        "atomic_numbers": atomic_numbers_array,
        "processed_smiles": processed_smi
    }

def precompute_all_and_filter(
    smiles_list: list[str],
    target_values: list[Any],  # float or list[float]
    max_hops: int,
    num_workers: int = 4
) -> tuple[list[str], list[Any], list[dict[str, Any]]]:
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
    logger.info(f"Precomputing multi-hop edge + features for {len(smiles_list)} SMILES using {num_workers} workers...")
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
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
    discarded = len(smiles_list) - len(valid_smiles)
    logger.info(f"Kept {len(valid_smiles)} valid SMILES; discarded {discarded} invalid or unparseable")

    return valid_smiles, valid_targets, precomputed_data

def precompute_and_write_hdf5_parallel_chunked(
    smiles_list: list[str],
    target_values: list[Any],
    max_hops: int,
    hdf5_path: str,
    num_workers: int = 4,
    chunk_size: int = 1000,
    sae_subtasks: list[int] | None = None,
    task_type: str = "regression",
    multi_target_columns: list[str] | None = None,
    preprocessing_applied: bool = True,
) -> None:
    """
    FIXED: Proper multiprocessing cleanup to prevent memory leaks.
    """
    logger.info(f"Using {num_workers} workers")

    # CRITICAL FIX: Ensure parent directory exists and is properly named
    hdf5_path = os.path.abspath(hdf5_path)
    parent_dir = os.path.dirname(hdf5_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        logger.info(f"Created directory: {parent_dir}")

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

        logger.info(f"Writing data to HDF5 (parallel + chunked) => {hdf5_path}")
        
        if preprocessing_applied:
            logger.info("Target values are ALREADY PREPROCESSED (SAE + scaling applied)")
            logger.info("No additional SAE normalization will be applied")
        else:
            logger.info("Target values are RAW, SAE normalization will be applied if requested")

        # FIXED: Proper multiprocessing cleanup
        func_partial = partial(_worker_bfs, max_hops=max_hops)
        
        pool = None
        try:
            pool = Pool(num_workers)
            
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
                    if buffer_indices:
                        dset[buffer_indices[0] : buffer_indices[-1] + 1] = buffer
                        buffer = []
                        buffer_indices = []

        except Exception as e:
            logger.error(f"Error during parallel processing: {e}")
            raise
        
        finally:
            # FIXED: Proper pool cleanup
            if pool is not None:
                pool.close()      # No new tasks
                pool.terminate()  # Kill workers immediately
                pool.join()       # Wait for cleanup
            
            # FIXED: Explicit garbage collection
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
        
        logger.info(f"HDF5 file created successfully at {hdf5_path}")
        logger.info(f"Estimated valid molecules: {estimated_valid_pct:.1f}% (based on sample of {sample_size})")
        if preprocessing_applied:
            logger.info("Data stored with PREPROCESSED targets (ready for training)")
        else:
            logger.info("Data stored with RAW targets (preprocessing applied during HDF5 creation)")

# Worker Functions for Parallel Processing

def _worker_bfs(smiles_and_target: tuple[str, Any], max_hops: int) -> dict[str, Any] | None:
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


def _worker_process_smiles(item: tuple[int, str, int]) -> tuple[int, dict[str, Any] | None]:
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