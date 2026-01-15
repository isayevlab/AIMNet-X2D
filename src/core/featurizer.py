"""
Batch-native molecular featurization.

Optimized for batch processing with minimal per-molecule overhead.
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from .batch import MolecularGraphBatch
from datasets.features import (
    build_numba_adjacency_list,
    compute_multi_hop_edges_bfs_numba,
)
from datasets.constants import ATOM_TYPES, DEGREES, HYBRIDIZATIONS
from utils.logging import get_logger

logger = get_logger(__name__)

# Constants for feature encoding - must match datasets.constants
MAX_HYDROGEN_COUNT = 9  # 0-8 hydrogens
MAX_DEGREE = 6  # Matches DEGREES = list(range(6))


def _featurize_single(args: tuple[str, int]) -> dict[str, Any] | None:
    """
    Featurize single molecule (for parallel processing).

    Returns dict with numpy arrays, or None if invalid.
    """
    smiles, num_hops = args

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        num_atoms = mol.GetNumAtoms()

        # Atom features
        atom_types = np.zeros(num_atoms, dtype=np.int32)
        degrees = np.zeros(num_atoms, dtype=np.int32)
        hybridizations = np.zeros(num_atoms, dtype=np.int32)
        hydrogen_counts = np.zeros(num_atoms, dtype=np.int32)
        atomic_numbers = np.zeros(num_atoms, dtype=np.int32)

        for i, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            atomic_numbers[i] = atomic_num

            # Map atomic number to index
            if atomic_num in ATOM_TYPES:
                atom_types[i] = ATOM_TYPES.index(atomic_num)
            else:
                atom_types[i] = len(ATOM_TYPES)

            # Degree
            deg = atom.GetTotalDegree()
            if deg in DEGREES:
                degrees[i] = DEGREES.index(deg)
            else:
                degrees[i] = len(DEGREES)

            # Hybridization
            hyb = atom.GetHybridization()
            if hyb in HYBRIDIZATIONS:
                hybridizations[i] = HYBRIDIZATIONS.index(hyb)
            else:
                hybridizations[i] = len(HYBRIDIZATIONS) - 1

            # Hydrogen count
            hydrogen_counts[i] = min(atom.GetTotalNumHs(), MAX_HYDROGEN_COUNT - 1)

        # Build adjacency for BFS
        adj_matrix = Chem.GetAdjacencyMatrix(mol).astype(np.int32)
        adj_list_numba = build_numba_adjacency_list(adj_matrix)

        # Compute multi-hop edges
        multi_hop_edges = compute_multi_hop_edges_bfs_numba(adj_list_numba, num_hops)

        # Stereochemistry
        chiral_centers = []
        cis_bonds = []
        trans_bonds = []

        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

            chiral_info = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            for atom_idx, _ in chiral_info:
                neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]
                if len(neighbors) >= 4:
                    chiral_centers.append([atom_idx] + neighbors[:3])

            for bond in mol.GetBonds():
                stereo = bond.GetStereo()
                if stereo == Chem.BondStereo.STEREOZ:
                    cis_bonds.append([
                        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    ])
                elif stereo == Chem.BondStereo.STEREOE:
                    trans_bonds.append([
                        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    ])
        except Exception:
            pass

        return {
            "smiles": smiles,
            "atom_types": atom_types,
            "degrees": degrees,
            "hybridizations": hybridizations,
            "hydrogen_counts": hydrogen_counts,
            "atomic_numbers": atomic_numbers,
            "multi_hop_edges": multi_hop_edges,
            "chiral_centers": chiral_centers,
            "cis_bonds": cis_bonds,
            "trans_bonds": trans_bonds,
            "total_charge": Chem.GetFormalCharge(mol),
        }

    except Exception as e:
        logger.debug(f"Failed to featurize {smiles[:20]}...: {e}")
        return None


@dataclass
class BatchFeaturizer:
    """Batch-native molecular featurizer."""

    num_hops: int = 3
    num_workers: int = 4

    def featurize(
        self,
        smiles_list: list[str],
        targets: np.ndarray,
    ) -> MolecularGraphBatch:
        """Featurize batch of molecules."""
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        args = [(smi, self.num_hops) for smi in smiles_list]

        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                results = list(pool.map(_featurize_single, args))
        else:
            results = [_featurize_single(a) for a in args]

        valid_data = []
        valid_targets = []

        for i, result in enumerate(results):
            if result is not None:
                valid_data.append(result)
                valid_targets.append(targets[i])

        if not valid_data:
            raise ValueError("No valid molecules in batch")

        return self._stack_batch(valid_data, np.array(valid_targets))

    def _stack_batch(
        self,
        molecules: list[dict[str, Any]],
        targets: np.ndarray,
    ) -> MolecularGraphBatch:
        """Stack molecule data into batch tensors."""
        num_molecules = len(molecules)

        atom_counts = [len(m["atom_types"]) for m in molecules]
        ptr = torch.tensor([0] + list(np.cumsum(atom_counts)), dtype=torch.int64)
        total_atoms = ptr[-1].item()

        atom_types = torch.zeros(total_atoms, dtype=torch.int32)
        degrees = torch.zeros(total_atoms, dtype=torch.int32)
        hybridizations = torch.zeros(total_atoms, dtype=torch.int32)
        hydrogen_counts = torch.zeros(total_atoms, dtype=torch.int32)
        atomic_numbers = torch.zeros(total_atoms, dtype=torch.int64)
        batch_idx = torch.zeros(total_atoms, dtype=torch.int64)

        hop_edges_lists: list[list[torch.Tensor]] = [[] for _ in range(self.num_hops)]
        chiral_indices_list = []
        cis_indices_list = []
        trans_indices_list = []

        total_charges = []
        smiles_list = []

        for mol_idx, mol in enumerate(molecules):
            start = ptr[mol_idx].item()
            end = ptr[mol_idx + 1].item()
            offset = start

            atom_types[start:end] = torch.from_numpy(mol["atom_types"])
            degrees[start:end] = torch.from_numpy(mol["degrees"])
            hybridizations[start:end] = torch.from_numpy(mol["hybridizations"])
            hydrogen_counts[start:end] = torch.from_numpy(mol["hydrogen_counts"])
            atomic_numbers[start:end] = torch.from_numpy(mol["atomic_numbers"])
            batch_idx[start:end] = mol_idx

            for hop_idx, edges in enumerate(mol["multi_hop_edges"]):
                if edges.size > 0:
                    edges_tensor = torch.from_numpy(edges).long() + offset
                    hop_edges_lists[hop_idx].append(edges_tensor)

            for chiral in mol["chiral_centers"]:
                chiral_indices_list.append([c + offset for c in chiral])
            for cis in mol["cis_bonds"]:
                cis_indices_list.append([c + offset for c in cis])
            for trans in mol["trans_bonds"]:
                trans_indices_list.append([c + offset for c in trans])

            total_charges.append(mol["total_charge"])
            smiles_list.append(mol["smiles"])

        edge_indices = []
        for hop_edges in hop_edges_lists:
            if hop_edges:
                edge_indices.append(torch.cat(hop_edges, dim=1))
            else:
                edge_indices.append(torch.zeros((2, 0), dtype=torch.int64))

        chiral_indices = (
            torch.tensor(chiral_indices_list, dtype=torch.int64)
            if chiral_indices_list
            else torch.zeros((0, 4), dtype=torch.int64)
        )
        cis_indices = (
            torch.tensor(cis_indices_list, dtype=torch.int64)
            if cis_indices_list
            else torch.zeros((0, 4), dtype=torch.int64)
        )
        trans_indices = (
            torch.tensor(trans_indices_list, dtype=torch.int64)
            if trans_indices_list
            else torch.zeros((0, 4), dtype=torch.int64)
        )

        return MolecularGraphBatch(
            atom_types=atom_types,
            degrees=degrees,
            hybridizations=hybridizations,
            hydrogen_counts=hydrogen_counts,
            atomic_numbers=atomic_numbers,
            batch_idx=batch_idx,
            ptr=ptr,
            edge_indices=edge_indices,
            targets=torch.from_numpy(targets).float(),
            total_charges=torch.tensor(total_charges, dtype=torch.float32),
            smiles=smiles_list,
            chiral_indices=chiral_indices,
            cis_bond_indices=cis_indices,
            trans_bond_indices=trans_indices,
            num_molecules=num_molecules,
        )
