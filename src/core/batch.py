"""MolecularGraphBatch dataclass for GPU-native molecular batching."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class MolecularGraphBatch:
    """
    A batch of molecular graphs optimized for GPU processing.

    This dataclass holds batched molecular data in a format suitable for
    efficient GPU computation. All per-atom tensors are concatenated,
    with batch_idx and ptr tensors tracking molecule boundaries.

    Attributes:
        atom_types: Atomic numbers for all atoms in batch [total_atoms]
        batch_idx: Molecule index for each atom [total_atoms]
        ptr: Cumulative atom counts [num_molecules + 1]
        num_molecules: Number of molecules in batch

    Optional Attributes:
        degrees: Atom degrees [total_atoms]
        hybridizations: Hybridization states [total_atoms]
        hydrogen_counts: Number of hydrogens per atom [total_atoms]
        edge_indices: List of edge index tensors for multi-hop edges
        targets: Target values [num_molecules, num_targets]
        total_charges: Molecular charges [num_molecules]
        smiles: SMILES strings for each molecule
        chiral_indices: Indices of chiral atoms
        cis_bond_indices: Indices of cis double bonds
        trans_bond_indices: Indices of trans double bonds
        atomic_numbers: Alternative atomic number representation
    """

    # Required attributes
    atom_types: torch.Tensor
    batch_idx: torch.Tensor
    ptr: torch.Tensor
    num_molecules: int

    # Optional attributes
    degrees: torch.Tensor | None = None
    hybridizations: torch.Tensor | None = None
    hydrogen_counts: torch.Tensor | None = None
    edge_indices: list[torch.Tensor] = field(default_factory=list)
    targets: torch.Tensor | None = None
    total_charges: torch.Tensor | None = None
    smiles: list[str] = field(default_factory=list)
    chiral_indices: torch.Tensor | None = None
    cis_bond_indices: torch.Tensor | None = None
    trans_bond_indices: torch.Tensor | None = None
    atomic_numbers: torch.Tensor | None = None

    @property
    def total_atoms(self) -> int:
        """Return total number of atoms in the batch."""
        return self.atom_types.shape[0]

    @property
    def device(self) -> torch.device:
        """Return the device of the batch tensors."""
        return self.atom_types.device

    def to(self, device: torch.device | str) -> MolecularGraphBatch:
        """
        Move all tensors to the specified device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device)

        Returns:
            New MolecularGraphBatch with tensors on the target device
        """
        if isinstance(device, str):
            device = torch.device(device)

        def move_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(device) if t is not None else None

        def move_tensor_list(
            tensors: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            return [t.to(device) for t in tensors]

        return MolecularGraphBatch(
            atom_types=self.atom_types.to(device),
            batch_idx=self.batch_idx.to(device),
            ptr=self.ptr.to(device),
            num_molecules=self.num_molecules,
            degrees=move_tensor(self.degrees),
            hybridizations=move_tensor(self.hybridizations),
            hydrogen_counts=move_tensor(self.hydrogen_counts),
            edge_indices=move_tensor_list(self.edge_indices),
            targets=move_tensor(self.targets),
            total_charges=move_tensor(self.total_charges),
            smiles=self.smiles.copy(),
            chiral_indices=move_tensor(self.chiral_indices),
            cis_bond_indices=move_tensor(self.cis_bond_indices),
            trans_bond_indices=move_tensor(self.trans_bond_indices),
            atomic_numbers=move_tensor(self.atomic_numbers),
        )

    @classmethod
    def from_molecules(
        cls,
        molecules: list[dict],
    ) -> MolecularGraphBatch:
        """
        Create a batch from a list of molecule dictionaries.

        Args:
            molecules: List of dicts with keys like 'atom_types', 'targets', etc.

        Returns:
            MolecularGraphBatch containing all molecules
        """
        all_atom_types: list[int] = []
        all_batch_idx: list[int] = []
        ptr: list[int] = [0]
        all_targets: list[list[float]] = []

        for mol_idx, mol in enumerate(molecules):
            atom_types = mol["atom_types"]
            num_atoms = len(atom_types)

            all_atom_types.extend(atom_types)
            all_batch_idx.extend([mol_idx] * num_atoms)
            ptr.append(ptr[-1] + num_atoms)

            if "targets" in mol:
                all_targets.append(mol["targets"])

        targets_tensor: torch.Tensor | None = None
        if all_targets:
            targets_tensor = torch.tensor(all_targets, dtype=torch.float32)

        return cls(
            atom_types=torch.tensor(all_atom_types, dtype=torch.int32),
            batch_idx=torch.tensor(all_batch_idx, dtype=torch.int64),
            ptr=torch.tensor(ptr, dtype=torch.int64),
            num_molecules=len(molecules),
            targets=targets_tensor,
        )

    def get_molecule(self, idx: int) -> dict:
        """
        Extract data for a single molecule from the batch.

        Args:
            idx: Index of the molecule (0-indexed)

        Returns:
            Dictionary with molecule data (atom_types, target, etc.)
        """
        if idx < 0 or idx >= self.num_molecules:
            raise IndexError(
                f"Molecule index {idx} out of range [0, {self.num_molecules})"
            )

        start = self.ptr[idx].item()
        end = self.ptr[idx + 1].item()

        result: dict = {
            "atom_types": self.atom_types[start:end],
        }

        if self.targets is not None:
            result["target"] = self.targets[idx]

        if self.degrees is not None:
            result["degrees"] = self.degrees[start:end]

        if self.hybridizations is not None:
            result["hybridizations"] = self.hybridizations[start:end]

        if self.hydrogen_counts is not None:
            result["hydrogen_counts"] = self.hydrogen_counts[start:end]

        if self.smiles and idx < len(self.smiles):
            result["smiles"] = self.smiles[idx]

        if self.total_charges is not None:
            result["total_charge"] = self.total_charges[idx]

        return result

    def atom_features_dict(self) -> dict[str, torch.Tensor]:
        """
        Get atom features as dictionary for model embedding layers.

        Returns dict with keys matching embedding layer names:
        - 'atom_type': [total_atoms] long tensor
        - 'degree': [total_atoms] long tensor (if degrees populated)
        - 'hybridization': [total_atoms] long tensor (if hybridizations populated)
        - 'hydrogen_count': [total_atoms] long tensor (if hydrogen_counts populated)
        """
        features = {"atom_type": self.atom_types.long()}

        if self.degrees is not None:
            features["degree"] = self.degrees.long()
        if self.hybridizations is not None:
            features["hybridization"] = self.hybridizations.long()
        if self.hydrogen_counts is not None:
            features["hydrogen_count"] = self.hydrogen_counts.long()

        return features

    def pin_memory(self) -> MolecularGraphBatch:
        """
        Pin tensors in memory for faster GPU transfer.

        Returns:
            New MolecularGraphBatch with pinned tensors
        """

        def pin_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.pin_memory() if t is not None else None

        def pin_tensor_list(
            tensors: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            return [t.pin_memory() for t in tensors]

        return MolecularGraphBatch(
            atom_types=self.atom_types.pin_memory(),
            batch_idx=self.batch_idx.pin_memory(),
            ptr=self.ptr.pin_memory(),
            num_molecules=self.num_molecules,
            degrees=pin_tensor(self.degrees),
            hybridizations=pin_tensor(self.hybridizations),
            hydrogen_counts=pin_tensor(self.hydrogen_counts),
            edge_indices=pin_tensor_list(self.edge_indices),
            targets=pin_tensor(self.targets),
            total_charges=pin_tensor(self.total_charges),
            smiles=self.smiles.copy(),
            chiral_indices=pin_tensor(self.chiral_indices),
            cis_bond_indices=pin_tensor(self.cis_bond_indices),
            trans_bond_indices=pin_tensor(self.trans_bond_indices),
            atomic_numbers=pin_tensor(self.atomic_numbers),
        )
