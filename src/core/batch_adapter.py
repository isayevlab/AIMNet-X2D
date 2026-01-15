"""
Batch adapter for bridging SAE and MolecularGraphBatch formats.

SAE transform expects padded tensors [batch_size, max_atoms] with atom_counts.
MolecularGraphBatch uses concatenated tensors [total_atoms] with batch_idx.

This adapter provides conversion between these formats.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .batch import MolecularGraphBatch


class BatchAdapter:
    """
    Adapter for converting between batch formats.

    Provides methods to:
    - Convert concatenated batch format to padded format for SAE
    - Extract atom counts from ptr tensor
    """

    def to_padded_format(
        self,
        batch: MolecularGraphBatch,
    ) -> tuple[Tensor, Tensor]:
        """
        Convert concatenated batch to padded format for SAE transform.

        Args:
            batch: MolecularGraphBatch with concatenated atomic_numbers

        Returns:
            Tuple of:
            - padded_atomic_numbers: [num_molecules, max_atoms] tensor
            - atom_counts: [num_molecules] tensor
        """
        device = batch.device
        num_molecules = batch.num_molecules

        # Get atomic numbers (prefer atomic_numbers, fall back to atom_types)
        if batch.atomic_numbers is not None:
            atomic_nums = batch.atomic_numbers
        else:
            atomic_nums = batch.atom_types.long()

        # Compute atom counts from ptr
        atom_counts = batch.ptr[1:] - batch.ptr[:-1]
        max_atoms = atom_counts.max().item()

        # Create padded tensor
        padded = torch.zeros(
            num_molecules,
            max_atoms,
            dtype=torch.int64,
            device=device,
        )

        # Fill in atomic numbers for each molecule
        for mol_idx in range(num_molecules):
            start = batch.ptr[mol_idx].item()
            end = batch.ptr[mol_idx + 1].item()
            count = end - start
            padded[mol_idx, :count] = atomic_nums[start:end]

        return padded, atom_counts

    def apply_sae_to_batch(
        self,
        batch: MolecularGraphBatch,
        sae_transform,
    ) -> Tensor:
        """
        Apply SAE transform to batch targets.

        Convenience method that handles format conversion automatically.

        Args:
            batch: MolecularGraphBatch with targets
            sae_transform: SAETransform instance

        Returns:
            Transformed targets [num_molecules, num_targets]
        """
        if batch.targets is None:
            raise ValueError("Batch has no targets to transform")

        padded_nums, atom_counts = self.to_padded_format(batch)

        return sae_transform.transform_batch(
            padded_nums,
            atom_counts,
            batch.targets,
        )

    def inverse_sae_from_batch(
        self,
        batch: MolecularGraphBatch,
        normalized: Tensor,
        sae_transform,
    ) -> Tensor:
        """
        Apply inverse SAE transform using batch atomic numbers.

        Args:
            batch: MolecularGraphBatch with atomic structure
            normalized: Normalized predictions [num_molecules, num_targets]
            sae_transform: SAETransform instance

        Returns:
            Denormalized predictions [num_molecules, num_targets]
        """
        padded_nums, atom_counts = self.to_padded_format(batch)

        return sae_transform.inverse_transform_batch(
            padded_nums,
            atom_counts,
            normalized,
        )
