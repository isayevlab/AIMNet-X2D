"""Tests for batch adapter that bridges SAE and MolecularGraphBatch formats."""

import pytest
import torch

from src.core.batch import MolecularGraphBatch
from src.core.batch_adapter import BatchAdapter


class TestBatchAdapter:
    """Tests for BatchAdapter class."""

    def test_to_padded_format(self):
        """Test conversion from concatenated to padded format."""
        # Create a batch with 3 molecules of different sizes
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 1, 1, 8, 1, 6, 6], dtype=torch.int32),
            atomic_numbers=torch.tensor([6, 1, 1, 8, 1, 6, 6], dtype=torch.int64),
            batch_idx=torch.tensor([0, 0, 0, 1, 1, 2, 2], dtype=torch.int64),
            ptr=torch.tensor([0, 3, 5, 7], dtype=torch.int64),
            num_molecules=3,
        )

        adapter = BatchAdapter()
        padded_nums, atom_counts = adapter.to_padded_format(batch)

        # Should produce [3, max_atoms] tensor
        assert padded_nums.shape[0] == 3
        assert padded_nums.shape[1] == 3  # max atoms in any molecule
        assert atom_counts.tolist() == [3, 2, 2]

        # Check values
        assert padded_nums[0, :3].tolist() == [6, 1, 1]
        assert padded_nums[1, :2].tolist() == [8, 1]
        assert padded_nums[2, :2].tolist() == [6, 6]

    def test_to_padded_format_single_molecule(self):
        """Test conversion with single molecule batch."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 8, 1], dtype=torch.int32),
            atomic_numbers=torch.tensor([6, 8, 1], dtype=torch.int64),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            num_molecules=1,
        )

        adapter = BatchAdapter()
        padded_nums, atom_counts = adapter.to_padded_format(batch)

        assert padded_nums.shape == (1, 3)
        assert atom_counts.tolist() == [3]
        assert padded_nums[0].tolist() == [6, 8, 1]

    def test_round_trip_consistency(self):
        """Test that padding preserves atom information."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 1, 1, 1, 1, 8, 1, 1], dtype=torch.int32),
            atomic_numbers=torch.tensor([6, 1, 1, 1, 1, 8, 1, 1], dtype=torch.int64),
            batch_idx=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int64),
            ptr=torch.tensor([0, 5, 8], dtype=torch.int64),
            num_molecules=2,
        )

        adapter = BatchAdapter()
        padded_nums, atom_counts = adapter.to_padded_format(batch)

        # Verify all original atoms are preserved
        for mol_idx in range(batch.num_molecules):
            start = batch.ptr[mol_idx].item()
            end = batch.ptr[mol_idx + 1].item()
            original = batch.atomic_numbers[start:end]
            padded = padded_nums[mol_idx, : atom_counts[mol_idx]]

            assert torch.equal(original, padded)

    def test_to_padded_format_without_atomic_numbers(self):
        """Test conversion falls back to atom_types when atomic_numbers is None."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 1, 8, 1], dtype=torch.int32),
            atomic_numbers=None,
            batch_idx=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
            ptr=torch.tensor([0, 2, 4], dtype=torch.int64),
            num_molecules=2,
        )

        adapter = BatchAdapter()
        padded_nums, atom_counts = adapter.to_padded_format(batch)

        assert padded_nums.shape == (2, 2)
        assert atom_counts.tolist() == [2, 2]
        assert padded_nums[0].tolist() == [6, 1]
        assert padded_nums[1].tolist() == [8, 1]

    def test_padding_values_are_zero(self):
        """Test that padded positions are filled with zeros."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 1, 1, 8], dtype=torch.int32),
            atomic_numbers=torch.tensor([6, 1, 1, 8], dtype=torch.int64),
            batch_idx=torch.tensor([0, 0, 0, 1], dtype=torch.int64),
            ptr=torch.tensor([0, 3, 4], dtype=torch.int64),
            num_molecules=2,
        )

        adapter = BatchAdapter()
        padded_nums, atom_counts = adapter.to_padded_format(batch)

        # Molecule 0 has 3 atoms, molecule 1 has 1 atom
        # Max atoms is 3, so molecule 1 should have 2 padding positions
        assert padded_nums.shape == (2, 3)
        assert padded_nums[1, 1].item() == 0  # Padding
        assert padded_nums[1, 2].item() == 0  # Padding
