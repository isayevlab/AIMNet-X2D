"""Tests for MolecularGraphBatch dataclass."""

import torch
import pytest
from src.core.batch import MolecularGraphBatch


class TestMolecularGraphBatch:
    """Test MolecularGraphBatch creation and operations."""

    def test_creation_minimal(self):
        """Test batch creation with minimal required fields."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8, 6, 7], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64),
            ptr=torch.tensor([0, 3, 5], dtype=torch.int64),
            num_molecules=2,
        )
        assert batch.num_molecules == 2
        assert batch.total_atoms == 5
        assert batch.atom_types.shape == (5,)

    def test_to_device(self):
        """Test moving batch to device."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            num_molecules=1,
        )
        moved = batch.to(torch.device("cpu"))
        assert moved.atom_types.device.type == "cpu"

    def test_from_molecule_list(self):
        """Test creating batch from list of molecule data."""
        molecules = [
            {"atom_types": [6, 6, 8], "targets": [1.5]},
            {"atom_types": [6, 7], "targets": [2.0]},
        ]
        batch = MolecularGraphBatch.from_molecules(molecules)
        assert batch.num_molecules == 2
        assert batch.total_atoms == 5
        assert batch.targets.shape == (2, 1)

    def test_batch_slicing(self):
        """Test extracting single molecule from batch."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8, 6, 7], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64),
            ptr=torch.tensor([0, 3, 5], dtype=torch.int64),
            num_molecules=2,
            targets=torch.tensor([[1.0], [2.0]]),
        )
        mol0 = batch.get_molecule(0)
        assert mol0["atom_types"].tolist() == [6, 6, 8]
        assert mol0["target"].item() == 1.0
