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


class TestMolecularGraphBatchFeatures:
    """Test batch with full molecular features."""

    def test_full_features_creation(self):
        """Test batch with all atom features."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8], dtype=torch.int32),
            degrees=torch.tensor([2, 3, 1], dtype=torch.int32),
            hybridizations=torch.tensor([2, 2, 3], dtype=torch.int32),
            hydrogen_counts=torch.tensor([2, 1, 0], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            num_molecules=1,
        )
        assert batch.degrees is not None
        assert batch.hybridizations is not None
        assert batch.hydrogen_counts is not None

    def test_edge_indices_multi_hop(self):
        """Test multi-hop edge indices."""
        hop1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64)
        hop2 = torch.tensor([[0, 2], [2, 0]], dtype=torch.int64)

        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 6], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            edge_indices=[hop1, hop2],
            num_molecules=1,
        )
        assert len(batch.edge_indices) == 2
        assert batch.edge_indices[0].shape == (2, 4)
        assert batch.edge_indices[1].shape == (2, 2)

    def test_stereochemistry_indices(self):
        """Test chiral and cis/trans indices."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 6, 6, 6], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 5], dtype=torch.int64),
            chiral_indices=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
            cis_bond_indices=torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            trans_bond_indices=torch.tensor([], dtype=torch.int64).reshape(0, 4),
            num_molecules=1,
        )
        assert batch.chiral_indices.shape == (1, 4)
        assert batch.cis_bond_indices.shape == (1, 4)
        assert batch.trans_bond_indices.shape == (0, 4)

    def test_atom_features_dict(self):
        """Test getting atom features as dict for model input."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8], dtype=torch.int32),
            degrees=torch.tensor([2, 3, 1], dtype=torch.int32),
            hybridizations=torch.tensor([2, 2, 3], dtype=torch.int32),
            hydrogen_counts=torch.tensor([2, 1, 0], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            num_molecules=1,
        )
        features = batch.atom_features_dict()
        assert "atom_type" in features
        assert "degree" in features
        assert "hybridization" in features
        assert "hydrogen_count" in features


class TestMolecularGraphBatchDeviceOptimization:
    """Test device transfer optimizations."""

    def test_to_same_device_returns_self(self):
        """Test that .to() returns same object if already on target device."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            num_molecules=1,
        )

        # Should return same object (identity check)
        batch2 = batch.to("cpu")
        assert batch2 is batch

    def test_to_uses_non_blocking_for_pinned(self):
        """Test that .to() uses non_blocking for pinned tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            num_molecules=1,
        )
        pinned = batch.pin_memory()

        # Should not raise
        gpu_batch = pinned.to("cuda")
        assert gpu_batch.device.type == "cuda"
