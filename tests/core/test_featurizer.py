"""Tests for batch-native featurizer."""

import torch
import numpy as np
import pytest
from src.core.featurizer import BatchFeaturizer
from src.core.batch import MolecularGraphBatch


class TestBatchFeaturizer:
    """Test batch-native molecular featurization."""

    def test_single_molecule(self):
        """Test featurizing single molecule."""
        featurizer = BatchFeaturizer(num_hops=2)

        smiles = ["CCO"]  # Ethanol
        targets = np.array([[1.0]])

        batch = featurizer.featurize(smiles, targets)

        assert isinstance(batch, MolecularGraphBatch)
        assert batch.num_molecules == 1
        assert batch.total_atoms == 9  # With hydrogens

    def test_batch_molecules(self):
        """Test featurizing batch of molecules."""
        featurizer = BatchFeaturizer(num_hops=3)

        smiles = ["C", "CC", "CCC"]
        targets = np.array([[1.0], [2.0], [3.0]])

        batch = featurizer.featurize(smiles, targets)

        assert batch.num_molecules == 3
        assert batch.targets.shape == (3, 1)
        assert len(batch.smiles) == 3

    def test_edge_indices_generated(self):
        """Test multi-hop edge indices are generated."""
        featurizer = BatchFeaturizer(num_hops=2)

        smiles = ["CC"]
        targets = np.array([[1.0]])

        batch = featurizer.featurize(smiles, targets)

        assert len(batch.edge_indices) == 2
        assert all(e.shape[0] == 2 for e in batch.edge_indices)

    def test_invalid_smiles_filtered(self):
        """Test invalid SMILES are filtered."""
        featurizer = BatchFeaturizer(num_hops=2)

        smiles = ["C", "INVALID_SMILES", "CC"]
        targets = np.array([[1.0], [2.0], [3.0]])

        batch = featurizer.featurize(smiles, targets)

        assert batch.num_molecules == 2
        assert batch.smiles == ["C", "CC"]

    def test_atom_features_populated(self):
        """Test all atom features are populated."""
        featurizer = BatchFeaturizer(num_hops=2)

        smiles = ["CCO"]
        targets = np.array([[1.0]])

        batch = featurizer.featurize(smiles, targets)

        assert batch.degrees is not None
        assert batch.hybridizations is not None
        assert batch.hydrogen_counts is not None
        assert batch.atomic_numbers is not None

    def test_parallel_featurization(self):
        """Test parallel processing works."""
        featurizer = BatchFeaturizer(num_hops=2, num_workers=2)

        smiles = ["C" * i for i in range(1, 11)]
        targets = np.arange(10).reshape(-1, 1).astype(np.float32)

        batch = featurizer.featurize(smiles, targets)

        assert batch.num_molecules == 10

    def test_all_invalid_raises_error(self):
        """Test that all invalid SMILES raises ValueError."""
        featurizer = BatchFeaturizer(num_hops=2)

        smiles = ["INVALID1", "INVALID2", "INVALID3"]
        targets = np.array([[1.0], [2.0], [3.0]])

        with pytest.raises(ValueError, match="No valid molecules"):
            featurizer.featurize(smiles, targets)
