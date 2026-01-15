# tests/integration/test_data_pipeline.py
"""
Integration tests for the data preprocessing and loading pipeline.
"""

import pytest
import numpy as np
import tempfile
import os

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestSAENormalizer:
    """Integration tests for SAE normalizer."""

    def test_sae_normalizer_fit(self):
        """Test SAE normalizer can fit on training data."""
        from data.preprocessing import SAENormalizer

        # Sample SMILES and targets
        smiles = ["C", "CC", "CCC", "CCCC", "CCCCC"]
        targets = [-10.0, -20.0, -30.0, -40.0, -50.0]

        normalizer = SAENormalizer(task_type="regression")
        stats = normalizer.fit(smiles, targets)

        assert stats is not None
        assert normalizer.is_fitted

    def test_sae_normalizer_transform(self):
        """Test SAE normalizer can transform targets."""
        from data.preprocessing import SAENormalizer

        smiles = ["C", "CC", "CCC", "CCCC", "CCCCC"]
        targets = [-10.0, -20.0, -30.0, -40.0, -50.0]

        normalizer = SAENormalizer(task_type="regression")
        normalizer.fit(smiles, targets)

        transformed = normalizer.transform(smiles, targets)

        assert transformed is not None
        assert len(transformed) == len(targets)


class TestStandardScaler:
    """Tests for StandardScaler."""

    def test_standard_scaler_fit(self):
        """Test standard scaler can fit."""
        from data.preprocessing import StandardScaler

        targets = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        scaler = StandardScaler()
        scaler.fit(targets)

        assert scaler.is_fitted
        assert scaler.means is not None
        assert scaler.stds is not None

    def test_standard_scaler_transform(self):
        """Test standard scaler can transform."""
        from data.preprocessing import StandardScaler

        targets = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        scaler = StandardScaler()
        scaler.fit(targets)

        transformed = scaler.transform(targets)

        assert transformed is not None
        assert transformed.shape == targets.shape


class TestFeatureComputation:
    """Integration tests for molecular feature computation."""

    def test_compute_features_for_single_molecule(self):
        """Test computing features for a single molecule."""
        from datasets.features import compute_all

        result = compute_all("CCO", max_hops=3)

        assert result is not None
        assert 'multi_hop_edges' in result
        assert 'atom_features' in result
        assert 'atomic_numbers' in result

    def test_compute_features_for_aromatic(self):
        """Test computing features for aromatic molecule."""
        from datasets.features import compute_all

        result = compute_all("c1ccccc1", max_hops=3)

        assert result is not None
        # Benzene has 6 carbons + 6 hydrogens (explicit H added)
        assert len(result['atomic_numbers']) == 12

    def test_compute_features_for_chiral(self):
        """Test computing features for chiral molecule."""
        from datasets.features import compute_all

        result = compute_all("C[C@H](O)F", max_hops=3)

        assert result is not None
        assert 'chiral_tensors' in result


class TestPrecomputeAndFilter:
    """Tests for batch precomputation."""

    def test_precompute_filters_invalid(self):
        """Test that precompute filters invalid SMILES."""
        from datasets.features import precompute_all_and_filter

        smiles_list = ["C", "CC", "invalid_smiles", "CCC"]
        targets = [1.0, 2.0, 3.0, 4.0]

        # Use num_workers=1 to avoid Pool(0) error
        valid_smiles, valid_targets, precomputed = precompute_all_and_filter(
            smiles_list, targets, max_hops=3, num_workers=1
        )

        # Should filter out invalid SMILES
        assert len(valid_smiles) == 3
        assert len(valid_targets) == 3
        assert len(precomputed) == 3
        assert "invalid_smiles" not in valid_smiles

    def test_precompute_preserves_order(self):
        """Test that precompute preserves order of valid molecules."""
        from datasets.features import precompute_all_and_filter

        smiles_list = ["C", "CC", "CCC", "CCCC"]
        targets = [1.0, 2.0, 3.0, 4.0]

        valid_smiles, valid_targets, precomputed = precompute_all_and_filter(
            smiles_list, targets, max_hops=3, num_workers=1
        )

        assert valid_smiles == smiles_list
        assert valid_targets == targets


class TestDatasetCreation:
    """Integration tests for dataset creation."""

    def test_create_pyg_dataset(self):
        """Test creating PyG dataset from SMILES."""
        from datasets import PyGSMILESDataset
        from datasets.features import precompute_all_and_filter

        smiles_list = ["C", "CC", "CCC"]
        targets = [1.0, 2.0, 3.0]

        valid_smiles, valid_targets, precomputed = precompute_all_and_filter(
            smiles_list, targets, max_hops=3, num_workers=1
        )

        dataset = PyGSMILESDataset(
            smiles_list=valid_smiles,
            targets=valid_targets,
            precomputed_data=precomputed
        )

        assert len(dataset) == 3

    def test_dataset_item_structure(self):
        """Test that dataset items have expected structure."""
        from datasets import PyGSMILESDataset
        from datasets.features import precompute_all_and_filter

        smiles_list = ["CCO"]
        targets = [1.0]

        valid_smiles, valid_targets, precomputed = precompute_all_and_filter(
            smiles_list, targets, max_hops=3, num_workers=1
        )

        dataset = PyGSMILESDataset(
            smiles_list=valid_smiles,
            targets=valid_targets,
            precomputed_data=precomputed
        )

        data = dataset[0]
        assert hasattr(data, 'y')
        assert hasattr(data, 'smiles')
        # SMILES are converted to explicit hydrogen form during preprocessing
        assert 'C' in data.smiles  # Contains carbon
        assert 'O' in data.smiles  # Contains oxygen


class TestMolecularBatch:
    """Tests for MolecularBatch creation."""

    def test_batch_from_data_list(self):
        """Test creating batch from data list."""
        from datasets import PyGSMILESDataset, MolecularBatch
        from datasets.features import precompute_all_and_filter

        smiles_list = ["C", "CC", "CCC"]
        targets = [1.0, 2.0, 3.0]

        valid_smiles, valid_targets, precomputed = precompute_all_and_filter(
            smiles_list, targets, max_hops=3, num_workers=1
        )

        dataset = PyGSMILESDataset(
            smiles_list=valid_smiles,
            targets=valid_targets,
            precomputed_data=precomputed
        )

        data_list = [dataset[i] for i in range(len(dataset))]
        batch = MolecularBatch.from_data_list(data_list)

        assert batch is not None
        assert hasattr(batch, 'batch_indices')
        assert hasattr(batch, 'smiles_list')
        assert len(batch.smiles_list) == 3

    def test_batch_preserves_smiles(self):
        """Test that batch preserves SMILES order."""
        from datasets import PyGSMILESDataset, MolecularBatch
        from datasets.features import precompute_all_and_filter

        smiles_list = ["C", "CC", "CCC"]
        targets = [1.0, 2.0, 3.0]

        valid_smiles, valid_targets, precomputed = precompute_all_and_filter(
            smiles_list, targets, max_hops=3, num_workers=1
        )

        dataset = PyGSMILESDataset(
            smiles_list=valid_smiles,
            targets=valid_targets,
            precomputed_data=precomputed
        )

        data_list = [dataset[i] for i in range(len(dataset))]
        batch = MolecularBatch.from_data_list(data_list)

        # SMILES are converted to explicit hydrogen form, but order preserved
        assert len(batch.smiles_list) == 3
        # Verify methane < ethane < propane by length (more hydrogens = longer)
        assert len(batch.smiles_list[0]) < len(batch.smiles_list[1]) < len(batch.smiles_list[2])


class TestHDF5Operations:
    """Integration tests for HDF5 operations."""

    def test_write_read_hdf5_metadata(self):
        """Test writing and reading HDF5 metadata."""
        import h5py

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            hdf5_path = f.name

        try:
            # Write
            with h5py.File(hdf5_path, 'w') as f:
                metadata = f.create_group('metadata')
                metadata.attrs['num_samples'] = 100
                metadata.attrs['max_hops'] = 3
                metadata.attrs['preprocessing_applied'] = True

            # Read and verify
            with h5py.File(hdf5_path, 'r') as f:
                assert f['metadata'].attrs['num_samples'] == 100
                assert f['metadata'].attrs['max_hops'] == 3
                assert f['metadata'].attrs['preprocessing_applied'] == True

        finally:
            os.unlink(hdf5_path)

    def test_write_read_smiles(self):
        """Test writing and reading SMILES to HDF5."""
        import h5py

        smiles_list = ["C", "CC", "CCC", "CCCC"]

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            hdf5_path = f.name

        try:
            # Write
            with h5py.File(hdf5_path, 'w') as f:
                smiles_encoded = [s.encode('utf-8') for s in smiles_list]
                f.create_dataset('smiles', data=smiles_encoded)

            # Read and verify
            with h5py.File(hdf5_path, 'r') as f:
                smiles_read = [s.decode('utf-8') for s in f['smiles'][:]]
                assert smiles_read == smiles_list

        finally:
            os.unlink(hdf5_path)
