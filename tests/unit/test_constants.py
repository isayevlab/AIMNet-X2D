# tests/unit/test_constants.py
"""
Unit tests for datasets constants module.
"""

import pytest


class TestDataProcessingConstants:
    """Tests for data processing constants."""

    def test_shuffle_buffer_size_is_positive(self):
        """Test that shuffle buffer size is a positive integer."""
        from datasets.constants import DEFAULT_SHUFFLE_BUFFER_SIZE
        assert DEFAULT_SHUFFLE_BUFFER_SIZE > 0
        assert isinstance(DEFAULT_SHUFFLE_BUFFER_SIZE, int)

    def test_chunk_size_is_positive(self):
        """Test that chunk size is a positive integer."""
        from datasets.constants import DEFAULT_CHUNK_SIZE
        assert DEFAULT_CHUNK_SIZE > 0
        assert isinstance(DEFAULT_CHUNK_SIZE, int)

    def test_molecule_estimate_is_positive(self):
        """Test that molecule estimate is a positive integer."""
        from datasets.constants import DEFAULT_MOLECULE_ESTIMATE
        assert DEFAULT_MOLECULE_ESTIMATE > 0
        assert isinstance(DEFAULT_MOLECULE_ESTIMATE, int)


class TestTrainingConstants:
    """Tests for training constants."""

    def test_splits_sum_to_one(self):
        """Test that default splits sum to approximately 1.0."""
        from datasets.constants import (
            DEFAULT_VAL_SPLIT,
            DEFAULT_TEST_SPLIT,
            DEFAULT_TRAIN_SPLIT,
        )
        total = DEFAULT_TRAIN_SPLIT + DEFAULT_VAL_SPLIT + DEFAULT_TEST_SPLIT
        assert abs(total - 1.0) < 0.001

    def test_splits_are_valid_fractions(self):
        """Test that splits are valid fractions between 0 and 1."""
        from datasets.constants import (
            DEFAULT_VAL_SPLIT,
            DEFAULT_TEST_SPLIT,
            DEFAULT_TRAIN_SPLIT,
        )
        for split in [DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT, DEFAULT_TRAIN_SPLIT]:
            assert 0.0 < split < 1.0

    def test_lr_step_gamma_is_valid(self):
        """Test that learning rate gamma is a valid decay factor."""
        from datasets.constants import DEFAULT_LR_STEP_GAMMA
        assert 0.0 < DEFAULT_LR_STEP_GAMMA < 1.0

    def test_random_seed_is_integer(self):
        """Test that random seed is an integer."""
        from datasets.constants import DEFAULT_RANDOM_SEED
        assert isinstance(DEFAULT_RANDOM_SEED, int)


class TestInferenceConstants:
    """Tests for inference constants."""

    def test_mc_dropout_rate_is_valid(self):
        """Test that MC dropout rate is a valid probability."""
        from datasets.constants import DEFAULT_MC_DROPOUT_RATE
        assert 0.0 <= DEFAULT_MC_DROPOUT_RATE <= 1.0

    def test_flush_threshold_is_positive(self):
        """Test that batch flush threshold is positive."""
        from datasets.constants import DEFAULT_BATCH_FLUSH_THRESHOLD
        assert DEFAULT_BATCH_FLUSH_THRESHOLD > 0

    def test_progress_log_interval_is_positive(self):
        """Test that progress log interval is positive."""
        from datasets.constants import DEFAULT_PROGRESS_LOG_INTERVAL
        assert DEFAULT_PROGRESS_LOG_INTERVAL > 0


class TestDistributedConstants:
    """Tests for distributed training constants."""

    def test_ddp_sync_delay_is_positive(self):
        """Test that DDP sync delay is positive."""
        from datasets.constants import DDP_SYNC_DELAY
        assert DDP_SYNC_DELAY > 0.0

    def test_ddp_sync_delay_is_reasonable(self):
        """Test that DDP sync delay is not too long."""
        from datasets.constants import DDP_SYNC_DELAY
        assert DDP_SYNC_DELAY < 10.0  # Should be less than 10 seconds


class TestAtomFeatureConstants:
    """Tests for atom feature constants."""

    def test_atom_types_covers_periodic_table(self):
        """Test that atom types cover the periodic table."""
        from datasets.constants import ATOM_TYPES
        assert len(ATOM_TYPES) == 118  # Elements 1-118
        assert 1 in ATOM_TYPES  # Hydrogen
        assert 6 in ATOM_TYPES  # Carbon
        assert 118 in ATOM_TYPES  # Oganesson

    def test_degrees_are_valid(self):
        """Test that degrees are valid (0-5)."""
        from datasets.constants import DEGREES
        assert len(DEGREES) == 6
        assert min(DEGREES) == 0
        assert max(DEGREES) == 5

    def test_hybridizations_are_valid(self):
        """Test that hybridizations are RDKit HybridizationType."""
        from datasets.constants import HYBRIDIZATIONS
        from rdkit.Chem.rdchem import HybridizationType
        assert len(HYBRIDIZATIONS) == 6
        for hyb in HYBRIDIZATIONS:
            assert isinstance(hyb, HybridizationType)
