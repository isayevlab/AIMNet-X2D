"""Tests for GPU-native preprocessing."""

import torch
import numpy as np
import pytest
from src.core.preprocessing import SAETransform


class TestSAETransform:
    """Test GPU-native SAE normalization."""

    def test_fit_from_atomic_numbers(self):
        """Test fitting SAE from atomic numbers and targets."""
        atomic_numbers = [
            [6, 6, 8],      # C-C-O
            [6, 6, 6, 7],   # C-C-C-N
        ]
        targets = np.array([[10.0], [15.0]])

        transform = SAETransform.fit(atomic_numbers, targets, subtasks=[0])

        assert 6 in transform.sae_dict
        assert 8 in transform.sae_dict
        assert 7 in transform.sae_dict

    def test_transform_gpu_native(self):
        """Test GPU-native SAE transformation."""
        transform = SAETransform(
            sae_dict={6: 1.0, 8: 2.0, 7: 1.5},
            subtasks=[0],
        )

        atomic_numbers = torch.tensor([
            [6, 6, 8, 0],
            [6, 6, 6, 7],
        ], dtype=torch.int64)
        atom_counts = torch.tensor([3, 4], dtype=torch.int64)
        targets = torch.tensor([[10.0], [15.0]])

        transformed = transform.transform_batch(
            atomic_numbers, atom_counts, targets
        )

        # Mol 0: 10.0 - (1.0 + 1.0 + 2.0) = 6.0
        # Mol 1: 15.0 - (1.0 + 1.0 + 1.0 + 1.5) = 10.5
        assert torch.allclose(transformed[:, 0], torch.tensor([6.0, 10.5]))

    def test_inverse_transform_gpu_native(self):
        """Test GPU-native inverse SAE transformation."""
        transform = SAETransform(
            sae_dict={6: 1.0, 8: 2.0, 7: 1.5},
            subtasks=[0],
        )

        atomic_numbers = torch.tensor([
            [6, 6, 8, 0],
            [6, 6, 6, 7],
        ], dtype=torch.int64)
        atom_counts = torch.tensor([3, 4], dtype=torch.int64)
        normalized = torch.tensor([[6.0], [10.5]])

        original = transform.inverse_transform_batch(
            atomic_numbers, atom_counts, normalized
        )

        assert torch.allclose(original[:, 0], torch.tensor([10.0, 15.0]))

    def test_multitask_sae(self):
        """Test SAE on specific subtasks only."""
        transform = SAETransform(
            sae_dict={6: 1.0, 8: 2.0},
            subtasks=[1, 2],
        )

        atomic_numbers = torch.tensor([[6, 6, 8]], dtype=torch.int64)
        atom_counts = torch.tensor([3], dtype=torch.int64)
        targets = torch.tensor([[5.0, 10.0, 20.0]])

        transformed = transform.transform_batch(
            atomic_numbers, atom_counts, targets
        )

        sae_shift = 1.0 + 1.0 + 2.0  # = 4.0
        expected = torch.tensor([[5.0, 10.0 - sae_shift, 20.0 - sae_shift]])
        assert torch.allclose(transformed, expected)
