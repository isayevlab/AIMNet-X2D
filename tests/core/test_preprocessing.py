"""Tests for GPU-native preprocessing."""

import torch
import numpy as np
import pytest
from src.core.preprocessing import SAETransform, StandardScaler, PreprocessingPipeline


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


class TestStandardScaler:
    """Test GPU-native standard scaling."""

    def test_fit_transform(self):
        """Test fitting and transforming."""
        targets = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

        scaler = StandardScaler.fit(targets)
        transformed = scaler.transform_batch(targets)

        assert torch.abs(transformed.mean()) < 1e-5
        assert torch.abs(transformed.std(unbiased=False) - 1.0) < 1e-5

    def test_inverse_transform(self):
        """Test inverse transformation."""
        targets = torch.tensor([[10.0], [20.0], [30.0]])

        scaler = StandardScaler.fit(targets)
        transformed = scaler.transform_batch(targets)
        recovered = scaler.inverse_transform_batch(transformed)

        assert torch.allclose(recovered, targets, atol=1e-5)


class TestPreprocessingPipeline:
    """Test combined preprocessing pipeline."""

    def test_sae_then_scale(self):
        """Test SAE followed by scaling."""
        atomic_numbers = [
            [6, 6, 8],
            [6, 6, 6, 7],
        ]
        targets = np.array([[100.0], [150.0]])

        pipeline = PreprocessingPipeline.fit(
            atomic_numbers_list=atomic_numbers,
            targets=targets,
            apply_sae=True,
            sae_subtasks=[0],
            apply_scaling=True,
        )

        atomic_nums_tensor = torch.tensor([
            [6, 6, 8, 0],
            [6, 6, 6, 7],
        ], dtype=torch.int64)
        atom_counts = torch.tensor([3, 4], dtype=torch.int64)
        targets_tensor = torch.tensor([[100.0], [150.0]])

        transformed = pipeline.transform_batch(
            atomic_nums_tensor, atom_counts, targets_tensor
        )

        assert transformed.shape == (2, 1)

    def test_inverse_full_pipeline(self):
        """Test full inverse transformation."""
        atomic_numbers = [[6, 6], [6, 6, 6]]
        targets = np.array([[50.0], [75.0]])

        pipeline = PreprocessingPipeline.fit(
            atomic_numbers_list=atomic_numbers,
            targets=targets,
            apply_sae=True,
            sae_subtasks=[0],
            apply_scaling=True,
        )

        atomic_nums_tensor = torch.tensor([[6, 6, 0], [6, 6, 6]], dtype=torch.int64)
        atom_counts = torch.tensor([2, 3], dtype=torch.int64)
        targets_tensor = torch.tensor([[50.0], [75.0]])

        transformed = pipeline.transform_batch(
            atomic_nums_tensor, atom_counts, targets_tensor
        )
        recovered = pipeline.inverse_transform_batch(
            atomic_nums_tensor, atom_counts, transformed
        )

        assert torch.allclose(recovered, targets_tensor, atol=1e-4)


def test_sae_transform_vectorized_multiple_subtasks():
    """Test SAE transform correctly applies to multiple subtasks."""
    from src.core.preprocessing import SAETransform

    # Create SAE transform with multiple subtasks
    sae_dict = {6: -38.0, 1: -0.5, 8: -75.0}  # C, H, O
    subtasks = [0, 2, 4]  # Apply to columns 0, 2, 4

    transform = SAETransform(sae_dict=sae_dict, subtasks=subtasks)

    # Batch: 2 molecules with padded atomic numbers
    # Mol 1: CH4 (5 atoms), Mol 2: H2O (3 atoms)
    atomic_numbers = torch.tensor([
        [6, 1, 1, 1, 1],  # CH4
        [8, 1, 1, 0, 0],  # H2O (padded)
    ], dtype=torch.int64)
    atom_counts = torch.tensor([5, 3], dtype=torch.int64)

    # 5 target columns
    targets = torch.tensor([
        [100.0, 50.0, 200.0, 75.0, 300.0],
        [150.0, 60.0, 250.0, 80.0, 350.0],
    ], dtype=torch.float32)

    result = transform.transform_batch(atomic_numbers, atom_counts, targets)

    # Expected SAE shifts:
    # CH4: -38.0 + 4*(-0.5) = -40.0
    # H2O: -75.0 + 2*(-0.5) = -76.0

    # Column 0: 100 - (-40) = 140, 150 - (-76) = 226
    # Column 2: 200 - (-40) = 240, 250 - (-76) = 326
    # Column 4: 300 - (-40) = 340, 350 - (-76) = 426
    # Columns 1, 3: unchanged

    assert torch.isclose(result[0, 0], torch.tensor(140.0))
    assert torch.isclose(result[0, 1], torch.tensor(50.0))  # unchanged
    assert torch.isclose(result[0, 2], torch.tensor(240.0))
    assert torch.isclose(result[1, 0], torch.tensor(226.0))
