# tests/unit/test_distributed.py
"""
Unit tests for distributed training utilities.
"""

import pytest
import numpy as np


class TestDistributedUtilities:
    """Tests for distributed utility functions."""

    def test_is_main_process_without_distributed(self):
        """Test is_main_process returns True when distributed is not initialized."""
        from utils.distributed import is_main_process
        # Without distributed initialized, should return True
        assert is_main_process() is True

    def test_safe_get_rank_without_distributed(self):
        """Test safe_get_rank returns 0 when distributed is not initialized."""
        from utils.distributed import safe_get_rank
        # Without distributed initialized, should return 0
        assert safe_get_rank() == 0

    def test_get_world_size_without_distributed(self):
        """Test get_world_size returns 1 when distributed is not initialized."""
        from utils.distributed import get_world_size
        # Without distributed initialized, should return 1
        assert get_world_size() == 1


class TestGatherFunctions:
    """Tests for gather functions without actual distributed environment."""

    def test_gather_ndarray_returns_same_without_distributed(self):
        """Test gather_ndarray_to_rank0 returns input when not distributed."""
        from utils.distributed import gather_ndarray_to_rank0

        arr = np.array([1.0, 2.0, 3.0])
        result = gather_ndarray_to_rank0(arr)

        np.testing.assert_array_equal(result, arr)

    def test_gather_ndarray_preserves_dtype(self):
        """Test gather_ndarray_to_rank0 preserves dtype."""
        from utils.distributed import gather_ndarray_to_rank0

        arr = np.array([1, 2, 3], dtype=np.int32)
        result = gather_ndarray_to_rank0(arr)

        # Without distributed, should return same array
        np.testing.assert_array_equal(result, arr)

    def test_gather_ndarray_handles_2d_array(self):
        """Test gather_ndarray_to_rank0 handles 2D arrays."""
        from utils.distributed import gather_ndarray_to_rank0

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = gather_ndarray_to_rank0(arr)

        np.testing.assert_array_equal(result, arr)

    def test_gather_strings_returns_same_without_distributed(self):
        """Test gather_strings_to_rank0 returns input when not distributed."""
        from utils.distributed import gather_strings_to_rank0

        strings = ["CC", "CCC", "CCCC"]
        result = gather_strings_to_rank0(strings)

        assert result == strings

    def test_gather_strings_handles_empty_list(self):
        """Test gather_strings_to_rank0 handles empty list."""
        from utils.distributed import gather_strings_to_rank0

        strings = []
        result = gather_strings_to_rank0(strings)

        assert result == []

    def test_gather_strings_handles_unicode(self):
        """Test gather_strings_to_rank0 handles unicode strings."""
        from utils.distributed import gather_strings_to_rank0

        strings = ["C=O", "C#N", "[Na+]"]
        result = gather_strings_to_rank0(strings)

        assert result == strings


class TestEdgeCases:
    """Tests for edge cases in distributed utilities."""

    def test_gather_empty_array(self):
        """Test gathering empty array."""
        from utils.distributed import gather_ndarray_to_rank0

        arr = np.array([])
        result = gather_ndarray_to_rank0(arr)

        assert len(result) == 0

    def test_gather_single_element_array(self):
        """Test gathering single element array."""
        from utils.distributed import gather_ndarray_to_rank0

        arr = np.array([42.0])
        result = gather_ndarray_to_rank0(arr)

        np.testing.assert_array_equal(result, arr)

    def test_gather_large_array(self):
        """Test gathering larger array."""
        from utils.distributed import gather_ndarray_to_rank0

        arr = np.random.randn(1000, 10)
        result = gather_ndarray_to_rank0(arr)

        np.testing.assert_array_equal(result, arr)
