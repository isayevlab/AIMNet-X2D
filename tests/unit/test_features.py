"""
Unit tests for molecular feature computation.

Tests SMILES parsing, atomic number extraction, and feature computation.
"""

import pytest
import numpy as np

from datasets.features import (
    partial_parse_atomic_numbers,
    compute_all,
    build_numba_adjacency_list,
    compute_multi_hop_edges_bfs_numba,
)


class TestPartialParseAtomicNumbers:
    """Tests for partial_parse_atomic_numbers function."""

    def test_simple_smiles(self):
        """Test parsing simple SMILES strings."""
        result = partial_parse_atomic_numbers("C")
        assert result is not None
        assert 6 in result  # Carbon

    def test_ethanol(self):
        """Test parsing ethanol (CCO)."""
        result = partial_parse_atomic_numbers("CCO")
        assert result is not None
        # Should contain 2 carbons, 1 oxygen, and hydrogens
        assert np.sum(result == 6) == 2  # 2 carbons
        assert np.sum(result == 8) == 1  # 1 oxygen
        assert np.sum(result == 1) > 0   # hydrogens added

    def test_invalid_smiles_returns_none(self):
        """Test that invalid SMILES returns None."""
        result = partial_parse_atomic_numbers("invalid_smiles")
        assert result is None

    def test_empty_smiles_returns_empty_array(self):
        """Test that empty SMILES returns empty array."""
        result = partial_parse_atomic_numbers("")
        assert result is not None
        assert len(result) == 0

    def test_benzene(self):
        """Test parsing benzene ring."""
        result = partial_parse_atomic_numbers("c1ccccc1")
        assert result is not None
        assert np.sum(result == 6) == 6  # 6 carbons

    def test_returns_numpy_array(self):
        """Test that result is a numpy array with correct dtype."""
        result = partial_parse_atomic_numbers("CC")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32


class TestComputeAll:
    """Tests for compute_all function."""

    def test_simple_molecule(self):
        """Test feature computation for simple molecule."""
        result = compute_all("CCO", max_hops=3)
        assert result is not None

        # Check required keys
        assert 'multi_hop_edges' in result
        assert 'atom_features' in result
        assert 'atomic_numbers' in result
        assert 'total_charge' in result

    def test_multi_hop_edges_structure(self):
        """Test that multi_hop_edges has correct structure."""
        result = compute_all("CC", max_hops=3)
        assert result is not None

        edges = result['multi_hop_edges']
        assert len(edges) == 3  # 3 hops

        # First hop should have edges (C-C bond)
        assert edges[0].shape[0] > 0

    def test_atom_features_structure(self):
        """Test that atom_features has correct structure."""
        result = compute_all("CCO", max_hops=3)
        assert result is not None

        features = result['atom_features']
        assert 'atom_type' in features
        assert 'degree' in features
        assert 'hydrogen_count' in features
        assert 'hybridization' in features

    def test_invalid_smiles_returns_none(self):
        """Test that invalid SMILES returns None."""
        result = compute_all("invalid", max_hops=3)
        assert result is None

    def test_chiral_molecule(self):
        """Test feature computation for chiral molecule."""
        result = compute_all("C[C@H](O)F", max_hops=3)
        assert result is not None

        # Should have chiral tensors (indices for tetrahedral centers)
        assert 'chiral_tensors' in result
        assert len(result['chiral_tensors']) > 0  # Should have at least one chiral center

    def test_different_max_hops(self):
        """Test that max_hops parameter works correctly."""
        result_2 = compute_all("CCCC", max_hops=2)
        result_3 = compute_all("CCCC", max_hops=3)

        assert result_2 is not None
        assert result_3 is not None

        assert len(result_2['multi_hop_edges']) == 2
        assert len(result_3['multi_hop_edges']) == 3


class TestBFSEdgeComputation:
    """Tests for BFS-based multi-hop edge computation."""

    def test_linear_chain(self):
        """Test BFS on linear molecule (C-C-C)."""
        # Create adjacency matrix for C-C-C
        adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=np.int32)

        adj_list = build_numba_adjacency_list(adj)
        edges = compute_multi_hop_edges_bfs_numba(adj_list, max_hops=2)

        # Hop 1: direct bonds (0-1, 1-2)
        hop1 = edges[0]
        assert hop1.shape[1] > 0

        # Hop 2: 2-hop paths (0-2)
        hop2 = edges[1]
        assert hop2.shape[1] > 0

    def test_cycle(self):
        """Test BFS on cyclic molecule (triangle)."""
        # Create adjacency matrix for triangle
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=np.int32)

        adj_list = build_numba_adjacency_list(adj)
        edges = compute_multi_hop_edges_bfs_numba(adj_list, max_hops=2)

        # All atoms are 1-hop from each other
        hop1 = edges[0]
        assert hop1.shape[1] == 6  # 3 atoms * 2 directions each

    def test_single_atom(self):
        """Test BFS on single atom (no bonds)."""
        adj = np.array([[0]], dtype=np.int32)

        adj_list = build_numba_adjacency_list(adj)
        edges = compute_multi_hop_edges_bfs_numba(adj_list, max_hops=2)

        # No edges at any hop
        assert edges[0].shape[1] == 0
        assert edges[1].shape[1] == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_long_smiles(self):
        """Test handling of very long SMILES."""
        # Create a long chain
        long_smiles = "C" * 50
        result = partial_parse_atomic_numbers(long_smiles)
        # Should either succeed or gracefully return None
        # (depends on RDKit's handling)

    def test_unicode_in_smiles(self):
        """Test that unicode characters are handled gracefully."""
        result = partial_parse_atomic_numbers("\u2603")  # snowman
        assert result is None

    def test_whitespace_smiles(self):
        """Test handling of whitespace in SMILES."""
        result = partial_parse_atomic_numbers(" CC ")
        # RDKit may or may not handle this
        # Just ensure no exception is raised
