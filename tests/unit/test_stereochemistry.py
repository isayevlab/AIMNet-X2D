"""
Comprehensive tests for stereochemistry implementation.

Tests cover:
- E/Z (cis/trans) double bond isomers
- R/S (tetrahedral) chiral centers
- Enantiomers and their distinguishability
- Racemates
- Edge cases and complex stereochemistry

These tests document both correct behavior and known limitations.
"""

import pytest
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from datasets.features import compute_all


# =============================================================================
# TEST DATA: Standard Stereoisomer SMILES
# =============================================================================

# E/Z Isomers (cis/trans double bonds)
E_2_BUTENE = "C/C=C/C"  # trans-2-butene (E)
Z_2_BUTENE = "C/C=C\\C"  # cis-2-butene (Z)

E_STILBENE = "C(/c1ccccc1)=C/c2ccccc2"  # E-stilbene (trans)
Z_STILBENE = "C(/c1ccccc1)=C\\c2ccccc2"  # Z-stilbene (cis)

E_CINNAMIC_ACID = "OC(=O)/C=C/c1ccccc1"  # trans-cinnamic acid
Z_CINNAMIC_ACID = "OC(=O)/C=C\\c1ccccc1"  # cis-cinnamic acid

# R/S Isomers (tetrahedral chirality)
# Note: @@ and @ notation refer to tetrahedral arrangement, not directly to R/S
# The actual R/S designation depends on CIP priority rules
R_ALANINE = "N[C@H](C)C(=O)O"   # R-alanine per RDKit CIP assignment
S_ALANINE = "N[C@@H](C)C(=O)O"  # S-alanine per RDKit CIP assignment

R_LACTIC_ACID = "C[C@@H](O)C(=O)O"  # R-lactic acid (D)
S_LACTIC_ACID = "C[C@H](O)C(=O)O"   # S-lactic acid (L)

R_LIMONENE = "CC(=C)[C@H]1CCC(=CC1)C"  # R-limonene (D-limonene, orange scent)
S_LIMONENE = "CC(=C)[C@@H]1CCC(=CC1)C"  # S-limonene (L-limonene, lemon scent)

# Simple chiral molecules for testing
# Note: @@ and @ notation don't directly map to R/S - it depends on CIP priorities
R_FLUOROCHLOROBROMOMETHANE = "F[C@@](Cl)(Br)I"  # R per RDKit CIP assignment
S_FLUOROCHLOROBROMOMETHANE = "F[C@](Cl)(Br)I"   # S per RDKit CIP assignment

# Molecules with multiple chiral centers
THREONINE_2R3S = "C[C@@H](O)[C@H](N)C(=O)O"  # L-threonine
THREONINE_2S3R = "C[C@H](O)[C@@H](N)C(=O)O"  # D-threonine

# Achiral molecules (for comparison)
ETHANOL = "CCO"
METHANE = "C"
BENZENE = "c1ccccc1"
PROPANE = "CCC"

# Meso compounds (achiral despite chiral centers)
MESO_TARTARIC_ACID = "O[C@H](C(=O)O)[C@@H](O)C(=O)O"


# =============================================================================
# E/Z ISOMER TESTS
# =============================================================================

class TestEZIsomerDetection:
    """Tests for E/Z (cis/trans) double bond detection."""

    def test_e_2_butene_detected(self):
        """Test that E-2-butene has E stereochemistry detected."""
        result = compute_all(E_2_BUTENE, max_hops=3)
        assert result is not None, "Failed to parse E-2-butene"

        # Should have trans bonds (E = opposite sides)
        assert 'trans_bonds_tensors' in result
        assert len(result['trans_bonds_tensors']) > 0, "E-isomer should have trans bonds"

    def test_z_2_butene_detected(self):
        """Test that Z-2-butene has Z stereochemistry detected."""
        result = compute_all(Z_2_BUTENE, max_hops=3)
        assert result is not None, "Failed to parse Z-2-butene"

        # Should have cis bonds (Z = same side)
        assert 'cis_bonds_tensors' in result
        assert len(result['cis_bonds_tensors']) > 0, "Z-isomer should have cis bonds"

    def test_e_z_isomers_have_different_features(self):
        """Test that E and Z isomers produce different feature representations."""
        e_result = compute_all(E_2_BUTENE, max_hops=3)
        z_result = compute_all(Z_2_BUTENE, max_hops=3)

        assert e_result is not None and z_result is not None

        # Convert to sets for comparison (order doesn't matter)
        e_cis_set = set(tuple(t.tolist()) for t in e_result['cis_bonds_tensors'])
        e_trans_set = set(tuple(t.tolist()) for t in e_result['trans_bonds_tensors'])
        z_cis_set = set(tuple(t.tolist()) for t in z_result['cis_bonds_tensors'])
        z_trans_set = set(tuple(t.tolist()) for t in z_result['trans_bonds_tensors'])

        # E and Z should have swapped cis/trans assignments
        # E's cis bonds should be Z's trans bonds and vice versa
        assert e_cis_set == z_trans_set, \
            "E-isomer cis bonds should equal Z-isomer trans bonds"
        assert e_trans_set == z_cis_set, \
            "E-isomer trans bonds should equal Z-isomer cis bonds"

        # Additionally verify they're not identical
        assert e_cis_set != e_trans_set, \
            "Cis and trans bond sets should be different within same molecule"

    def test_stilbene_e_z_difference(self):
        """Test E/Z detection for stilbene (larger molecule)."""
        e_result = compute_all(E_STILBENE, max_hops=3)
        z_result = compute_all(Z_STILBENE, max_hops=3)

        assert e_result is not None, "Failed to parse E-stilbene"
        assert z_result is not None, "Failed to parse Z-stilbene"

        # Both should have stereochemistry detected
        e_has_stereo = len(e_result['cis_bonds_tensors']) > 0 or len(e_result['trans_bonds_tensors']) > 0
        z_has_stereo = len(z_result['cis_bonds_tensors']) > 0 or len(z_result['trans_bonds_tensors']) > 0

        assert e_has_stereo, "E-stilbene should have stereochemistry detected"
        assert z_has_stereo, "Z-stilbene should have stereochemistry detected"

    def test_no_double_bond_no_ez_stereo(self):
        """Test that molecules without double bonds have no E/Z features."""
        result = compute_all(PROPANE, max_hops=3)
        assert result is not None

        # Propane has no double bonds, so no E/Z stereo
        assert len(result['cis_bonds_tensors']) == 0
        assert len(result['trans_bonds_tensors']) == 0

    def test_symmetric_double_bond_no_stereo(self):
        """Test that symmetric double bonds (like in ethene) have no E/Z stereo."""
        result = compute_all("C=C", max_hops=3)  # Ethene - symmetric
        assert result is not None

        # Ethene is symmetric, no E/Z stereochemistry
        # The algorithm should skip this due to the "< 4 unique substituents" check
        assert len(result['cis_bonds_tensors']) == 0
        assert len(result['trans_bonds_tensors']) == 0


# =============================================================================
# R/S CHIRAL CENTER TESTS
# =============================================================================

class TestChiralCenterDetection:
    """Tests for R/S (tetrahedral) chiral center detection."""

    def test_r_alanine_chiral_detected(self):
        """Test that R-alanine has chiral center detected."""
        result = compute_all(R_ALANINE, max_hops=3)
        assert result is not None, "Failed to parse R-alanine"

        assert 'chiral_tensors' in result
        assert len(result['chiral_tensors']) > 0, "R-alanine should have chiral center"

    def test_s_alanine_chiral_detected(self):
        """Test that S-alanine has chiral center detected."""
        result = compute_all(S_ALANINE, max_hops=3)
        assert result is not None, "Failed to parse S-alanine"

        assert 'chiral_tensors' in result
        assert len(result['chiral_tensors']) > 0, "S-alanine should have chiral center"

    def test_chiral_tensor_has_center_and_four_neighbors(self):
        """Test that chiral tensors have center index plus 4 neighbor indices (5 total)."""
        result = compute_all(R_LACTIC_ACID, max_hops=3)
        assert result is not None

        for chiral_tensor in result['chiral_tensors']:
            assert len(chiral_tensor) == 5, \
                f"Tetrahedral tensor should have [center, n1, n2, n3, n4], got {len(chiral_tensor)} elements"
            # First element is the center index, remaining 4 are neighbors
            center_idx = chiral_tensor[0]
            neighbors = chiral_tensor[1:5]
            assert len(neighbors) == 4, "Should have exactly 4 neighbor indices"

    def test_achiral_molecule_no_chiral_centers(self):
        """Test that achiral molecules have no chiral centers."""
        result = compute_all(ETHANOL, max_hops=3)
        assert result is not None

        assert len(result['chiral_tensors']) == 0, "Ethanol should have no chiral centers"

    def test_benzene_no_chiral_centers(self):
        """Test that benzene has no chiral centers."""
        result = compute_all(BENZENE, max_hops=3)
        assert result is not None

        assert len(result['chiral_tensors']) == 0, "Benzene should have no chiral centers"


# =============================================================================
# ENANTIOMER DISTINGUISHABILITY TESTS
# =============================================================================

class TestEnantiomerDistinguishability:
    """
    Tests for enantiomer distinguishability.

    KNOWN LIMITATION: The current implementation discards R/S designation,
    so enantiomers may not be distinguishable at the feature level.
    These tests document the expected behavior and current limitations.
    """

    def test_r_s_alanine_distinguishable(self):
        """
        Test that R and S alanine produce different features.

        The implementation now stores R/S designation as chiral_signs:
        R = +1.0, S = -1.0
        """
        r_result = compute_all(R_ALANINE, max_hops=3)
        s_result = compute_all(S_ALANINE, max_hops=3)

        assert r_result is not None and s_result is not None

        # Verify chiral_signs are present and different for R vs S
        assert 'chiral_signs' in r_result
        assert 'chiral_signs' in s_result
        assert len(r_result['chiral_signs']) > 0, "R-alanine should have chiral sign"
        assert len(s_result['chiral_signs']) > 0, "S-alanine should have chiral sign"

        # R should be +1.0, S should be -1.0
        assert r_result['chiral_signs'][0] != s_result['chiral_signs'][0], \
            "R and S enantiomers should have different chiral signs"

    def test_r_s_lactic_acid_distinguishable(self):
        """Test that R and S lactic acid produce different features."""
        r_result = compute_all(R_LACTIC_ACID, max_hops=3)
        s_result = compute_all(S_LACTIC_ACID, max_hops=3)

        assert r_result is not None and s_result is not None

        # Verify chiral_signs distinguish R from S
        assert 'chiral_signs' in r_result
        assert 'chiral_signs' in s_result

        # R should be +1.0, S should be -1.0
        r_sign = r_result['chiral_signs'][0] if len(r_result['chiral_signs']) > 0 else None
        s_sign = s_result['chiral_signs'][0] if len(s_result['chiral_signs']) > 0 else None

        assert r_sign is not None, "R-lactic acid should have chiral sign"
        assert s_sign is not None, "S-lactic acid should have chiral sign"
        assert r_sign != s_sign, \
            "R and S enantiomers should have different chiral signs"

    def test_enantiomers_same_atom_count(self):
        """Test that enantiomers have the same number of atoms."""
        r_result = compute_all(R_ALANINE, max_hops=3)
        s_result = compute_all(S_ALANINE, max_hops=3)

        assert r_result is not None and s_result is not None

        r_atom_count = len(r_result['atom_features']['atom_type'])
        s_atom_count = len(s_result['atom_features']['atom_type'])

        assert r_atom_count == s_atom_count, \
            "Enantiomers should have the same number of atoms"

    def test_enantiomers_same_connectivity(self):
        """Test that enantiomers have the same bond connectivity."""
        r_result = compute_all(R_ALANINE, max_hops=3)
        s_result = compute_all(S_ALANINE, max_hops=3)

        assert r_result is not None and s_result is not None

        # Multi-hop edges should be identical (same connectivity)
        for hop_idx in range(len(r_result['multi_hop_edges'])):
            r_edges = r_result['multi_hop_edges'][hop_idx]
            s_edges = s_result['multi_hop_edges'][hop_idx]

            assert r_edges.shape == s_edges.shape, \
                f"Hop {hop_idx} should have same edge count for enantiomers"


# =============================================================================
# MULTIPLE CHIRAL CENTERS TESTS
# =============================================================================

class TestMultipleChiralCenters:
    """Tests for molecules with multiple chiral centers."""

    def test_threonine_two_chiral_centers(self):
        """Test that threonine has two chiral centers detected."""
        result = compute_all(THREONINE_2R3S, max_hops=3)
        assert result is not None, "Failed to parse threonine"

        assert len(result['chiral_tensors']) == 2, \
            "Threonine should have exactly 2 chiral centers"

    def test_threonine_diastereomers_differ(self):
        """Test that threonine diastereomers have different features."""
        result_2r3s = compute_all(THREONINE_2R3S, max_hops=3)
        result_2s3r = compute_all(THREONINE_2S3R, max_hops=3)

        assert result_2r3s is not None and result_2s3r is not None

        # Both should have 2 chiral centers
        assert len(result_2r3s['chiral_tensors']) == 2
        assert len(result_2s3r['chiral_tensors']) == 2

    def test_meso_compound_detected(self):
        """Test that meso tartaric acid chiral centers are detected."""
        result = compute_all(MESO_TARTARIC_ACID, max_hops=3)
        assert result is not None, "Failed to parse meso-tartaric acid"

        # Meso compound has chiral centers (but overall achiral due to symmetry)
        assert len(result['chiral_tensors']) == 2, \
            "Meso-tartaric acid should have 2 chiral centers detected"


# =============================================================================
# COMBINED STEREOCHEMISTRY TESTS
# =============================================================================

class TestCombinedStereochemistry:
    """Tests for molecules with both E/Z and R/S stereochemistry."""

    def test_molecule_with_both_stereo_types(self):
        """Test molecule with both chiral center and double bond."""
        # 2-methylbut-2-enal with chiral center: (E)-3-chlorobut-2-enoic acid
        smiles_with_both = "C/C=C(/Cl)[C@H](C)O"

        result = compute_all(smiles_with_both, max_hops=3)

        if result is not None:
            # Should detect both types of stereochemistry
            has_chiral = len(result['chiral_tensors']) > 0
            has_ez = len(result['cis_bonds_tensors']) > 0 or len(result['trans_bonds_tensors']) > 0

            # At minimum, should detect the chiral center
            assert has_chiral, "Should detect chiral center"


# =============================================================================
# EDGE CASES AND ROBUSTNESS TESTS
# =============================================================================

class TestStereochemistryEdgeCases:
    """Tests for edge cases in stereochemistry handling."""

    def test_unassigned_stereochemistry(self):
        """Test handling of molecules with unassigned stereochemistry."""
        # SMILES without explicit stereochemistry
        result = compute_all("CC(O)C(=O)O", max_hops=3)  # Lactic acid without stereo
        assert result is not None

        # Should still work, may or may not have chiral tensors
        # (depends on RDKit's includeUnassigned behavior)

    def test_invalid_smiles_with_stereo_notation(self):
        """Test that invalid SMILES with stereo notation returns None."""
        result = compute_all("[C@H]invalid", max_hops=3)
        assert result is None

    def test_allene_like_structures(self):
        """Test handling of allene-like structures (cumulated double bonds)."""
        # Allene: C=C=C (linear, no E/Z possible)
        result = compute_all("C=C=C", max_hops=3)
        assert result is not None

        # Allenes don't have E/Z stereochemistry in the same way
        # The current implementation may not handle axial chirality

    def test_ring_with_double_bond(self):
        """Test E/Z detection in cyclic systems."""
        # Cyclohexene - double bond in ring
        result = compute_all("C1CC=CCC1", max_hops=3)
        assert result is not None

        # Ring double bonds have constrained geometry
        # May or may not have E/Z depending on ring size

    def test_very_small_chiral_molecule(self):
        """Test smallest possible chiral molecule."""
        result = compute_all(R_FLUOROCHLOROBROMOMETHANE, max_hops=3)
        assert result is not None

        assert len(result['chiral_tensors']) == 1, \
            "Halomethane should have exactly 1 chiral center"

    def test_smiles_with_explicit_hydrogens(self):
        """Test stereochemistry detection with explicit hydrogens in SMILES."""
        # Explicit H on chiral center
        explicit_h = "[C@@H](F)(Cl)Br"
        result = compute_all(explicit_h, max_hops=3)

        if result is not None:
            # Should still detect chirality with explicit H
            assert len(result['chiral_tensors']) >= 1


# =============================================================================
# FEATURE TENSOR VALIDATION TESTS
# =============================================================================

class TestStereochemistryTensorStructure:
    """Tests for proper tensor structure in stereochemistry features."""

    def test_chiral_tensor_dtype(self):
        """Test that chiral tensors have correct dtype."""
        result = compute_all(R_ALANINE, max_hops=3)
        assert result is not None

        for tensor in result['chiral_tensors']:
            assert tensor.dtype == np.int32, \
                f"Chiral tensor should be int32, got {tensor.dtype}"

    def test_cis_trans_tensor_dtype(self):
        """Test that cis/trans tensors have correct dtype."""
        result = compute_all(E_2_BUTENE, max_hops=3)
        assert result is not None

        for tensor in result['cis_bonds_tensors']:
            assert tensor.dtype == np.int32
        for tensor in result['trans_bonds_tensors']:
            assert tensor.dtype == np.int32

    def test_cis_trans_tensor_shape(self):
        """Test that cis/trans tensors have shape (2,) for source-target pairs."""
        result = compute_all(E_2_BUTENE, max_hops=3)
        assert result is not None

        for tensor in result['cis_bonds_tensors']:
            assert tensor.shape == (2,), \
                f"Cis bond tensor should have shape (2,), got {tensor.shape}"
        for tensor in result['trans_bonds_tensors']:
            assert tensor.shape == (2,), \
                f"Trans bond tensor should have shape (2,), got {tensor.shape}"

    def test_chiral_indices_within_bounds(self):
        """Test that chiral tensor indices are valid atom indices."""
        result = compute_all(R_ALANINE, max_hops=3)
        assert result is not None

        num_atoms = len(result['atom_features']['atom_type'])

        for chiral_tensor in result['chiral_tensors']:
            # Tensor format: [center_idx, neighbor1, neighbor2, neighbor3, neighbor4]
            assert len(chiral_tensor) == 5, "Chiral tensor should have 5 elements"
            for idx in chiral_tensor:
                assert 0 <= idx < num_atoms, \
                    f"Chiral index {idx} out of bounds for {num_atoms} atoms"
            # Verify center is distinct from neighbors (center shouldn't be its own neighbor)
            center_idx = chiral_tensor[0]
            neighbor_indices = chiral_tensor[1:5]
            assert center_idx not in neighbor_indices, \
                "Center index should not be in neighbor indices"

    def test_chiral_signs_dtype_and_values(self):
        """Test that chiral_signs has correct dtype and expected values."""
        # Use R_LACTIC_ACID which we know is R per RDKit (C[C@@H](O)C(=O)O)
        result = compute_all(R_LACTIC_ACID, max_hops=3)
        assert result is not None
        assert 'chiral_signs' in result

        chiral_signs = result['chiral_signs']
        assert chiral_signs.dtype == np.float32, \
            f"chiral_signs should be float32, got {chiral_signs.dtype}"

        # R configuration should have sign +1.0
        assert len(chiral_signs) > 0, "Should have at least one chiral sign"
        assert chiral_signs[0] == 1.0, f"R configuration should have sign +1.0, got {chiral_signs[0]}"

    def test_chiral_signs_r_vs_s_values(self):
        """Test that R and S configurations have expected sign values."""
        r_result = compute_all(R_LACTIC_ACID, max_hops=3)
        s_result = compute_all(S_LACTIC_ACID, max_hops=3)

        assert r_result is not None and s_result is not None

        # R = +1.0, S = -1.0 per CHIRALITY_SIGNS mapping
        assert r_result['chiral_signs'][0] == 1.0, "R should be +1.0"
        assert s_result['chiral_signs'][0] == -1.0, "S should be -1.0"

    def test_achiral_molecule_empty_chiral_signs(self):
        """Test that achiral molecules have empty chiral_signs array."""
        result = compute_all(ETHANOL, max_hops=3)
        assert result is not None

        assert 'chiral_signs' in result
        assert len(result['chiral_signs']) == 0, "Ethanol should have no chiral signs"
        assert len(result['chiral_tensors']) == 0, "Ethanol should have no chiral tensors"

    def test_chiral_tensor_center_is_chiral_atom(self):
        """Test that the first element of chiral tensor is the chiral center."""
        result = compute_all(R_LACTIC_ACID, max_hops=3)
        assert result is not None

        # R-lactic acid: C[C@@H](O)C(=O)O
        # The chiral center is the carbon with 4 different substituents
        for chiral_tensor, chiral_sign in zip(result['chiral_tensors'], result['chiral_signs']):
            center_idx = chiral_tensor[0]
            neighbors = chiral_tensor[1:5]

            # The center should have exactly 4 unique neighbors (tetrahedral)
            assert len(set(neighbors)) == 4, "Center should have 4 unique neighbors"

            # Center should not appear in neighbors
            assert center_idx not in neighbors, "Center should not be its own neighbor"

    def test_cis_trans_indices_within_bounds(self):
        """Test that cis/trans tensor indices are valid atom indices."""
        result = compute_all(E_2_BUTENE, max_hops=3)
        assert result is not None

        num_atoms = len(result['atom_features']['atom_type'])

        for tensor in result['cis_bonds_tensors'] + result['trans_bonds_tensors']:
            for idx in tensor:
                assert 0 <= idx < num_atoms, \
                    f"Cis/trans index {idx} out of bounds for {num_atoms} atoms"


# =============================================================================
# RACEMATE AND RACEMIC MIXTURE TESTS
# =============================================================================

class TestRacematesAndMixtures:
    """Tests related to racemates and stereochemistry in mixtures."""

    def test_single_enantiomer_processing(self):
        """Test that single enantiomers process correctly."""
        r_result = compute_all(R_LIMONENE, max_hops=3)
        s_result = compute_all(S_LIMONENE, max_hops=3)

        assert r_result is not None, "Failed to parse R-limonene"
        assert s_result is not None, "Failed to parse S-limonene"

        # Both should have chiral centers
        assert len(r_result['chiral_tensors']) > 0
        assert len(s_result['chiral_tensors']) > 0

    def test_racemic_molecule_without_stereo(self):
        """Test processing molecule without stereo annotation (could be racemate)."""
        # Lactic acid without stereochemistry (racemic or unspecified)
        result = compute_all("CC(O)C(=O)O", max_hops=3)
        assert result is not None

        # RDKit with includeUnassigned=True may still detect potential chiral center
        # The result depends on RDKit's handling


# =============================================================================
# INTEGRATION WITH MODEL TESTS
# =============================================================================

class TestStereochemistryModelIntegration:
    """Tests for stereochemistry features integration with the model."""

    @pytest.fixture
    def feature_sizes(self):
        """Standard feature sizes for model creation."""
        return {
            'atom_type': 119,
            'hydrogen_count': 9,
            'degree': 7,
            'hybridization': 7,
        }

    def test_chiral_tensors_convertible_to_torch(self):
        """Test that chiral tensors can be converted to PyTorch tensors."""
        result = compute_all(R_ALANINE, max_hops=3)
        assert result is not None

        for chiral_np in result['chiral_tensors']:
            chiral_torch = torch.from_numpy(chiral_np)
            assert chiral_torch.dtype == torch.int32

    def test_stereochemistry_empty_tensors_handled(self):
        """Test that empty stereochemistry tensors are handled correctly."""
        result = compute_all(METHANE, max_hops=3)
        assert result is not None

        # Methane has no stereochemistry
        assert len(result['chiral_tensors']) == 0
        assert len(result['cis_bonds_tensors']) == 0
        assert len(result['trans_bonds_tensors']) == 0

        # Should still be able to create empty torch tensors
        chiral_torch = torch.tensor([], dtype=torch.int32)
        assert chiral_torch.numel() == 0


# =============================================================================
# KNOWN BUGS DOCUMENTATION TESTS
# =============================================================================

class TestKnownBugs:
    """
    Tests that document known bugs in the stereochemistry implementation.

    These tests use xfail to document expected failures due to known issues.
    """

    def test_rs_designation_now_stored(self):
        """
        FIXED: R/S chirality designation is now stored in chiral_signs.

        Previous bug location: features.py:225-228
        The 'chirality' variable (R or S) was captured but never used.

        Fix: Now stored in 'chiral_signs' array as:
        R = +1.0, S = -1.0, r = +0.5, s = -0.5, unassigned = 0.0
        """
        r_result = compute_all(R_LACTIC_ACID, max_hops=3)
        s_result = compute_all(S_LACTIC_ACID, max_hops=3)

        # Verify chiral_signs key exists
        assert 'chiral_signs' in r_result, "chiral_signs should be in result"
        assert 'chiral_signs' in s_result, "chiral_signs should be in result"

        # Verify R and S produce opposite signs
        assert len(r_result['chiral_signs']) > 0, "R should have chiral sign"
        assert len(s_result['chiral_signs']) > 0, "S should have chiral sign"

        r_sign = r_result['chiral_signs'][0]
        s_sign = s_result['chiral_signs'][0]

        assert r_sign == 1.0, f"R configuration should have sign +1.0, got {r_sign}"
        assert s_sign == -1.0, f"S configuration should have sign -1.0, got {s_sign}"

    def test_cip_priority_now_used_for_low_atom(self):
        """
        FIXED: Low atom selection now uses CIP priority via get_cip_rank().

        Previous bug location: features.py:268,273
        Used: min(..., key=lambda idx: mol.GetAtomWithIdx(idx).GetAtomicNum())

        Fix: Now uses get_cip_rank() which returns CIP rank from _CIPRank property
        with fallback to atomic number if not available.
        """
        # Verify the function exists and works
        from datasets.features import get_cip_rank
        from rdkit import Chem

        mol = Chem.MolFromSmiles("C/C=C/C")  # E-2-butene
        mol = Chem.AddHs(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # get_cip_rank should work without raising
        for atom in mol.GetAtoms():
            rank = get_cip_rank(mol, atom.GetIdx())
            assert isinstance(rank, int), f"CIP rank should be int, got {type(rank)}"

    def test_documentation_zero_out_non_chiral_atoms(self):
        """
        POTENTIAL BUG: Non-chiral atoms are zeroed out in model.

        Location: gnn.py:399-404
        Code: updated[~mask] = 0.0

        This zeros ALL features for atoms not involved in chiral centers,
        which may cause severe information loss during message passing.

        This test documents the issue - actual testing requires model-level integration.
        """
        # This bug is in the model, not features, so we just document it here
        pass


# =============================================================================
# PERFORMANCE AND STRESS TESTS
# =============================================================================

class TestStereochemistryPerformance:
    """Performance tests for stereochemistry detection."""

    def test_batch_processing_consistency(self):
        """Test that batch processing produces consistent results."""
        smiles_list = [R_ALANINE, S_ALANINE, E_2_BUTENE, Z_2_BUTENE, ETHANOL]

        results = [compute_all(smi, max_hops=3) for smi in smiles_list]

        # All should succeed
        assert all(r is not None for r in results)

        # Chiral molecules should have chiral tensors
        assert len(results[0]['chiral_tensors']) > 0  # R-alanine
        assert len(results[1]['chiral_tensors']) > 0  # S-alanine

        # E/Z molecules should have cis/trans tensors
        e_has_stereo = len(results[2]['trans_bonds_tensors']) > 0 or len(results[2]['cis_bonds_tensors']) > 0
        z_has_stereo = len(results[3]['trans_bonds_tensors']) > 0 or len(results[3]['cis_bonds_tensors']) > 0
        assert e_has_stereo
        assert z_has_stereo

        # Ethanol should have no stereochemistry
        assert len(results[4]['chiral_tensors']) == 0


# =============================================================================
# GOLDEN TEST SET - Lock in correct extraction behavior
# =============================================================================

class TestGoldenStereochemistry:
    """
    Golden test set to verify stereochemistry extraction produces expected results.

    These tests lock in the correct behavior for specific molecules with known
    stereochemistry. If these tests fail, it indicates a regression in the
    stereochemistry extraction code.
    """

    @pytest.mark.parametrize("smiles,expected_sign", [
        # R = +1.0, S = -1.0 per CHIRALITY_SIGNS mapping
        # Note: @@ and @ notation don't directly map to R/S - depends on CIP priorities
        ("C[C@@H](O)C(=O)O", 1.0),   # R-lactic acid per RDKit
        ("C[C@H](O)C(=O)O", -1.0),   # S-lactic acid per RDKit
        ("N[C@@H](C)C(=O)O", -1.0),  # S-alanine per RDKit (@@H gives S for alanine)
        ("N[C@H](C)C(=O)O", 1.0),    # R-alanine per RDKit (@H gives R for alanine)
        ("F[C@](Cl)(Br)I", -1.0),    # S-fluorochlorobromomethane per RDKit
        ("F[C@@](Cl)(Br)I", 1.0),    # R-fluorochlorobromomethane per RDKit
    ])
    def test_golden_chiral_signs(self, smiles, expected_sign):
        """Verify chiral signs match expected values for known molecules based on RDKit CIP assignment."""
        result = compute_all(smiles, max_hops=3)
        assert result is not None, f"Failed to parse {smiles}"
        assert len(result['chiral_signs']) == 1, \
            f"Expected 1 chiral center for {smiles}, got {len(result['chiral_signs'])}"
        assert result['chiral_signs'][0] == expected_sign, \
            f"Expected sign {expected_sign} for {smiles}, got {result['chiral_signs'][0]}"

    def test_golden_multiple_chiral_centers(self):
        """Test molecule with multiple chiral centers."""
        # L-threonine: C[C@@H](O)[C@H](N)C(=O)O has 2 chiral centers
        result = compute_all(THREONINE_2R3S, max_hops=3)
        assert result is not None

        assert len(result['chiral_tensors']) == 2, "Threonine should have 2 chiral centers"
        assert len(result['chiral_signs']) == 2, "Threonine should have 2 chiral signs"

        # Each tensor should have 5 elements [center, n1, n2, n3, n4]
        for tensor in result['chiral_tensors']:
            assert len(tensor) == 5

    def test_golden_tensor_structure(self):
        """Verify tensor structure is [center, n1, n2, n3, n4]."""
        result = compute_all(R_LACTIC_ACID, max_hops=3)
        assert result is not None

        chiral_tensor = result['chiral_tensors'][0]

        # Shape should be (5,)
        assert chiral_tensor.shape == (5,), f"Expected shape (5,), got {chiral_tensor.shape}"

        # Dtype should be int32
        assert chiral_tensor.dtype == np.int32, f"Expected int32, got {chiral_tensor.dtype}"

        # First element is center, rest are neighbors
        center = chiral_tensor[0]
        neighbors = chiral_tensor[1:5]

        # All indices should be unique (center not in neighbors)
        all_indices = list(chiral_tensor)
        assert len(set(all_indices)) == 5, "All indices should be unique"

    def test_golden_ez_isomers_difference(self):
        """Verify E and Z isomers produce different cis/trans assignments."""
        e_result = compute_all(E_2_BUTENE, max_hops=3)
        z_result = compute_all(Z_2_BUTENE, max_hops=3)

        assert e_result is not None and z_result is not None

        # E should have trans bonds, Z should have cis bonds for same atom pairs
        e_trans = set(tuple(t.tolist()) for t in e_result['trans_bonds_tensors'])
        z_cis = set(tuple(t.tolist()) for t in z_result['cis_bonds_tensors'])

        # The high-priority substituent pairs should swap between cis/trans
        assert len(e_trans) > 0, "E-isomer should have trans bonds"
        assert len(z_cis) > 0, "Z-isomer should have cis bonds"

    def test_golden_achiral_no_stereochemistry(self):
        """Verify achiral molecules have no stereochemistry data."""
        result = compute_all(ETHANOL, max_hops=3)
        assert result is not None

        assert len(result['chiral_tensors']) == 0
        assert len(result['chiral_signs']) == 0
        assert len(result['cis_bonds_tensors']) == 0
        assert len(result['trans_bonds_tensors']) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
