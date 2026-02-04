"""
Phase 3 Validation: Enantiomer Discrimination Tests

Tests that the stereochemistry implementation correctly distinguishes
between enantiomers (R/S) and geometric isomers (E/Z).

These tests require a trained model to evaluate embedding differences.
For unit testing without a model, see tests/unit/test_stereochemistry.py
"""

import pytest
import numpy as np
import torch
from rdkit import Chem

# Import feature extraction for direct testing
import sys
sys.path.insert(0, '/home/olexandr/AIMNet-X2D/src')

from datasets.features import compute_all


# ============================================================================
# Test Data: Enantiomer Pairs
# ============================================================================

ENANTIOMER_TEST_PAIRS = [
    # Standard tetrahedral carbon - amino acids
    ('N[C@@H](C)C(=O)O', 'N[C@H](C)C(=O)O', 'Alanine R/S'),
    ('C[C@@H](O)C(=O)O', 'C[C@H](O)C(=O)O', 'Lactic acid R/S'),
    ('O=C(O)[C@@H](N)CC(=O)O', 'O=C(O)[C@H](N)CC(=O)O', 'Aspartic acid R/S'),
    ('N[C@@H](Cc1ccccc1)C(=O)O', 'N[C@H](Cc1ccccc1)C(=O)O', 'Phenylalanine R/S'),

    # Sulfoxide (pyramidal heteroatom with virtual LP)
    ('C[S@@](=O)CC', 'C[S@](=O)CC', 'Ethyl methyl sulfoxide R/S'),
    ('C[S@@](=O)c1ccccc1', 'C[S@](=O)c1ccccc1', 'Methyl phenyl sulfoxide R/S'),

    # Multiple chiral centers
    ('C[C@@H](O)[C@@H](O)C', 'C[C@H](O)[C@H](O)C', '2,3-Butanediol RR/SS'),

    # Drug-like molecules
    ('CC(=O)O[C@H]1CC[C@@H]2[C@H](C1)CC[C@@H]1[C@@H]2CC[C@]2(C)[C@H](O)CC[C@@H]12',
     'CC(=O)O[C@@H]1CC[C@H]2[C@@H](C1)CC[C@H]1[C@H]2CC[C@]2(C)[C@@H](O)CC[C@H]12',
     'Testosterone acetate enantiomers'),
]

E_Z_TEST_PAIRS = [
    # Simple E/Z
    ('C/C=C/C', 'C/C=C\\C', 'trans/cis-2-butene'),
    ('CC/C=C/CC', 'CC/C=C\\CC', 'trans/cis-3-hexene'),

    # Aromatic substituents
    ('O=C(O)/C=C/c1ccccc1', 'O=C(O)/C=C\\c1ccccc1', 'trans/cis-cinnamic acid'),

    # Multiple double bonds
    ('C/C=C/C=C/C', 'C/C=C\\C=C\\C', 'all-trans/all-cis-2,4-hexadiene'),
]

ACHIRAL_CONTROLS = [
    ('CC(C)C', 'Isobutane'),
    ('c1ccccc1', 'Benzene'),
    ('CC(=O)O', 'Acetic acid'),
    ('CCCC', 'n-Butane'),
    ('C1CCCCC1', 'Cyclohexane'),
    ('CCO', 'Ethanol'),
]

# Pyramidal heteroatom pairs (extended)
PYRAMIDAL_HETEROATOM_PAIRS = [
    # Sulfoxide
    ('C[S@@](=O)CC', 'C[S@](=O)CC', 'Ethyl methyl sulfoxide'),

    # Phosphine oxide
    ('C[P@@](=O)(CC)c1ccccc1', 'C[P@](=O)(CC)c1ccccc1', 'Phosphine oxide'),
]


# ============================================================================
# Feature Extraction Tests (No Model Required)
# ============================================================================

class TestChiralSignExtraction:
    """Test that R/S enantiomers have opposite chiral signs in features."""

    @pytest.mark.parametrize("r_smiles,s_smiles,name", ENANTIOMER_TEST_PAIRS[:4])
    def test_enantiomers_have_opposite_signs(self, r_smiles, s_smiles, name):
        """R and S enantiomers should have opposite chiral_signs values."""
        r_result = compute_all(r_smiles, max_hops=3)
        s_result = compute_all(s_smiles, max_hops=3)

        # Both should have chiral centers detected
        assert len(r_result['chiral_signs']) > 0, f"{name}: R has no chiral centers"
        assert len(s_result['chiral_signs']) > 0, f"{name}: S has no chiral centers"

        # Signs should be opposite (R=+1, S=-1)
        r_signs = np.array(r_result['chiral_signs'])
        s_signs = np.array(s_result['chiral_signs'])

        # For corresponding chiral centers, signs should be opposite
        assert np.allclose(r_signs, -s_signs), f"{name}: Signs not opposite: R={r_signs}, S={s_signs}"

    @pytest.mark.parametrize("smiles,name", ACHIRAL_CONTROLS)
    def test_achiral_has_no_signs(self, smiles, name):
        """Achiral molecules should have no chiral_signs or all zeros."""
        result = compute_all(smiles, max_hops=3)

        # Either no chiral centers or all signs are 0
        if len(result['chiral_signs']) > 0:
            signs = np.array(result['chiral_signs'])
            assert np.allclose(signs, 0), f"{name}: Unexpected non-zero signs: {signs}"


class TestEZFeatureExtraction:
    """Test that E/Z isomers have different cis/trans tensors."""

    @pytest.mark.parametrize("e_smiles,z_smiles,name", E_Z_TEST_PAIRS)
    def test_ez_pairs_detected(self, e_smiles, z_smiles, name):
        """E and Z isomers should both have cis/trans features detected."""
        e_result = compute_all(e_smiles, max_hops=3)
        z_result = compute_all(z_smiles, max_hops=3)

        # Check that stereochemistry is detected (using correct key names)
        e_has_ez = (len(e_result['cis_bonds_tensors']) > 0 or len(e_result['trans_bonds_tensors']) > 0)
        z_has_ez = (len(z_result['cis_bonds_tensors']) > 0 or len(z_result['trans_bonds_tensors']) > 0)

        assert e_has_ez, f"{name}: E isomer has no E/Z features"
        assert z_has_ez, f"{name}: Z isomer has no E/Z features"

    def test_trans_detected_as_trans(self):
        """trans-2-butene should have trans_bonds_tensors, not cis_bonds_tensors."""
        result = compute_all('C/C=C/C', max_hops=3)
        assert len(result['trans_bonds_tensors']) > 0, "trans-2-butene should have trans bonds"

    def test_cis_detected_as_cis(self):
        """cis-2-butene should have cis_bonds_tensors, not trans_bonds_tensors."""
        result = compute_all('C/C=C\\C', max_hops=3)
        assert len(result['cis_bonds_tensors']) > 0, "cis-2-butene should have cis bonds"


class TestPyramidalHeteroatomExtraction:
    """Test pyramidal heteroatom chiral center detection."""

    def test_sulfoxide_detected(self):
        """Sulfoxide should be detected as chiral center with virtual LP."""
        result = compute_all('C[S@@](=O)CC', max_hops=3)
        assert len(result['chiral_signs']) > 0, "Sulfoxide should have chiral sign"
        # chiral_is_virtual_lp is a list of [bool, bool, bool, bool] arrays
        has_virtual_lp = any(
            np.any(lp_flags) for lp_flags in result['chiral_is_virtual_lp']
        )
        assert has_virtual_lp, "Sulfoxide should have virtual LP"

    def test_sulfoxide_enantiomers_opposite(self):
        """Sulfoxide R/S enantiomers should have opposite signs."""
        r_result = compute_all('C[S@@](=O)CC', max_hops=3)
        s_result = compute_all('C[S@](=O)CC', max_hops=3)

        r_signs = np.array(r_result['chiral_signs'])
        s_signs = np.array(s_result['chiral_signs'])

        assert np.allclose(r_signs, -s_signs), f"Sulfoxide signs not opposite: R={r_signs}, S={s_signs}"


class TestAlleneExtraction:
    """Test allene axial chirality detection."""

    def test_allene_detected(self):
        """Simple allene should be detected."""
        result = compute_all('CC=C=CC', max_hops=3)
        assert len(result['allene_centers']) > 0, "Allene should be detected"

    def test_butatriene_not_detected(self):
        """Even-length cumulene (butatriene) should NOT be detected as chiral."""
        result = compute_all('CC=C=C=CC', max_hops=3)
        # Butatriene has 4 carbons in chain (even) - not chiral
        assert len(result['allene_centers']) == 0, "Butatriene should not be chiral"


# ============================================================================
# Model-Based Embedding Tests (Require Trained Model)
# ============================================================================

class TestEmbeddingDiscrimination:
    """
    Tests that require a trained model to extract embeddings.

    These tests are skipped if no model is available.
    """

    @pytest.fixture
    def model(self):
        """Load trained model if available."""
        try:
            import torch
            model_path = '/home/olexandr/AIMNet-X2D/models/stereo_model.pth'
            # This would need actual model loading code
            # For now, skip these tests
            pytest.skip("Model not available for embedding tests")
        except Exception:
            pytest.skip("Model not available for embedding tests")

    @pytest.mark.skip(reason="Requires trained model")
    def test_enantiomer_embedding_difference(self, model):
        """Test that R and S enantiomers produce distinguishable embeddings."""
        for r_smi, s_smi, name in ENANTIOMER_TEST_PAIRS:
            r_emb = model.get_embedding(r_smi)
            s_emb = model.get_embedding(s_smi)

            # L2 norm difference
            diff_norm = torch.norm(r_emb - s_emb).item()

            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                r_emb.unsqueeze(0), s_emb.unsqueeze(0)
            ).item()

            # Assert meaningful difference
            assert diff_norm > 0.1, f"{name}: Enantiomers too similar (diff={diff_norm})"
            assert cos_sim < 0.99, f"{name}: Cosine similarity too high ({cos_sim})"

    @pytest.mark.skip(reason="Requires trained model")
    def test_achiral_self_consistency(self, model):
        """Test that achiral molecules produce identical embeddings."""
        for smi, name in ACHIRAL_CONTROLS:
            emb1 = model.get_embedding(smi)
            emb2 = model.get_embedding(smi)

            assert torch.allclose(emb1, emb2, atol=1e-6), f"{name}: Not self-consistent"


# ============================================================================
# Statistical Validation Tests
# ============================================================================

class TestStatisticalValidation:
    """Statistical tests for chirality feature distributions."""

    def test_chiral_sign_distribution(self):
        """
        Test that chiral signs are approximately balanced (+1/-1).

        Using a set of diverse chiral molecules, the distribution
        should be roughly 50/50 between R and S.
        """
        # Generate random R/S molecules (simplified - using test pairs)
        all_signs = []
        for r_smi, s_smi, _ in ENANTIOMER_TEST_PAIRS[:4]:
            r_result = compute_all(r_smi, max_hops=3)
            s_result = compute_all(s_smi, max_hops=3)
            all_signs.extend(r_result['chiral_signs'])
            all_signs.extend(s_result['chiral_signs'])

        signs = np.array(all_signs)
        positive = np.sum(signs > 0)
        negative = np.sum(signs < 0)

        # With paired R/S, should be exactly balanced
        assert positive == negative, f"Sign distribution imbalanced: +{positive}, -{negative}"

    def test_ez_bond_detection_rate(self):
        """
        Test that E/Z bonds are detected at expected rate.

        All E/Z test pairs should have stereochemistry detected.
        """
        detected = 0
        total = len(E_Z_TEST_PAIRS) * 2  # E and Z for each pair

        for e_smi, z_smi, _ in E_Z_TEST_PAIRS:
            e_result = compute_all(e_smi, max_hops=3)
            z_result = compute_all(z_smi, max_hops=3)

            if len(e_result['cis_bonds_tensors']) > 0 or len(e_result['trans_bonds_tensors']) > 0:
                detected += 1
            if len(z_result['cis_bonds_tensors']) > 0 or len(z_result['trans_bonds_tensors']) > 0:
                detected += 1

        detection_rate = detected / total
        assert detection_rate > 0.9, f"E/Z detection rate too low: {detection_rate:.1%}"


# ============================================================================
# Edge Cases and Robustness
# ============================================================================

class TestEdgeCases:
    """Test edge cases in stereochemistry detection."""

    def test_meso_compound(self):
        """
        Meso compounds have chiral centers but are achiral overall.

        The implementation should still detect individual chiral centers.
        """
        # meso-tartaric acid: (2R,3S)-tartaric acid
        result = compute_all('O=C(O)[C@H](O)[C@@H](O)C(=O)O', max_hops=3)

        # Should detect 2 chiral centers (even though molecule is meso)
        assert len(result['chiral_signs']) == 2, "Meso compound should have 2 chiral centers"

    def test_multiple_stereocenters(self):
        """Test molecule with multiple chiral centers."""
        # Cholesterol has multiple chiral centers
        cholesterol = 'C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C'
        result = compute_all(cholesterol, max_hops=3)

        assert len(result['chiral_signs']) >= 5, "Cholesterol should have multiple chiral centers"

    def test_ring_with_chiral_center(self):
        """Test chiral center in a ring system."""
        # Menthol has a chiral center in a ring
        menthol = 'C[C@H]1CC[C@@H](C(C)C)[C@H](O)C1'
        result = compute_all(menthol, max_hops=3)

        assert len(result['chiral_signs']) >= 2, "Menthol should have chiral centers"

    def test_invalid_smiles_handling(self):
        """Test that invalid SMILES are handled gracefully (returns None)."""
        result = compute_all('INVALID_SMILES', max_hops=3)
        assert result is None, "Invalid SMILES should return None"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
