# Phase 3: Stereochemistry Validation Plan

**Date:** 2026-02-04
**Status:** Ready for implementation
**Prerequisite:** Phase 2 complete and verified

---

## Overview

Phase 3 validates the stereochemistry implementation through:
1. Synthetic enantiomer benchmark
2. Enantiomer discrimination testing
3. Property prediction evaluation
4. Ablation studies

---

## Batch 1: Enantiomer Discrimination Tests

### Task 1.1: Create Enantiomer Test Dataset

Create a comprehensive test dataset with R/S pairs:

```python
ENANTIOMER_TEST_PAIRS = [
    # Standard tetrahedral carbon
    ('N[C@@H](C)C(=O)O', 'N[C@H](C)C(=O)O', 'L-Alanine / D-Alanine'),
    ('C[C@@H](O)C(=O)O', 'C[C@H](O)C(=O)O', 'L-Lactic acid / D-Lactic acid'),
    ('O=C(O)[C@@H](N)CC(=O)O', 'O=C(O)[C@H](N)CC(=O)O', 'L-Aspartic acid / D-Aspartic acid'),

    # Sulfoxide (pyramidal heteroatom)
    ('C[S@@](=O)CC', 'C[S@](=O)CC', 'Ethyl methyl sulfoxide enantiomers'),
    ('C[S@@](=O)c1ccccc1', 'C[S@](=O)c1ccccc1', 'Methyl phenyl sulfoxide enantiomers'),

    # Selenoxide (pyramidal heteroatom)
    ('C[Se@@](=O)CC', 'C[Se@](=O)CC', 'Ethyl methyl selenoxide enantiomers'),

    # Phosphine oxide (pyramidal heteroatom)
    ('C[P@@](=O)(CC)c1ccccc1', 'C[P@](=O)(CC)c1ccccc1', 'Phosphine oxide enantiomers'),

    # Multiple chiral centers
    ('C[C@@H](O)[C@@H](O)C', 'C[C@H](O)[C@H](O)C', '2,3-Butanediol enantiomers'),
    ('N[C@@H](Cc1ccccc1)C(=O)O', 'N[C@H](Cc1ccccc1)C(=O)O', 'L-Phenylalanine / D-Phenylalanine'),
]

E_Z_TEST_PAIRS = [
    # E/Z isomers
    ('C/C=C/C', 'C/C=C\\C', 'trans-2-butene / cis-2-butene'),
    ('CC/C=C/CC', 'CC/C=C\\CC', 'trans-3-hexene / cis-3-hexene'),
    ('O=C(O)/C=C/c1ccccc1', 'O=C(O)/C=C\\c1ccccc1', 'trans-cinnamic acid / cis-cinnamic acid'),
]

ACHIRAL_CONTROLS = [
    ('CC(C)C', 'Isobutane - no chiral center'),
    ('c1ccccc1', 'Benzene - achiral'),
    ('CC(=O)O', 'Acetic acid - achiral'),
    ('CCCC', 'n-Butane - achiral'),
]
```

**File:** `tests/validation/test_enantiomer_discrimination.py`

### Task 1.2: Implement Embedding Extraction Test

```python
def test_enantiomer_embedding_difference():
    """
    Test that R and S enantiomers produce distinguishable embeddings.

    Metrics:
    - Embedding difference norm > threshold
    - Cosine similarity < 0.99 for enantiomers
    - Cosine similarity > 0.999 for same molecule
    """
    model = load_trained_model()

    for r_smi, s_smi, name in ENANTIOMER_TEST_PAIRS:
        r_emb = model.get_embedding(r_smi)
        s_emb = model.get_embedding(s_smi)

        # L2 norm difference
        diff_norm = torch.norm(r_emb - s_emb).item()

        # Cosine similarity
        cos_sim = F.cosine_similarity(r_emb, s_emb, dim=0).item()

        # Assert meaningful difference
        assert diff_norm > 0.1, f"{name}: diff_norm={diff_norm}"
        assert cos_sim < 0.99, f"{name}: cos_sim={cos_sim}"
```

### Task 1.3: Implement Self-Consistency Test

```python
def test_achiral_self_consistency():
    """
    Test that achiral molecules produce identical embeddings.
    """
    model = load_trained_model()

    for smi, name in ACHIRAL_CONTROLS:
        emb1 = model.get_embedding(smi)
        emb2 = model.get_embedding(smi)

        assert torch.allclose(emb1, emb2, atol=1e-6), f"{name}: not self-consistent"
```

**Verification:** Run tests with `pytest tests/validation/test_enantiomer_discrimination.py -v`

---

## Batch 2: Synthetic Benchmark Dataset

### Task 2.1: Create Enantiomer Property Prediction Dataset

Generate synthetic data where property depends on chirality:

```python
# Pseudo-code for dataset generation
def generate_chirality_dependent_dataset(n_samples=1000):
    """
    Generate molecules where a property directly depends on chirality.

    Property formula:
    - R enantiomer: property = base_value + 1.0
    - S enantiomer: property = base_value - 1.0
    - Achiral: property = base_value
    """
    data = []

    for _ in range(n_samples):
        scaffold = random_scaffold()  # Generate random molecular scaffold

        # Create R, S, and achiral variants
        r_smi = add_r_chiral_center(scaffold)
        s_smi = add_s_chiral_center(scaffold)
        achiral_smi = scaffold

        base_value = compute_base_property(scaffold)

        data.append({'smiles': r_smi, 'property': base_value + 1.0, 'chirality': 'R'})
        data.append({'smiles': s_smi, 'property': base_value - 1.0, 'chirality': 'S'})
        data.append({'smiles': achiral_smi, 'property': base_value, 'chirality': 'achiral'})

    return pd.DataFrame(data)
```

**File:** `scripts/generate_chirality_benchmark.py`

### Task 2.2: Train and Evaluate on Synthetic Benchmark

```bash
# Train with stereochemistry enabled
python main.py \
  --train_data data/chirality_benchmark_train.csv \
  --val_data data/chirality_benchmark_val.csv \
  --test_data data/chirality_benchmark_test.csv \
  --target_column property \
  --use_stereochemistry \
  --epochs 50 \
  --model_save_path models/stereo_benchmark.pth

# Train without stereochemistry (baseline)
python main.py \
  --train_data data/chirality_benchmark_train.csv \
  --val_data data/chirality_benchmark_val.csv \
  --test_data data/chirality_benchmark_test.csv \
  --target_column property \
  --epochs 50 \
  --model_save_path models/no_stereo_benchmark.pth
```

**Success Criteria:**
- Stereo model: MAE < 0.2 (can distinguish R/S)
- No-stereo model: MAE > 0.8 (cannot distinguish R/S)

---

## Batch 3: Ablation Studies

### Task 3.1: Tetrahedral Ablation

Disable tetrahedral module only:

```python
# In model config or forward pass
def forward_ablation_no_tet(self, ...):
    # Skip tetrahedral calculation
    f_tet = torch.zeros_like(atom_features)
    # Keep E/Z and allene
    f_ez = self._cis_trans_calculation(...)
    f_allene = self._allene_feature_calculation(...)
```

**Metrics to collect:**
- Enantiomer discrimination accuracy (should drop significantly)
- Property prediction MAE on stereo-dependent properties
- Property prediction MAE on stereo-independent properties (should stay same)

### Task 3.2: E/Z Ablation

Disable E/Z module only:

```python
def forward_ablation_no_ez(self, ...):
    f_tet = self._tetrahedral_feature_calculation_physics_inspired(...)
    # Skip E/Z
    f_ez = torch.zeros_like(atom_features)
    f_allene = self._allene_feature_calculation(...)
```

### Task 3.3: Allene Ablation

Disable allene module only:

```python
def forward_ablation_no_allene(self, ...):
    f_tet = self._tetrahedral_feature_calculation_physics_inspired(...)
    f_ez = self._cis_trans_calculation(...)
    # Skip allene
    f_allene = torch.zeros_like(atom_features)
```

### Task 3.4: Full Ablation (No Stereochemistry)

Disable all stereochemistry:

```bash
python main.py --train_data ... --use_stereochemistry false
```

**File:** `scripts/run_ablation_studies.py`

---

## Batch 4: Real-World Property Prediction

### Task 4.1: Identify Stereo-Dependent Properties

Properties known to depend on stereochemistry:
- Optical rotation
- Biological activity (e.g., drug potency)
- Binding affinity
- Taste/smell perception

**Datasets to consider:**
- ChEMBL bioactivity data with stereoisomers
- PubChem with optical rotation data
- MoleculeNet chirality subsets

### Task 4.2: Evaluate on ChEMBL Stereo Subset

```python
# Filter ChEMBL for molecules with defined stereochemistry
# and corresponding stereoisomer pairs with different activities

def find_stereoisomer_pairs_with_activity_diff(chembl_df, threshold=0.5):
    """
    Find pairs where stereoisomers have significantly different activities.
    """
    pairs = []

    for inchi_key_no_stereo, group in chembl_df.groupby('inchi_key_no_stereo'):
        if len(group) >= 2:
            # Check if activities differ
            activities = group['pIC50'].values
            if np.std(activities) > threshold:
                pairs.append(group)

    return pairs
```

**Success Criteria:**
- Model with stereo: can predict activity differences between stereoisomers
- Model without stereo: predicts same activity for stereoisomers

---

## Batch 5: Summary and Documentation

### Task 5.1: Compile Results

Create summary table:

| Benchmark | Stereo Model | No-Stereo Model | Improvement |
|-----------|--------------|-----------------|-------------|
| Enantiomer discrimination | X% | Y% | +Z% |
| E/Z discrimination | X% | Y% | +Z% |
| Synthetic benchmark MAE | X | Y | -Z |
| ChEMBL stereo subset | X | Y | +Z% |
| Ablation: no tet | X% | - | -Z% |
| Ablation: no E/Z | X% | - | -Z% |
| Ablation: no allene | X% | - | -Z% |

### Task 5.2: Update Documentation

- Add validation results to `docs/plans/`
- Update `CLAUDE.md` with benchmark instructions
- Document any limitations discovered

---

## Success Criteria

| Criterion | Target | Notes |
|-----------|--------|-------|
| Enantiomer discrimination | > 95% | R/S pairs distinguishable |
| E/Z classification | > 95% | E/Z pairs distinguishable |
| Sulfoxide enantiomers | > 90% | Pyramidal heteroatom test |
| Property improvement | > 10% | ChEMBL stereo-dependent |
| Ablation: no tet | Measurable drop | Quantify contribution |
| Ablation: no E/Z | Measurable drop | Quantify contribution |

---

## Files to Create

1. `tests/validation/test_enantiomer_discrimination.py` - Enantiomer tests
2. `tests/validation/test_ez_discrimination.py` - E/Z tests
3. `scripts/generate_chirality_benchmark.py` - Synthetic data generator
4. `scripts/run_ablation_studies.py` - Ablation runner
5. `docs/results/phase3-validation-results.md` - Results documentation

---

## Dependencies

- Trained model with stereochemistry enabled
- Test datasets (synthetic + real)
- Baseline model without stereochemistry

---

*Plan created: 2026-02-04*
*Status: Ready for implementation*
