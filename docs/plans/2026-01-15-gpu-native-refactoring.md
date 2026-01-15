# GPU-Native Architecture Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor AIMNet-X2D to maximize GPU utilization and batch-native processing, breaking all backwards compatibility to achieve 2-3x training speedup and 50% code reduction.

**Architecture:** Replace PyG Data objects with flat `MolecularGraphBatch` dataclass. Unify training/inference into single `Engine` class. Move SAE normalization to GPU. Eliminate per-molecule Python loops in favor of vectorized tensor operations.

**Tech Stack:** Python 3.12, PyTorch 2.5+, torch_scatter, Numba (CPU feature extraction only)

---

## Overview

| Phase | Description | Impact | Risk |
|-------|-------------|--------|------|
| 1 | Core Data Structures | Foundation for all changes | Medium |
| 2 | GPU-Native Preprocessing | 50-100x SAE speedup | Low |
| 3 | Batch-Native Feature Pipeline | 3-4x data loading speedup | Medium |
| 4 | Simplified Model | 30-50% forward pass speedup | Medium |
| 5 | Unified Engine | 50% code reduction | Low |
| 6 | Streamlined CLI | Clean API | Low |

**Breaking Changes:**
- PyG `Data` objects → `MolecularGraphBatch` dataclass
- `InferencePipeline`, `InferenceEngine`, `InferenceConfig` → unified `Engine`
- `PyGSMILESDataset`, `HDF5MolecularIterableDataset` → `MolecularDataLoader`
- 45 CLI parameters → 22 flat config options
- Model checkpoint format change (version 2.0)

---

## Phase 1: Core Data Structures

### Task 1.1: Create MolecularGraphBatch Dataclass

**Files:**
- Create: `src/core/__init__.py`
- Create: `src/core/batch.py`
- Create: `tests/core/test_batch.py`

**Step 1: Create core package**

Create `/home/olexandr/AIMNet-X2D/src/core/__init__.py`:

```python
"""
Core module for GPU-native molecular processing.

This module contains the fundamental data structures and operations
optimized for batch processing on GPU.
"""

from .batch import MolecularGraphBatch

__all__ = ["MolecularGraphBatch"]
```

**Step 2: Write failing test for MolecularGraphBatch**

Create `/home/olexandr/AIMNet-X2D/tests/core/__init__.py`:
```python
"""Core module tests."""
```

Create `/home/olexandr/AIMNet-X2D/tests/core/test_batch.py`:

```python
"""Tests for MolecularGraphBatch dataclass."""

import torch
import pytest
from src.core.batch import MolecularGraphBatch


class TestMolecularGraphBatch:
    """Test MolecularGraphBatch creation and operations."""

    def test_creation_minimal(self):
        """Test batch creation with minimal required fields."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8, 6, 7], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64),
            ptr=torch.tensor([0, 3, 5], dtype=torch.int64),
            num_molecules=2,
        )
        assert batch.num_molecules == 2
        assert batch.total_atoms == 5
        assert batch.atom_types.shape == (5,)

    def test_to_device(self):
        """Test moving batch to device."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            num_molecules=1,
        )
        # Move to CPU (always available)
        moved = batch.to(torch.device("cpu"))
        assert moved.atom_types.device.type == "cpu"

    def test_from_molecule_list(self):
        """Test creating batch from list of molecule data."""
        molecules = [
            {"atom_types": [6, 6, 8], "targets": [1.5]},
            {"atom_types": [6, 7], "targets": [2.0]},
        ]
        batch = MolecularGraphBatch.from_molecules(molecules)
        assert batch.num_molecules == 2
        assert batch.total_atoms == 5
        assert batch.targets.shape == (2, 1)

    def test_batch_slicing(self):
        """Test extracting single molecule from batch."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8, 6, 7], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64),
            ptr=torch.tensor([0, 3, 5], dtype=torch.int64),
            num_molecules=2,
            targets=torch.tensor([[1.0], [2.0]]),
        )
        mol0 = batch.get_molecule(0)
        assert mol0["atom_types"].tolist() == [6, 6, 8]
        assert mol0["target"].item() == 1.0
```

**Step 3: Run test to verify it fails**

```bash
python -m pytest tests/core/test_batch.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'src.core.batch'"

**Step 4: Implement MolecularGraphBatch**

Create `/home/olexandr/AIMNet-X2D/src/core/batch.py`:

```python
"""
GPU-native batch container for molecular graphs.

This replaces PyG Data objects with a flat, efficient structure
optimized for batch processing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np


@dataclass
class MolecularGraphBatch:
    """
    Batched molecular graph data optimized for GPU processing.

    All tensors are pre-stacked for efficient GPU operations.
    No per-molecule Python loops in forward pass.

    Attributes:
        atom_types: Atomic numbers [total_atoms]
        degrees: Atom degrees [total_atoms]
        hybridizations: Hybridization states [total_atoms]
        hydrogen_counts: Hydrogen counts [total_atoms]
        batch_idx: Molecule index for each atom [total_atoms]
        ptr: Cumulative atom counts [num_molecules + 1]
        edge_indices: Multi-hop edge indices list[Tensor[2, E_h]]
        targets: Target values [num_molecules, num_tasks]
        chiral_indices: Tetrahedral chiral atom indices [num_chiral, 4]
        cis_bond_indices: Cis bond pairs [num_cis, 4]
        trans_bond_indices: Trans bond pairs [num_trans, 4]
        total_charges: Molecular charges [num_molecules]
        smiles: Original SMILES strings
        num_molecules: Number of molecules in batch
    """

    # Required fields
    atom_types: torch.Tensor
    batch_idx: torch.Tensor
    ptr: torch.Tensor
    num_molecules: int

    # Optional atom features (populated during featurization)
    degrees: torch.Tensor | None = None
    hybridizations: torch.Tensor | None = None
    hydrogen_counts: torch.Tensor | None = None

    # Graph structure
    edge_indices: list[torch.Tensor] = field(default_factory=list)

    # Targets and metadata
    targets: torch.Tensor | None = None
    total_charges: torch.Tensor | None = None
    smiles: list[str] = field(default_factory=list)

    # Stereochemistry (pre-offset for batch)
    chiral_indices: torch.Tensor | None = None
    cis_bond_indices: torch.Tensor | None = None
    trans_bond_indices: torch.Tensor | None = None

    # Atomic numbers for SAE (may differ from atom_types for rare elements)
    atomic_numbers: torch.Tensor | None = None

    @property
    def total_atoms(self) -> int:
        """Total number of atoms across all molecules."""
        return self.atom_types.shape[0]

    @property
    def device(self) -> torch.device:
        """Device where tensors are stored."""
        return self.atom_types.device

    def to(self, device: torch.device) -> MolecularGraphBatch:
        """Move all tensors to specified device."""
        return MolecularGraphBatch(
            atom_types=self.atom_types.to(device),
            degrees=self.degrees.to(device) if self.degrees is not None else None,
            hybridizations=self.hybridizations.to(device) if self.hybridizations is not None else None,
            hydrogen_counts=self.hydrogen_counts.to(device) if self.hydrogen_counts is not None else None,
            batch_idx=self.batch_idx.to(device),
            ptr=self.ptr.to(device),
            edge_indices=[e.to(device) for e in self.edge_indices],
            targets=self.targets.to(device) if self.targets is not None else None,
            total_charges=self.total_charges.to(device) if self.total_charges is not None else None,
            smiles=self.smiles,  # Keep on CPU
            chiral_indices=self.chiral_indices.to(device) if self.chiral_indices is not None else None,
            cis_bond_indices=self.cis_bond_indices.to(device) if self.cis_bond_indices is not None else None,
            trans_bond_indices=self.trans_bond_indices.to(device) if self.trans_bond_indices is not None else None,
            atomic_numbers=self.atomic_numbers.to(device) if self.atomic_numbers is not None else None,
            num_molecules=self.num_molecules,
        )

    @classmethod
    def from_molecules(
        cls,
        molecules: list[dict[str, Any]],
        include_targets: bool = True,
    ) -> MolecularGraphBatch:
        """
        Create batch from list of molecule dictionaries.

        Args:
            molecules: List of dicts with 'atom_types', optionally 'targets', 'smiles', etc.
            include_targets: Whether to include target values

        Returns:
            MolecularGraphBatch with stacked tensors
        """
        if not molecules:
            raise ValueError("Cannot create batch from empty molecule list")

        # Compute sizes
        num_atoms_list = [len(m["atom_types"]) for m in molecules]
        ptr = torch.tensor([0] + list(np.cumsum(num_atoms_list)), dtype=torch.int64)
        total_atoms = ptr[-1].item()

        # Pre-allocate atom feature tensors
        atom_types = torch.zeros(total_atoms, dtype=torch.int32)
        batch_idx = torch.zeros(total_atoms, dtype=torch.int64)

        # Fill tensors
        for mol_idx, mol in enumerate(molecules):
            start, end = ptr[mol_idx].item(), ptr[mol_idx + 1].item()
            atom_types[start:end] = torch.tensor(mol["atom_types"], dtype=torch.int32)
            batch_idx[start:end] = mol_idx

        # Targets
        targets = None
        if include_targets and "targets" in molecules[0]:
            targets = torch.tensor(
                [m["targets"] for m in molecules],
                dtype=torch.float32
            )

        # SMILES
        smiles = [m.get("smiles", "") for m in molecules]

        return cls(
            atom_types=atom_types,
            batch_idx=batch_idx,
            ptr=ptr,
            num_molecules=len(molecules),
            targets=targets,
            smiles=smiles,
        )

    def get_molecule(self, idx: int) -> dict[str, Any]:
        """Extract single molecule data from batch."""
        if idx < 0 or idx >= self.num_molecules:
            raise IndexError(f"Molecule index {idx} out of range [0, {self.num_molecules})")

        start, end = self.ptr[idx].item(), self.ptr[idx + 1].item()

        result = {
            "atom_types": self.atom_types[start:end],
        }

        if self.targets is not None:
            result["target"] = self.targets[idx]

        if self.smiles:
            result["smiles"] = self.smiles[idx]

        return result

    def pin_memory(self) -> MolecularGraphBatch:
        """Pin tensors to memory for faster GPU transfer."""
        return MolecularGraphBatch(
            atom_types=self.atom_types.pin_memory(),
            degrees=self.degrees.pin_memory() if self.degrees is not None else None,
            hybridizations=self.hybridizations.pin_memory() if self.hybridizations is not None else None,
            hydrogen_counts=self.hydrogen_counts.pin_memory() if self.hydrogen_counts is not None else None,
            batch_idx=self.batch_idx.pin_memory(),
            ptr=self.ptr.pin_memory(),
            edge_indices=[e.pin_memory() for e in self.edge_indices],
            targets=self.targets.pin_memory() if self.targets is not None else None,
            total_charges=self.total_charges.pin_memory() if self.total_charges is not None else None,
            smiles=self.smiles,
            chiral_indices=self.chiral_indices.pin_memory() if self.chiral_indices is not None else None,
            cis_bond_indices=self.cis_bond_indices.pin_memory() if self.cis_bond_indices is not None else None,
            trans_bond_indices=self.trans_bond_indices.pin_memory() if self.trans_bond_indices is not None else None,
            atomic_numbers=self.atomic_numbers.pin_memory() if self.atomic_numbers is not None else None,
            num_molecules=self.num_molecules,
        )
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/core/test_batch.py -v
```

Expected: 4 tests PASS

**Step 6: Commit**

```bash
git add src/core/ tests/core/
git commit -m "feat: add MolecularGraphBatch dataclass for GPU-native batching"
```

---

### Task 1.2: Add Full Atom Features to MolecularGraphBatch

**Files:**
- Modify: `src/core/batch.py`
- Modify: `tests/core/test_batch.py`

**Step 1: Add test for full features**

Add to `/home/olexandr/AIMNet-X2D/tests/core/test_batch.py`:

```python
class TestMolecularGraphBatchFeatures:
    """Test batch with full molecular features."""

    def test_full_features_creation(self):
        """Test batch with all atom features."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8], dtype=torch.int32),
            degrees=torch.tensor([2, 3, 1], dtype=torch.int32),
            hybridizations=torch.tensor([2, 2, 3], dtype=torch.int32),
            hydrogen_counts=torch.tensor([2, 1, 0], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            num_molecules=1,
        )
        assert batch.degrees is not None
        assert batch.hybridizations is not None
        assert batch.hydrogen_counts is not None

    def test_edge_indices_multi_hop(self):
        """Test multi-hop edge indices."""
        # Simple 3-atom chain: 0-1-2
        hop1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64)
        hop2 = torch.tensor([[0, 2], [2, 0]], dtype=torch.int64)

        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 6], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            edge_indices=[hop1, hop2],
            num_molecules=1,
        )
        assert len(batch.edge_indices) == 2
        assert batch.edge_indices[0].shape == (2, 4)
        assert batch.edge_indices[1].shape == (2, 2)

    def test_stereochemistry_indices(self):
        """Test chiral and cis/trans indices."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 6, 6, 6], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 5], dtype=torch.int64),
            chiral_indices=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
            cis_bond_indices=torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            trans_bond_indices=torch.tensor([], dtype=torch.int64).reshape(0, 4),
            num_molecules=1,
        )
        assert batch.chiral_indices.shape == (1, 4)
        assert batch.cis_bond_indices.shape == (1, 4)
        assert batch.trans_bond_indices.shape == (0, 4)

    def test_atom_features_dict(self):
        """Test getting atom features as dict for model input."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 6, 8], dtype=torch.int32),
            degrees=torch.tensor([2, 3, 1], dtype=torch.int32),
            hybridizations=torch.tensor([2, 2, 3], dtype=torch.int32),
            hydrogen_counts=torch.tensor([2, 1, 0], dtype=torch.int32),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            num_molecules=1,
        )
        features = batch.atom_features_dict()
        assert "atom_type" in features
        assert "degree" in features
        assert "hybridization" in features
        assert "hydrogen_count" in features
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/core/test_batch.py::TestMolecularGraphBatchFeatures -v
```

Expected: FAIL with "AttributeError: 'MolecularGraphBatch' object has no attribute 'atom_features_dict'"

**Step 3: Add atom_features_dict method**

Add to `MolecularGraphBatch` class in `/home/olexandr/AIMNet-X2D/src/core/batch.py`:

```python
    def atom_features_dict(self) -> dict[str, torch.Tensor]:
        """
        Get atom features as dictionary for model embedding layers.

        Returns dict with keys matching embedding layer names.
        """
        features = {"atom_type": self.atom_types.long()}

        if self.degrees is not None:
            features["degree"] = self.degrees.long()
        if self.hybridizations is not None:
            features["hybridization"] = self.hybridizations.long()
        if self.hydrogen_counts is not None:
            features["hydrogen_count"] = self.hydrogen_counts.long()

        return features
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/core/test_batch.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/core/batch.py tests/core/test_batch.py
git commit -m "feat: add full atom features and stereochemistry to MolecularGraphBatch"
```

---

## Phase 2: GPU-Native Preprocessing

### Task 2.1: Create GPU-Native SAE Transform

**Files:**
- Create: `src/core/preprocessing.py`
- Create: `tests/core/test_preprocessing.py`

**Step 1: Write failing test**

Create `/home/olexandr/AIMNet-X2D/tests/core/test_preprocessing.py`:

```python
"""Tests for GPU-native preprocessing."""

import torch
import numpy as np
import pytest
from src.core.preprocessing import SAETransform


class TestSAETransform:
    """Test GPU-native SAE normalization."""

    def test_fit_from_atomic_numbers(self):
        """Test fitting SAE from atomic numbers and targets."""
        # Simple dataset: 2 molecules
        atomic_numbers = [
            [6, 6, 8],      # C-C-O
            [6, 6, 6, 7],   # C-C-C-N
        ]
        targets = np.array([[10.0], [15.0]])  # Single task

        transform = SAETransform.fit(atomic_numbers, targets, subtasks=[0])

        # Should have learned SAE values for C(6), O(8), N(7)
        assert 6 in transform.sae_dict
        assert 8 in transform.sae_dict
        assert 7 in transform.sae_dict

    def test_transform_gpu_native(self):
        """Test GPU-native SAE transformation."""
        transform = SAETransform(
            sae_dict={6: 1.0, 8: 2.0, 7: 1.5},
            subtasks=[0],
        )

        # Batch of 2 molecules
        atomic_numbers = torch.tensor([
            [6, 6, 8, 0],    # C-C-O (padded)
            [6, 6, 6, 7],    # C-C-C-N
        ], dtype=torch.int64)
        atom_counts = torch.tensor([3, 4], dtype=torch.int64)
        targets = torch.tensor([[10.0], [15.0]])

        # GPU transform (on CPU for test)
        transformed = transform.transform_batch(
            atomic_numbers, atom_counts, targets
        )

        # Expected: target - sum(sae for each atom)
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
            subtasks=[1, 2],  # Only apply to tasks 1 and 2
        )

        atomic_numbers = torch.tensor([[6, 6, 8]], dtype=torch.int64)
        atom_counts = torch.tensor([3], dtype=torch.int64)
        targets = torch.tensor([[5.0, 10.0, 20.0]])  # 3 tasks

        transformed = transform.transform_batch(
            atomic_numbers, atom_counts, targets
        )

        # Task 0 unchanged, tasks 1,2 normalized
        sae_shift = 1.0 + 1.0 + 2.0  # = 4.0
        expected = torch.tensor([[5.0, 10.0 - sae_shift, 20.0 - sae_shift]])
        assert torch.allclose(transformed, expected)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/core/test_preprocessing.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement GPU-native SAETransform**

Create `/home/olexandr/AIMNet-X2D/src/core/preprocessing.py`:

```python
"""
GPU-native preprocessing for molecular property prediction.

Key optimization: All transforms operate on batched tensors,
eliminating per-molecule Python loops.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np
from sklearn.linear_model import LinearRegression

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SAETransform:
    """
    GPU-native Self Atomic Energy (SAE) normalization.

    SAE removes molecular size dependence from extensive properties
    (energies, enthalpies) by subtracting sum of atomic contributions.

    Key optimization: Uses GPU lookup table instead of Python dict iteration.
    """

    sae_dict: dict[int, float]
    subtasks: list[int]
    max_atomic_num: int = 119  # Up to Oganesson

    # GPU lookup table (lazy initialized)
    _sae_lookup: torch.Tensor | None = field(default=None, repr=False)

    @classmethod
    def fit(
        cls,
        atomic_numbers_list: list[list[int]],
        targets: np.ndarray,
        subtasks: list[int],
    ) -> SAETransform:
        """
        Fit SAE coefficients using linear regression.

        Args:
            atomic_numbers_list: List of atomic numbers per molecule
            targets: Target values [num_molecules, num_tasks]
            subtasks: Which task indices to apply SAE to

        Returns:
            Fitted SAETransform
        """
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        # Count atoms per element per molecule
        unique_atoms = set()
        for nums in atomic_numbers_list:
            unique_atoms.update(nums)
        unique_atoms = sorted(unique_atoms)

        # Build count matrix [num_molecules, num_elements]
        atom_to_idx = {a: i for i, a in enumerate(unique_atoms)}
        count_matrix = np.zeros((len(atomic_numbers_list), len(unique_atoms)))

        for mol_idx, nums in enumerate(atomic_numbers_list):
            for atom_num in nums:
                count_matrix[mol_idx, atom_to_idx[atom_num]] += 1

        # Fit linear regression for each subtask
        sae_dict: dict[int, float] = {}

        for subtask_idx in subtasks:
            y = targets[:, subtask_idx]
            reg = LinearRegression(fit_intercept=False)
            reg.fit(count_matrix, y)

            # Average coefficients across subtasks for each element
            for atom_num, coef_idx in atom_to_idx.items():
                if atom_num not in sae_dict:
                    sae_dict[atom_num] = 0.0
                sae_dict[atom_num] += reg.coef_[coef_idx] / len(subtasks)

        logger.info(f"Fitted SAE for {len(sae_dict)} elements on subtasks {subtasks}")

        return cls(sae_dict=sae_dict, subtasks=subtasks)

    def _get_sae_lookup(self, device: torch.device) -> torch.Tensor:
        """Get or create GPU lookup table."""
        if self._sae_lookup is None or self._sae_lookup.device != device:
            lookup = torch.zeros(self.max_atomic_num, dtype=torch.float32)
            for atom_num, sae_val in self.sae_dict.items():
                if 0 <= atom_num < self.max_atomic_num:
                    lookup[atom_num] = sae_val
            self._sae_lookup = lookup.to(device)
        return self._sae_lookup

    def transform_batch(
        self,
        atomic_numbers: torch.Tensor,
        atom_counts: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply SAE normalization to batch (GPU-native).

        Args:
            atomic_numbers: Padded atomic numbers [batch_size, max_atoms]
            atom_counts: Actual atom counts per molecule [batch_size]
            targets: Target values [batch_size, num_tasks]

        Returns:
            Normalized targets [batch_size, num_tasks]
        """
        device = targets.device
        batch_size, max_atoms = atomic_numbers.shape

        # Get lookup table
        sae_lookup = self._get_sae_lookup(device)

        # Create mask for valid atoms
        atom_indices = torch.arange(max_atoms, device=device).unsqueeze(0)
        valid_mask = atom_indices < atom_counts.unsqueeze(1)

        # Lookup SAE values and mask invalid
        sae_values = sae_lookup[atomic_numbers.clamp(0, self.max_atomic_num - 1)]
        sae_values = sae_values * valid_mask.float()

        # Sum SAE per molecule
        sae_shifts = sae_values.sum(dim=1, keepdim=True)  # [batch_size, 1]

        # Apply to subtasks only
        result = targets.clone()
        for subtask_idx in self.subtasks:
            if subtask_idx < result.shape[1]:
                result[:, subtask_idx] = result[:, subtask_idx] - sae_shifts.squeeze(1)

        return result

    def inverse_transform_batch(
        self,
        atomic_numbers: torch.Tensor,
        atom_counts: torch.Tensor,
        normalized: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inverse SAE transformation (GPU-native).

        Args:
            atomic_numbers: Padded atomic numbers [batch_size, max_atoms]
            atom_counts: Actual atom counts per molecule [batch_size]
            normalized: Normalized values [batch_size, num_tasks]

        Returns:
            Original scale values [batch_size, num_tasks]
        """
        device = normalized.device
        batch_size, max_atoms = atomic_numbers.shape

        sae_lookup = self._get_sae_lookup(device)

        atom_indices = torch.arange(max_atoms, device=device).unsqueeze(0)
        valid_mask = atom_indices < atom_counts.unsqueeze(1)

        sae_values = sae_lookup[atomic_numbers.clamp(0, self.max_atomic_num - 1)]
        sae_values = sae_values * valid_mask.float()
        sae_shifts = sae_values.sum(dim=1, keepdim=True)

        result = normalized.clone()
        for subtask_idx in self.subtasks:
            if subtask_idx < result.shape[1]:
                result[:, subtask_idx] = result[:, subtask_idx] + sae_shifts.squeeze(1)

        return result

    def state_dict(self) -> dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "sae_dict": self.sae_dict,
            "subtasks": self.subtasks,
            "max_atomic_num": self.max_atomic_num,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> SAETransform:
        """Deserialize from checkpoint."""
        return cls(
            sae_dict=state["sae_dict"],
            subtasks=state["subtasks"],
            max_atomic_num=state.get("max_atomic_num", 119),
        )
```

**Step 4: Update core __init__.py**

```python
"""
Core module for GPU-native molecular processing.
"""

from .batch import MolecularGraphBatch
from .preprocessing import SAETransform

__all__ = ["MolecularGraphBatch", "SAETransform"]
```

**Step 5: Run tests**

```bash
python -m pytest tests/core/test_preprocessing.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/core/preprocessing.py src/core/__init__.py tests/core/test_preprocessing.py
git commit -m "feat: add GPU-native SAETransform with batch operations"
```

---

### Task 2.2: Add StandardScaler to Preprocessing

**Files:**
- Modify: `src/core/preprocessing.py`
- Modify: `tests/core/test_preprocessing.py`

**Step 1: Add test for StandardScaler**

Add to `/home/olexandr/AIMNet-X2D/tests/core/test_preprocessing.py`:

```python
from src.core.preprocessing import SAETransform, StandardScaler, PreprocessingPipeline


class TestStandardScaler:
    """Test GPU-native standard scaling."""

    def test_fit_transform(self):
        """Test fitting and transforming."""
        targets = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

        scaler = StandardScaler.fit(targets)
        transformed = scaler.transform_batch(targets)

        # Should be zero mean, unit variance
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

        # Transform with batch tensors
        atomic_nums_tensor = torch.tensor([
            [6, 6, 8, 0],
            [6, 6, 6, 7],
        ], dtype=torch.int64)
        atom_counts = torch.tensor([3, 4], dtype=torch.int64)
        targets_tensor = torch.tensor([[100.0], [150.0]])

        transformed = pipeline.transform_batch(
            atomic_nums_tensor, atom_counts, targets_tensor
        )

        # Should be normalized
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
```

**Step 2: Run tests to verify failure**

```bash
python -m pytest tests/core/test_preprocessing.py::TestStandardScaler -v
```

Expected: FAIL

**Step 3: Implement StandardScaler and PreprocessingPipeline**

Add to `/home/olexandr/AIMNet-X2D/src/core/preprocessing.py`:

```python
@dataclass
class StandardScaler:
    """
    GPU-native standard scaling (zero mean, unit variance).
    """

    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-8

    @classmethod
    def fit(cls, targets: torch.Tensor, eps: float = 1e-8) -> StandardScaler:
        """Fit scaler to targets."""
        mean = targets.mean(dim=0)
        std = targets.std(dim=0, unbiased=False)
        std = torch.clamp(std, min=eps)
        return cls(mean=mean, std=std, eps=eps)

    def transform_batch(self, targets: torch.Tensor) -> torch.Tensor:
        """Apply scaling."""
        mean = self.mean.to(targets.device)
        std = self.std.to(targets.device)
        return (targets - mean) / std

    def inverse_transform_batch(self, normalized: torch.Tensor) -> torch.Tensor:
        """Inverse scaling."""
        mean = self.mean.to(normalized.device)
        std = self.std.to(normalized.device)
        return normalized * std + mean

    def state_dict(self) -> dict[str, Any]:
        """Serialize."""
        return {
            "mean": self.mean.cpu().numpy().tolist(),
            "std": self.std.cpu().numpy().tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> StandardScaler:
        """Deserialize."""
        return cls(
            mean=torch.tensor(state["mean"]),
            std=torch.tensor(state["std"]),
            eps=state.get("eps", 1e-8),
        )


@dataclass
class PreprocessingPipeline:
    """
    Combined preprocessing pipeline: SAE → Scaling.

    All operations are GPU-native batch operations.
    """

    sae_transform: SAETransform | None = None
    scaler: StandardScaler | None = None

    @classmethod
    def fit(
        cls,
        atomic_numbers_list: list[list[int]],
        targets: np.ndarray,
        apply_sae: bool = False,
        sae_subtasks: list[int] | None = None,
        apply_scaling: bool = True,
    ) -> PreprocessingPipeline:
        """
        Fit preprocessing pipeline.

        Order: SAE first (removes size dependence), then scaling.
        """
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        targets_tensor = torch.from_numpy(targets).float()

        # Fit SAE
        sae_transform = None
        if apply_sae and sae_subtasks:
            sae_transform = SAETransform.fit(
                atomic_numbers_list, targets, sae_subtasks
            )
            # Apply SAE to targets for scaler fitting
            max_atoms = max(len(nums) for nums in atomic_numbers_list)
            atomic_nums_padded = torch.zeros(len(atomic_numbers_list), max_atoms, dtype=torch.int64)
            atom_counts = torch.zeros(len(atomic_numbers_list), dtype=torch.int64)

            for i, nums in enumerate(atomic_numbers_list):
                atomic_nums_padded[i, :len(nums)] = torch.tensor(nums)
                atom_counts[i] = len(nums)

            targets_tensor = sae_transform.transform_batch(
                atomic_nums_padded, atom_counts, targets_tensor
            )

        # Fit scaler on SAE-normalized data
        scaler = None
        if apply_scaling:
            scaler = StandardScaler.fit(targets_tensor)

        return cls(sae_transform=sae_transform, scaler=scaler)

    def transform_batch(
        self,
        atomic_numbers: torch.Tensor,
        atom_counts: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply full pipeline."""
        result = targets

        if self.sae_transform is not None:
            result = self.sae_transform.transform_batch(
                atomic_numbers, atom_counts, result
            )

        if self.scaler is not None:
            result = self.scaler.transform_batch(result)

        return result

    def inverse_transform_batch(
        self,
        atomic_numbers: torch.Tensor,
        atom_counts: torch.Tensor,
        normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse full pipeline (reverse order)."""
        result = normalized

        if self.scaler is not None:
            result = self.scaler.inverse_transform_batch(result)

        if self.sae_transform is not None:
            result = self.sae_transform.inverse_transform_batch(
                atomic_numbers, atom_counts, result
            )

        return result

    def state_dict(self) -> dict[str, Any]:
        """Serialize."""
        return {
            "sae_transform": self.sae_transform.state_dict() if self.sae_transform else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> PreprocessingPipeline:
        """Deserialize."""
        return cls(
            sae_transform=SAETransform.from_state_dict(state["sae_transform"]) if state.get("sae_transform") else None,
            scaler=StandardScaler.from_state_dict(state["scaler"]) if state.get("scaler") else None,
        )
```

**Step 4: Update exports**

```python
from .preprocessing import SAETransform, StandardScaler, PreprocessingPipeline

__all__ = ["MolecularGraphBatch", "SAETransform", "StandardScaler", "PreprocessingPipeline"]
```

**Step 5: Run tests**

```bash
python -m pytest tests/core/test_preprocessing.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/core/preprocessing.py src/core/__init__.py tests/core/test_preprocessing.py
git commit -m "feat: add StandardScaler and PreprocessingPipeline for GPU-native preprocessing"
```

---

## Phase 3: Batch-Native Feature Pipeline

### Task 3.1: Create BatchFeaturizer

**Files:**
- Create: `src/core/featurizer.py`
- Create: `tests/core/test_featurizer.py`

**Step 1: Write failing test**

Create `/home/olexandr/AIMNet-X2D/tests/core/test_featurizer.py`:

```python
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
        assert batch.total_atoms == 9  # With hydrogens: C(4H) + C(2H) + O(1H)

    def test_batch_molecules(self):
        """Test featurizing batch of molecules."""
        featurizer = BatchFeaturizer(num_hops=3)

        smiles = ["C", "CC", "CCC"]  # Methane, ethane, propane
        targets = np.array([[1.0], [2.0], [3.0]])

        batch = featurizer.featurize(smiles, targets)

        assert batch.num_molecules == 3
        assert batch.targets.shape == (3, 1)
        assert len(batch.smiles) == 3

    def test_edge_indices_generated(self):
        """Test multi-hop edge indices are generated."""
        featurizer = BatchFeaturizer(num_hops=2)

        smiles = ["CC"]  # Ethane
        targets = np.array([[1.0]])

        batch = featurizer.featurize(smiles, targets)

        assert len(batch.edge_indices) == 2  # 2 hops
        assert all(e.shape[0] == 2 for e in batch.edge_indices)  # COO format

    def test_invalid_smiles_filtered(self):
        """Test invalid SMILES are filtered."""
        featurizer = BatchFeaturizer(num_hops=2)

        smiles = ["C", "INVALID_SMILES", "CC"]
        targets = np.array([[1.0], [2.0], [3.0]])

        batch = featurizer.featurize(smiles, targets)

        # Invalid filtered out
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

        # Larger batch to test parallelism
        smiles = ["C" * i for i in range(1, 11)]  # C, CC, CCC, ...
        targets = np.arange(10).reshape(-1, 1).astype(np.float32)

        batch = featurizer.featurize(smiles, targets)

        assert batch.num_molecules == 10
```

**Step 2: Run tests**

```bash
python -m pytest tests/core/test_featurizer.py -v
```

Expected: FAIL

**Step 3: Implement BatchFeaturizer**

Create `/home/olexandr/AIMNet-X2D/src/core/featurizer.py`:

```python
"""
Batch-native molecular featurization.

Optimized for batch processing with minimal per-molecule overhead.
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from .batch import MolecularGraphBatch
from datasets.features import compute_multi_hop_edges_bfs_numba
from datasets.constants import (
    ATOM_TYPES, HYBRIDIZATION_TYPES, MAX_HYDROGEN_COUNT, MAX_DEGREE
)
from utils.logging import get_logger

logger = get_logger(__name__)


def _featurize_single(args: tuple[str, int]) -> dict[str, Any] | None:
    """
    Featurize single molecule (for parallel processing).

    Returns dict with numpy arrays, or None if invalid.
    """
    smiles, num_hops = args

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        num_atoms = mol.GetNumAtoms()

        # Atom features
        atom_types = np.zeros(num_atoms, dtype=np.int32)
        degrees = np.zeros(num_atoms, dtype=np.int32)
        hybridizations = np.zeros(num_atoms, dtype=np.int32)
        hydrogen_counts = np.zeros(num_atoms, dtype=np.int32)
        atomic_numbers = np.zeros(num_atoms, dtype=np.int32)

        for i, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            atomic_numbers[i] = atomic_num

            # Map to index
            if atomic_num in ATOM_TYPES:
                atom_types[i] = ATOM_TYPES.index(atomic_num)
            else:
                atom_types[i] = len(ATOM_TYPES)  # Unknown

            degrees[i] = min(atom.GetTotalDegree(), MAX_DEGREE - 1)

            hyb = atom.GetHybridization()
            hyb_str = str(hyb)
            if hyb_str in HYBRIDIZATION_TYPES:
                hybridizations[i] = HYBRIDIZATION_TYPES.index(hyb_str)
            else:
                hybridizations[i] = len(HYBRIDIZATION_TYPES) - 1

            hydrogen_counts[i] = min(atom.GetTotalNumHs(), MAX_HYDROGEN_COUNT - 1)

        # Build adjacency for BFS
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        adj_list = [np.where(adj_matrix[i] > 0)[0].astype(np.int32) for i in range(num_atoms)]

        # Compute multi-hop edges
        multi_hop_edges = compute_multi_hop_edges_bfs_numba(adj_list, num_hops)

        # Stereochemistry
        chiral_centers = []
        cis_bonds = []
        trans_bonds = []

        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

            # Tetrahedral centers
            chiral_info = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            for atom_idx, chirality in chiral_info:
                neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]
                if len(neighbors) >= 4:
                    chiral_centers.append([atom_idx] + neighbors[:3])

            # Cis/trans bonds
            for bond in mol.GetBonds():
                stereo = bond.GetStereo()
                if stereo == Chem.BondStereo.STEREOZ:
                    cis_bonds.append([
                        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    ])
                elif stereo == Chem.BondStereo.STEREOE:
                    trans_bonds.append([
                        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    ])
        except Exception:
            pass  # Stereo assignment can fail for some molecules

        return {
            "smiles": smiles,
            "atom_types": atom_types,
            "degrees": degrees,
            "hybridizations": hybridizations,
            "hydrogen_counts": hydrogen_counts,
            "atomic_numbers": atomic_numbers,
            "multi_hop_edges": multi_hop_edges,
            "chiral_centers": chiral_centers,
            "cis_bonds": cis_bonds,
            "trans_bonds": trans_bonds,
            "total_charge": Chem.GetFormalCharge(mol),
        }

    except Exception as e:
        logger.debug(f"Failed to featurize {smiles[:20]}...: {e}")
        return None


@dataclass
class BatchFeaturizer:
    """
    Batch-native molecular featurizer.

    Converts SMILES to MolecularGraphBatch with GPU-ready tensors.
    """

    num_hops: int = 3
    num_workers: int = 4

    def featurize(
        self,
        smiles_list: list[str],
        targets: np.ndarray,
    ) -> MolecularGraphBatch:
        """
        Featurize batch of molecules.

        Args:
            smiles_list: SMILES strings
            targets: Target values [num_molecules, num_tasks]

        Returns:
            MolecularGraphBatch with all features
        """
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        # Parallel featurization
        args = [(smi, self.num_hops) for smi in smiles_list]

        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                results = list(pool.map(_featurize_single, args))
        else:
            results = [_featurize_single(a) for a in args]

        # Filter valid and collect
        valid_data = []
        valid_targets = []

        for i, result in enumerate(results):
            if result is not None:
                valid_data.append(result)
                valid_targets.append(targets[i])

        if not valid_data:
            raise ValueError("No valid molecules in batch")

        # Stack into batch
        return self._stack_batch(valid_data, np.array(valid_targets))

    def _stack_batch(
        self,
        molecules: list[dict[str, Any]],
        targets: np.ndarray,
    ) -> MolecularGraphBatch:
        """Stack molecule data into batch tensors."""
        num_molecules = len(molecules)

        # Compute atom counts and offsets
        atom_counts = [len(m["atom_types"]) for m in molecules]
        ptr = torch.tensor([0] + list(np.cumsum(atom_counts)), dtype=torch.int64)
        total_atoms = ptr[-1].item()

        # Pre-allocate tensors
        atom_types = torch.zeros(total_atoms, dtype=torch.int32)
        degrees = torch.zeros(total_atoms, dtype=torch.int32)
        hybridizations = torch.zeros(total_atoms, dtype=torch.int32)
        hydrogen_counts = torch.zeros(total_atoms, dtype=torch.int32)
        atomic_numbers = torch.zeros(total_atoms, dtype=torch.int64)
        batch_idx = torch.zeros(total_atoms, dtype=torch.int64)

        # Collect edges with offsets
        hop_edges_lists: list[list[torch.Tensor]] = [[] for _ in range(self.num_hops)]
        chiral_indices_list = []
        cis_indices_list = []
        trans_indices_list = []

        total_charges = []
        smiles_list = []

        # Fill tensors
        for mol_idx, mol in enumerate(molecules):
            start = ptr[mol_idx].item()
            end = ptr[mol_idx + 1].item()
            offset = start

            # Atom features
            atom_types[start:end] = torch.from_numpy(mol["atom_types"])
            degrees[start:end] = torch.from_numpy(mol["degrees"])
            hybridizations[start:end] = torch.from_numpy(mol["hybridizations"])
            hydrogen_counts[start:end] = torch.from_numpy(mol["hydrogen_counts"])
            atomic_numbers[start:end] = torch.from_numpy(mol["atomic_numbers"])
            batch_idx[start:end] = mol_idx

            # Multi-hop edges with offset
            for hop_idx, edges in enumerate(mol["multi_hop_edges"]):
                if edges.size > 0:
                    edges_tensor = torch.from_numpy(edges).long() + offset
                    hop_edges_lists[hop_idx].append(edges_tensor)

            # Stereochemistry with offset
            for chiral in mol["chiral_centers"]:
                chiral_indices_list.append([c + offset for c in chiral])
            for cis in mol["cis_bonds"]:
                cis_indices_list.append([c + offset for c in cis])
            for trans in mol["trans_bonds"]:
                trans_indices_list.append([c + offset for c in trans])

            total_charges.append(mol["total_charge"])
            smiles_list.append(mol["smiles"])

        # Concatenate edges
        edge_indices = []
        for hop_edges in hop_edges_lists:
            if hop_edges:
                edge_indices.append(torch.cat(hop_edges, dim=1))
            else:
                edge_indices.append(torch.zeros((2, 0), dtype=torch.int64))

        # Build stereo tensors
        chiral_indices = torch.tensor(chiral_indices_list, dtype=torch.int64) if chiral_indices_list else torch.zeros((0, 4), dtype=torch.int64)
        cis_indices = torch.tensor(cis_indices_list, dtype=torch.int64) if cis_indices_list else torch.zeros((0, 4), dtype=torch.int64)
        trans_indices = torch.tensor(trans_indices_list, dtype=torch.int64) if trans_indices_list else torch.zeros((0, 4), dtype=torch.int64)

        return MolecularGraphBatch(
            atom_types=atom_types,
            degrees=degrees,
            hybridizations=hybridizations,
            hydrogen_counts=hydrogen_counts,
            atomic_numbers=atomic_numbers,
            batch_idx=batch_idx,
            ptr=ptr,
            edge_indices=edge_indices,
            targets=torch.from_numpy(targets).float(),
            total_charges=torch.tensor(total_charges, dtype=torch.float32),
            smiles=smiles_list,
            chiral_indices=chiral_indices,
            cis_bond_indices=cis_indices,
            trans_bond_indices=trans_indices,
            num_molecules=num_molecules,
        )
```

**Step 4: Update exports**

```python
from .featurizer import BatchFeaturizer

__all__ = ["MolecularGraphBatch", "SAETransform", "StandardScaler", "PreprocessingPipeline", "BatchFeaturizer"]
```

**Step 5: Run tests**

```bash
python -m pytest tests/core/test_featurizer.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/core/featurizer.py src/core/__init__.py tests/core/test_featurizer.py
git commit -m "feat: add BatchFeaturizer for parallel batch-native featurization"
```

---

## Phase 4: Simplified Model (Summary)

**Tasks 4.1-4.3**: Refactor GNN model to use `MolecularGraphBatch` directly.

Key changes:
- Replace 22 constructor parameters with `ModelConfig` dataclass
- Remove optional feature flags (stereochemistry always computed)
- Fused scatter operations in message passing
- Remove tensor clones in stereochemistry calculation

**Files to create:**
- `src/core/model.py` - Simplified GNN
- `src/core/layers.py` - Streamlined ShellConvBlock
- `tests/core/test_model.py`

---

## Phase 5: Unified Engine (Summary)

**Tasks 5.1-5.3**: Merge training and inference into single `Engine` class.

Key changes:
- Single `Engine.train()` and `Engine.predict()` interface
- Remove `InferencePipeline`, `InferenceEngine`, `InferenceConfig`
- Shared forward pass logic
- GPU-native preprocessing integration

**Files to create:**
- `src/core/engine.py` - Unified training/inference
- `tests/core/test_engine.py`

**Files to delete:**
- `src/inference/config.py`
- `src/inference/engine.py`
- `src/inference/pipeline.py`
- `src/inference/results_writer.py`

---

## Phase 6: Streamlined CLI (Summary)

**Tasks 6.1-6.2**: Flatten configuration and simplify entry point.

Key changes:
- Replace 45 CLI args with 22 flat config
- Single `main.py` entry point
- Remove `runner.py` orchestration complexity

**New CLI:**
```bash
# Training
aimnet train --data train.csv --val val.csv --epochs 100 --output model.pth

# Inference
aimnet predict --model model.pth --input molecules.csv --output predictions.csv
```

---

## Verification Commands

```bash
# Run all new tests
python -m pytest tests/core/ -v

# Verify no regressions
python -m pytest tests/ -v

# Check code reduction
find src/core -name "*.py" | xargs wc -l
find src/inference -name "*.py" | xargs wc -l

# Benchmark comparison (after full implementation)
python benchmark_old_vs_new.py
```

---

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Training throughput (mol/s) | ~3,000 | ~8,000 | 2.7x |
| Inference throughput (mol/s) | ~8,000 | ~25,000 | 3x |
| GPU utilization | 45-55% | 75-85% | 1.5x |
| Lines of code | ~5,500 | ~2,800 | 49% reduction |
| Config parameters | 45 | 22 | 51% reduction |

---

## Migration Notes

**Breaking changes:**
1. Model checkpoint format v2.0 (incompatible with v1.x)
2. CLI arguments completely restructured
3. Python API changed (new `Engine` class)
4. HDF5 format changed (pre-batched tensors)

**No migration path** - this is a clean break. Users should:
1. Retrain models with v2.0
2. Update scripts to new CLI
3. Regenerate HDF5 files
