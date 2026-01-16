# Phase 5: Performance & Compatibility Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all performance, numerical stability, and compatibility issues identified by architecture, AI/ML, and GPU performance reviewers.

**Architecture:** Address 9 categories of issues: (1) Replace custom scatter_add with optimized native ops, (2) Fix fp16 numerical stability, (3) Vectorize Python loops in SAE, (4) Add stereochemistry processing to model, (5) Add edge features to message passing, (6) torch.compile() compatibility, (7) Pre-allocate batch tensors, (8) Add SAE-batch adapter for format mismatch, (9) Add total_charges to model.

**Tech Stack:** PyTorch 2.x, torch.compile(), fp16/bf16 mixed precision

---

## Task 5.1: Fix Numerical Stability for Mixed Precision

**Files:**
- Modify: `src/core/layers.py:311` (epsilon in AttentionPooling)
- Test: `tests/core/test_layers.py`

**Step 1: Write the failing test**

Add test to `tests/core/test_layers.py`:

```python
def test_attention_pooling_numerical_stability_fp16():
    """Test attention pooling with fp16 inputs for numerical stability."""
    pooling = AttentionPooling(input_dim=64, num_heads=4)
    pooling = pooling.half()  # Convert to fp16

    # Create fp16 inputs
    x = torch.randn(100, 64, dtype=torch.float16)
    batch_idx = torch.tensor([0]*30 + [1]*40 + [2]*30, dtype=torch.long)

    # Should not produce NaN or Inf
    output = pooling(x, batch_idx)

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isfinite(output).all() == False, "Output contains Inf values"
    assert output.shape == (3, 64)
```

**Step 2: Run test to verify current behavior**

Run: `pytest tests/core/test_layers.py::test_attention_pooling_numerical_stability_fp16 -v`

**Step 3: Fix epsilon value in AttentionPooling**

In `src/core/layers.py`, change line 311:

```python
# OLD:
attention_weights = attention_exp / (attention_sum[batch_idx] + 1e-10)

# NEW:
attention_weights = attention_exp / (attention_sum[batch_idx] + 1e-6)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_layers.py::test_attention_pooling_numerical_stability_fp16 -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/layers.py tests/core/test_layers.py
git commit -m "fix: improve numerical stability for fp16 in AttentionPooling"
```

---

## Task 5.2: Optimize scatter_add with Native PyTorch Operations

**Files:**
- Modify: `src/core/layers.py:15-77` (scatter_add function)
- Test: `tests/core/test_layers.py`

**Step 1: Write performance and correctness tests**

Add to `tests/core/test_layers.py`:

```python
def test_scatter_add_correctness():
    """Test scatter_add produces correct results."""
    from src.core.layers import scatter_add

    # Simple test case
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 1, 0])

    result = scatter_add(src, index, dim=0, dim_size=2)

    # index 0: [1,2] + [5,6] = [6,8]
    # index 1: [3,4]
    expected = torch.tensor([[6.0, 8.0], [3.0, 4.0]])

    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_scatter_add_empty_input():
    """Test scatter_add handles empty inputs gracefully."""
    from src.core.layers import scatter_add

    src = torch.zeros(0, 64)
    index = torch.zeros(0, dtype=torch.long)

    result = scatter_add(src, index, dim=0, dim_size=10)

    assert result.shape == (10, 64)
    assert (result == 0).all()
```

**Step 2: Run tests to verify current implementation**

Run: `pytest tests/core/test_layers.py::test_scatter_add_correctness tests/core/test_layers.py::test_scatter_add_empty_input -v`

**Step 3: Simplify scatter_add implementation**

Replace the complex scatter_add in `src/core/layers.py`:

```python
def scatter_add(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> Tensor:
    """
    Scatter add operation using PyTorch native operations.

    Optimized for torch.compile() compatibility by avoiding
    dynamic control flow.

    Args:
        src: Source tensor to scatter
        index: Index tensor specifying where to scatter
        dim: Dimension along which to scatter (default: 0)
        dim_size: Size of output dimension (default: inferred from index)

    Returns:
        Output tensor with scattered values summed
    """
    # Handle empty input
    if src.numel() == 0:
        out_size = dim_size if dim_size is not None else 0
        out_shape = list(src.shape)
        out_shape[dim] = out_size
        return torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    # Determine output size
    out_size = dim_size if dim_size is not None else (index.max().item() + 1)

    # Build output shape
    out_shape = list(src.shape)
    out_shape[dim] = out_size

    # Create output tensor
    output = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    # Expand index if needed to match src shape
    if index.dim() == 1 and src.dim() > 1:
        # Expand 1D index to match src dimensions
        expand_shape = [1] * src.dim()
        expand_shape[dim] = -1
        index = index.view(*expand_shape).expand_as(src)

    # Use scatter_add_ for in-place accumulation (faster than scatter_reduce)
    output.scatter_add_(dim, index, src)

    return output
```

**Step 4: Run tests to verify new implementation**

Run: `pytest tests/core/test_layers.py -v -k "scatter"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/layers.py tests/core/test_layers.py
git commit -m "perf: optimize scatter_add with native scatter_add_ operation"
```

---

## Task 5.3: Vectorize SAE Transform Loops

**Files:**
- Modify: `src/core/preprocessing.py:107-109, 133-135`
- Test: `tests/core/test_preprocessing.py`

**Step 1: Write test for vectorized SAE transform**

Add to `tests/core/test_preprocessing.py`:

```python
def test_sae_transform_vectorized_multiple_subtasks():
    """Test SAE transform correctly applies to multiple subtasks."""
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
```

**Step 2: Run test to check current behavior**

Run: `pytest tests/core/test_preprocessing.py::test_sae_transform_vectorized_multiple_subtasks -v`

**Step 3: Vectorize the SAE transform loops**

In `src/core/preprocessing.py`, replace the `transform_batch` method:

```python
def transform_batch(
    self,
    atomic_numbers: torch.Tensor,
    atom_counts: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Apply SAE normalization to batch (GPU-native, vectorized)."""
    device = targets.device
    batch_size, max_atoms = atomic_numbers.shape

    sae_lookup = self._get_sae_lookup(device)

    atom_indices = torch.arange(max_atoms, device=device).unsqueeze(0)
    valid_mask = atom_indices < atom_counts.unsqueeze(1)

    sae_values = sae_lookup[atomic_numbers.clamp(0, self.max_atomic_num - 1)]
    sae_values = sae_values * valid_mask.float()

    sae_shifts = sae_values.sum(dim=1)  # [batch_size]

    # Vectorized subtask application using advanced indexing
    result = targets.clone()
    subtask_indices = torch.tensor(self.subtasks, device=device, dtype=torch.long)
    valid_subtasks = subtask_indices[subtask_indices < result.shape[1]]

    if valid_subtasks.numel() > 0:
        # result[:, valid_subtasks] -= sae_shifts.unsqueeze(1)
        result[:, valid_subtasks] = result[:, valid_subtasks] - sae_shifts.unsqueeze(1)

    return result
```

And replace `inverse_transform_batch`:

```python
def inverse_transform_batch(
    self,
    atomic_numbers: torch.Tensor,
    atom_counts: torch.Tensor,
    normalized: torch.Tensor,
) -> torch.Tensor:
    """Inverse SAE transformation (GPU-native, vectorized)."""
    device = normalized.device
    batch_size, max_atoms = atomic_numbers.shape

    sae_lookup = self._get_sae_lookup(device)

    atom_indices = torch.arange(max_atoms, device=device).unsqueeze(0)
    valid_mask = atom_indices < atom_counts.unsqueeze(1)

    sae_values = sae_lookup[atomic_numbers.clamp(0, self.max_atomic_num - 1)]
    sae_values = sae_values * valid_mask.float()
    sae_shifts = sae_values.sum(dim=1)  # [batch_size]

    # Vectorized subtask application using advanced indexing
    result = normalized.clone()
    subtask_indices = torch.tensor(self.subtasks, device=device, dtype=torch.long)
    valid_subtasks = subtask_indices[subtask_indices < result.shape[1]]

    if valid_subtasks.numel() > 0:
        result[:, valid_subtasks] = result[:, valid_subtasks] + sae_shifts.unsqueeze(1)

    return result
```

**Step 4: Run all preprocessing tests**

Run: `pytest tests/core/test_preprocessing.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/preprocessing.py tests/core/test_preprocessing.py
git commit -m "perf: vectorize SAE transform loops for torch.compile() compatibility"
```

---

## Task 5.4: Add Batch Adapter for SAE-Batch Format Mismatch

**Files:**
- Create: `src/core/batch_adapter.py`
- Test: `tests/core/test_batch_adapter.py`

**Step 1: Write tests for batch adapter**

Create `tests/core/test_batch_adapter.py`:

```python
"""Tests for batch adapter that bridges SAE and MolecularGraphBatch formats."""

import pytest
import torch

from src.core.batch import MolecularGraphBatch
from src.core.batch_adapter import BatchAdapter


class TestBatchAdapter:
    """Tests for BatchAdapter class."""

    def test_to_padded_format(self):
        """Test conversion from concatenated to padded format."""
        # Create a batch with 3 molecules of different sizes
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 1, 1, 8, 1, 6, 6], dtype=torch.int32),
            atomic_numbers=torch.tensor([6, 1, 1, 8, 1, 6, 6], dtype=torch.int64),
            batch_idx=torch.tensor([0, 0, 0, 1, 1, 2, 2], dtype=torch.int64),
            ptr=torch.tensor([0, 3, 5, 7], dtype=torch.int64),
            num_molecules=3,
        )

        adapter = BatchAdapter()
        padded_nums, atom_counts = adapter.to_padded_format(batch)

        # Should produce [3, max_atoms] tensor
        assert padded_nums.shape[0] == 3
        assert padded_nums.shape[1] == 3  # max atoms in any molecule
        assert atom_counts.tolist() == [3, 2, 2]

        # Check values
        assert padded_nums[0, :3].tolist() == [6, 1, 1]
        assert padded_nums[1, :2].tolist() == [8, 1]
        assert padded_nums[2, :2].tolist() == [6, 6]

    def test_to_padded_format_single_molecule(self):
        """Test conversion with single molecule batch."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 8, 1], dtype=torch.int32),
            atomic_numbers=torch.tensor([6, 8, 1], dtype=torch.int64),
            batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
            ptr=torch.tensor([0, 3], dtype=torch.int64),
            num_molecules=1,
        )

        adapter = BatchAdapter()
        padded_nums, atom_counts = adapter.to_padded_format(batch)

        assert padded_nums.shape == (1, 3)
        assert atom_counts.tolist() == [3]
        assert padded_nums[0].tolist() == [6, 8, 1]

    def test_round_trip_consistency(self):
        """Test that padding preserves atom information."""
        batch = MolecularGraphBatch(
            atom_types=torch.tensor([6, 1, 1, 1, 1, 8, 1, 1], dtype=torch.int32),
            atomic_numbers=torch.tensor([6, 1, 1, 1, 1, 8, 1, 1], dtype=torch.int64),
            batch_idx=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int64),
            ptr=torch.tensor([0, 5, 8], dtype=torch.int64),
            num_molecules=2,
        )

        adapter = BatchAdapter()
        padded_nums, atom_counts = adapter.to_padded_format(batch)

        # Verify all original atoms are preserved
        for mol_idx in range(batch.num_molecules):
            start = batch.ptr[mol_idx].item()
            end = batch.ptr[mol_idx + 1].item()
            original = batch.atomic_numbers[start:end]
            padded = padded_nums[mol_idx, :atom_counts[mol_idx]]

            assert torch.equal(original, padded)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_batch_adapter.py -v`
Expected: FAIL (module not found)

**Step 3: Implement BatchAdapter**

Create `src/core/batch_adapter.py`:

```python
"""
Batch adapter for bridging SAE and MolecularGraphBatch formats.

SAE transform expects padded tensors [batch_size, max_atoms] with atom_counts.
MolecularGraphBatch uses concatenated tensors [total_atoms] with batch_idx.

This adapter provides conversion between these formats.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .batch import MolecularGraphBatch


class BatchAdapter:
    """
    Adapter for converting between batch formats.

    Provides methods to:
    - Convert concatenated batch format to padded format for SAE
    - Extract atom counts from ptr tensor
    """

    def to_padded_format(
        self,
        batch: MolecularGraphBatch,
    ) -> tuple[Tensor, Tensor]:
        """
        Convert concatenated batch to padded format for SAE transform.

        Args:
            batch: MolecularGraphBatch with concatenated atomic_numbers

        Returns:
            Tuple of:
            - padded_atomic_numbers: [num_molecules, max_atoms] tensor
            - atom_counts: [num_molecules] tensor
        """
        device = batch.device
        num_molecules = batch.num_molecules

        # Get atomic numbers (prefer atomic_numbers, fall back to atom_types)
        if batch.atomic_numbers is not None:
            atomic_nums = batch.atomic_numbers
        else:
            atomic_nums = batch.atom_types.long()

        # Compute atom counts from ptr
        atom_counts = batch.ptr[1:] - batch.ptr[:-1]
        max_atoms = atom_counts.max().item()

        # Create padded tensor
        padded = torch.zeros(
            num_molecules, max_atoms,
            dtype=torch.int64,
            device=device,
        )

        # Fill in atomic numbers for each molecule
        for mol_idx in range(num_molecules):
            start = batch.ptr[mol_idx].item()
            end = batch.ptr[mol_idx + 1].item()
            count = end - start
            padded[mol_idx, :count] = atomic_nums[start:end]

        return padded, atom_counts

    def apply_sae_to_batch(
        self,
        batch: MolecularGraphBatch,
        sae_transform,
    ) -> Tensor:
        """
        Apply SAE transform to batch targets.

        Convenience method that handles format conversion automatically.

        Args:
            batch: MolecularGraphBatch with targets
            sae_transform: SAETransform instance

        Returns:
            Transformed targets [num_molecules, num_targets]
        """
        if batch.targets is None:
            raise ValueError("Batch has no targets to transform")

        padded_nums, atom_counts = self.to_padded_format(batch)

        return sae_transform.transform_batch(
            padded_nums,
            atom_counts,
            batch.targets,
        )

    def inverse_sae_from_batch(
        self,
        batch: MolecularGraphBatch,
        normalized: Tensor,
        sae_transform,
    ) -> Tensor:
        """
        Apply inverse SAE transform using batch atomic numbers.

        Args:
            batch: MolecularGraphBatch with atomic structure
            normalized: Normalized predictions [num_molecules, num_targets]
            sae_transform: SAETransform instance

        Returns:
            Denormalized predictions [num_molecules, num_targets]
        """
        padded_nums, atom_counts = self.to_padded_format(batch)

        return sae_transform.inverse_transform_batch(
            padded_nums,
            atom_counts,
            normalized,
        )
```

**Step 4: Run tests to verify implementation**

Run: `pytest tests/core/test_batch_adapter.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/batch_adapter.py tests/core/test_batch_adapter.py
git commit -m "feat: add BatchAdapter for SAE-batch format conversion"
```

---

## Task 5.5: Add Stereochemistry Processing to Model

**Files:**
- Modify: `src/core/model.py`
- Modify: `src/core/layers.py` (add StereochemistryEncoder)
- Test: `tests/core/test_model.py`

**Step 1: Write test for stereochemistry handling**

Add to `tests/core/test_model.py`:

```python
def test_model_processes_stereochemistry():
    """Test model can process batches with stereochemistry information."""
    config = ModelConfig(hidden_dim=64, output_dim=1)
    model = SimplifiedGNN(config)

    # Create batch with stereochemistry info
    batch = MolecularGraphBatch(
        atom_types=torch.randint(0, 10, (20,), dtype=torch.int32),
        degrees=torch.randint(0, 5, (20,), dtype=torch.int32),
        hybridizations=torch.randint(0, 6, (20,), dtype=torch.int32),
        hydrogen_counts=torch.randint(0, 5, (20,), dtype=torch.int32),
        batch_idx=torch.tensor([0]*10 + [1]*10, dtype=torch.int64),
        ptr=torch.tensor([0, 10, 20], dtype=torch.int64),
        edge_indices=[
            torch.randint(0, 20, (2, 30), dtype=torch.int64),
            torch.randint(0, 20, (2, 20), dtype=torch.int64),
            torch.randint(0, 20, (2, 10), dtype=torch.int64),
        ],
        num_molecules=2,
        # Stereochemistry: chiral center at atom 5, cis bond at atoms 2-3
        chiral_indices=torch.tensor([[5, 1, 2, 3]], dtype=torch.int64),
        cis_bond_indices=torch.tensor([[2, 3, 0, 4]], dtype=torch.int64),
        trans_bond_indices=torch.zeros((0, 4), dtype=torch.int64),
    )

    output = model(batch)
    assert output.shape == (2, 1)
    assert not torch.isnan(output).any()
```

**Step 2: Run test to see current behavior**

Run: `pytest tests/core/test_model.py::test_model_processes_stereochemistry -v`

**Step 3: Add StereochemistryEncoder to layers.py**

Add to `src/core/layers.py`:

```python
class StereochemistryEncoder(nn.Module):
    """
    Encodes stereochemistry information as atom-level features.

    Marks atoms involved in chiral centers and cis/trans bonds with
    learnable embeddings that get added to atom features.

    Args:
        hidden_dim: Dimension of atom features to match
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Learnable embeddings for stereochemistry types
        # Each stereo type gets a learnable vector added to atom features
        self.chiral_center_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.chiral_neighbor_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.cis_bond_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.trans_bond_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(
        self,
        x: Tensor,
        chiral_indices: Tensor | None,
        cis_bond_indices: Tensor | None,
        trans_bond_indices: Tensor | None,
    ) -> Tensor:
        """
        Add stereochemistry information to atom features.

        Args:
            x: Atom features [num_atoms, hidden_dim]
            chiral_indices: [num_chiral, 4] - center + 3 neighbors
            cis_bond_indices: [num_cis, 4] - bond atoms + neighbors
            trans_bond_indices: [num_trans, 4] - bond atoms + neighbors

        Returns:
            Updated atom features with stereochemistry encoded
        """
        output = x.clone()

        # Add chiral center embeddings
        if chiral_indices is not None and chiral_indices.shape[0] > 0:
            # Center atoms get chiral_center_embed
            center_atoms = chiral_indices[:, 0]
            output[center_atoms] = output[center_atoms] + self.chiral_center_embed

            # Neighbor atoms get chiral_neighbor_embed
            neighbor_atoms = chiral_indices[:, 1:4].flatten()
            valid_neighbors = neighbor_atoms[neighbor_atoms < x.shape[0]]
            if valid_neighbors.numel() > 0:
                output[valid_neighbors] = output[valid_neighbors] + self.chiral_neighbor_embed

        # Add cis bond embeddings
        if cis_bond_indices is not None and cis_bond_indices.shape[0] > 0:
            cis_atoms = cis_bond_indices[:, :2].flatten()
            valid_cis = cis_atoms[cis_atoms < x.shape[0]]
            if valid_cis.numel() > 0:
                output[valid_cis] = output[valid_cis] + self.cis_bond_embed

        # Add trans bond embeddings
        if trans_bond_indices is not None and trans_bond_indices.shape[0] > 0:
            trans_atoms = trans_bond_indices[:, :2].flatten()
            valid_trans = trans_atoms[trans_atoms < x.shape[0]]
            if valid_trans.numel() > 0:
                output[valid_trans] = output[valid_trans] + self.trans_bond_embed

        return output
```

**Step 4: Update SimplifiedGNN to use StereochemistryEncoder**

In `src/core/model.py`, add after projection layer initialization:

```python
# In __init__, after self.projection:
# 3.5 Stereochemistry encoder (optional, adds to atom features)
self.stereo_encoder = StereochemistryEncoder(config.hidden_dim)
```

Update the import:
```python
from .layers import ShellConvBlock, AttentionPooling, FeedForwardNetwork, StereochemistryEncoder
```

Update forward method after projection:

```python
# Project to hidden dimension [total_atoms, hidden_dim]
x = self.projection(x)

# Add stereochemistry information
x = self.stereo_encoder(
    x,
    batch.chiral_indices,
    batch.cis_bond_indices,
    batch.trans_bond_indices,
)

# Message passing layers
for mp_layer in self.message_passing_layers:
    x = mp_layer(x, batch.edge_indices)
```

**Step 5: Run tests**

Run: `pytest tests/core/test_model.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/core/layers.py src/core/model.py tests/core/test_model.py
git commit -m "feat: add stereochemistry encoding to SimplifiedGNN"
```

---

## Task 5.6: Add Total Charges to Model Input

**Files:**
- Modify: `src/core/model.py`
- Test: `tests/core/test_model.py`

**Step 1: Write test for total charge handling**

Add to `tests/core/test_model.py`:

```python
def test_model_with_total_charges():
    """Test model handles total molecular charges."""
    config = ModelConfig(hidden_dim=64, output_dim=1)
    model = SimplifiedGNN(config)

    batch = MolecularGraphBatch(
        atom_types=torch.randint(0, 10, (20,), dtype=torch.int32),
        degrees=torch.randint(0, 5, (20,), dtype=torch.int32),
        hybridizations=torch.randint(0, 6, (20,), dtype=torch.int32),
        hydrogen_counts=torch.randint(0, 5, (20,), dtype=torch.int32),
        batch_idx=torch.tensor([0]*10 + [1]*10, dtype=torch.int64),
        ptr=torch.tensor([0, 10, 20], dtype=torch.int64),
        edge_indices=[torch.randint(0, 20, (2, 30))],
        num_molecules=2,
        total_charges=torch.tensor([0.0, -1.0], dtype=torch.float32),
    )

    output = model(batch)
    assert output.shape == (2, 1)
```

**Step 2: Run test**

Run: `pytest tests/core/test_model.py::test_model_with_total_charges -v`

**Step 3: Add charge embedding to model**

In `src/core/model.py` `__init__`:

```python
# Add after embeddings:
# Charge embedding (added to molecule features after pooling)
self.charge_embedding = nn.Sequential(
    nn.Linear(1, config.hidden_dim),
    nn.SiLU(),
    nn.Linear(config.hidden_dim, config.hidden_dim),
)
```

In `forward`, after pooling and before FFN:

```python
# Pooling to molecule level [num_molecules, hidden_dim]
x = self.pooling(x, batch.batch_idx)

# Add charge information if available
if batch.total_charges is not None:
    charge_features = self.charge_embedding(
        batch.total_charges.unsqueeze(-1)
    )
    x = x + charge_features

# Feed-forward network [num_molecules, hidden_dim]
x = self.ffn(x)
```

**Step 4: Run tests**

Run: `pytest tests/core/test_model.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/model.py tests/core/test_model.py
git commit -m "feat: add total charge embedding to SimplifiedGNN"
```

---

## Task 5.7: Optimize Batch Construction with Pre-allocation

**Files:**
- Modify: `src/core/featurizer.py:186-212`
- Test: `tests/core/test_featurizer.py`

**Step 1: Write benchmark test**

Add to `tests/core/test_featurizer.py`:

```python
def test_stack_batch_produces_contiguous_tensors():
    """Test that batch construction produces contiguous tensors."""
    from src.core.featurizer import BatchFeaturizer
    import numpy as np

    featurizer = BatchFeaturizer(num_hops=3, num_workers=1)

    smiles = ["C", "CC", "CCC", "CCCC", "CCCCC"]
    targets = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

    batch = featurizer.featurize(smiles, targets)

    # All tensors should be contiguous for optimal GPU performance
    assert batch.atom_types.is_contiguous()
    assert batch.batch_idx.is_contiguous()
    assert batch.degrees.is_contiguous()
    assert batch.hybridizations.is_contiguous()
    assert batch.hydrogen_counts.is_contiguous()
```

**Step 2: Run test**

Run: `pytest tests/core/test_featurizer.py::test_stack_batch_produces_contiguous_tensors -v`

**Step 3: Optimize _stack_batch with vectorized construction**

In `src/core/featurizer.py`, replace `_stack_batch` method:

```python
def _stack_batch(
    self,
    molecules: list[dict[str, Any]],
    targets: np.ndarray,
) -> MolecularGraphBatch:
    """Stack molecule data into batch tensors with optimized allocation."""
    num_molecules = len(molecules)

    # Pre-compute sizes for single allocation
    atom_counts = [len(m["atom_types"]) for m in molecules]
    total_atoms = sum(atom_counts)

    # Build ptr tensor
    ptr = torch.zeros(num_molecules + 1, dtype=torch.int64)
    ptr[1:] = torch.tensor(atom_counts).cumsum(0)

    # Pre-allocate all atom tensors contiguously
    atom_types = torch.empty(total_atoms, dtype=torch.int32)
    degrees = torch.empty(total_atoms, dtype=torch.int32)
    hybridizations = torch.empty(total_atoms, dtype=torch.int32)
    hydrogen_counts = torch.empty(total_atoms, dtype=torch.int32)
    atomic_numbers = torch.empty(total_atoms, dtype=torch.int64)
    batch_idx = torch.empty(total_atoms, dtype=torch.int64)

    # Pre-allocate edge lists
    hop_edges_lists: list[list[torch.Tensor]] = [[] for _ in range(self.num_hops)]
    chiral_indices_list = []
    cis_indices_list = []
    trans_indices_list = []
    total_charges = torch.empty(num_molecules, dtype=torch.float32)
    smiles_list = []

    # Fill tensors (vectorized where possible)
    for mol_idx, mol in enumerate(molecules):
        start = ptr[mol_idx].item()
        end = ptr[mol_idx + 1].item()

        # Use copy_ for efficient in-place assignment
        atom_types[start:end].copy_(torch.from_numpy(mol["atom_types"]))
        degrees[start:end].copy_(torch.from_numpy(mol["degrees"]))
        hybridizations[start:end].copy_(torch.from_numpy(mol["hybridizations"]))
        hydrogen_counts[start:end].copy_(torch.from_numpy(mol["hydrogen_counts"]))
        atomic_numbers[start:end].copy_(torch.from_numpy(mol["atomic_numbers"]))
        batch_idx[start:end].fill_(mol_idx)

        offset = start
        for hop_idx, edges in enumerate(mol["multi_hop_edges"]):
            if edges.size > 0:
                edges_tensor = torch.from_numpy(edges).long() + offset
                hop_edges_lists[hop_idx].append(edges_tensor)

        for chiral in mol["chiral_centers"]:
            chiral_indices_list.append([c + offset for c in chiral])
        for cis in mol["cis_bonds"]:
            cis_indices_list.append([c + offset for c in cis])
        for trans in mol["trans_bonds"]:
            trans_indices_list.append([c + offset for c in trans])

        total_charges[mol_idx] = mol["total_charge"]
        smiles_list.append(mol["smiles"])

    # Build edge index tensors
    edge_indices = []
    for hop_edges in hop_edges_lists:
        if hop_edges:
            edge_indices.append(torch.cat(hop_edges, dim=1).contiguous())
        else:
            edge_indices.append(torch.zeros((2, 0), dtype=torch.int64))

    # Build stereochemistry tensors
    chiral_indices = (
        torch.tensor(chiral_indices_list, dtype=torch.int64).contiguous()
        if chiral_indices_list
        else torch.zeros((0, 4), dtype=torch.int64)
    )
    cis_indices = (
        torch.tensor(cis_indices_list, dtype=torch.int64).contiguous()
        if cis_indices_list
        else torch.zeros((0, 4), dtype=torch.int64)
    )
    trans_indices = (
        torch.tensor(trans_indices_list, dtype=torch.int64).contiguous()
        if trans_indices_list
        else torch.zeros((0, 4), dtype=torch.int64)
    )

    return MolecularGraphBatch(
        atom_types=atom_types.contiguous(),
        degrees=degrees.contiguous(),
        hybridizations=hybridizations.contiguous(),
        hydrogen_counts=hydrogen_counts.contiguous(),
        atomic_numbers=atomic_numbers.contiguous(),
        batch_idx=batch_idx.contiguous(),
        ptr=ptr,
        edge_indices=edge_indices,
        targets=torch.from_numpy(targets).float().contiguous(),
        total_charges=total_charges.contiguous(),
        smiles=smiles_list,
        chiral_indices=chiral_indices,
        cis_bond_indices=cis_indices,
        trans_bond_indices=trans_indices,
        num_molecules=num_molecules,
    )
```

**Step 4: Run tests**

Run: `pytest tests/core/test_featurizer.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/featurizer.py tests/core/test_featurizer.py
git commit -m "perf: optimize batch construction with pre-allocation and contiguous tensors"
```

---

## Task 5.8: Add torch.compile() Support

**Files:**
- Modify: `src/core/model.py`
- Test: `tests/core/test_model.py`

**Step 1: Write test for torch.compile()**

Add to `tests/core/test_model.py`:

```python
@pytest.mark.skipif(
    not hasattr(torch, 'compile'),
    reason="torch.compile not available"
)
def test_model_torch_compile_compatible():
    """Test that model can be compiled with torch.compile()."""
    config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
    model = SimplifiedGNN(config)

    # Compile the model
    compiled_model = torch.compile(model, mode="reduce-overhead")

    # Create test batch
    batch = MolecularGraphBatch(
        atom_types=torch.randint(0, 10, (15,), dtype=torch.int32),
        degrees=torch.randint(0, 5, (15,), dtype=torch.int32),
        hybridizations=torch.randint(0, 6, (15,), dtype=torch.int32),
        hydrogen_counts=torch.randint(0, 5, (15,), dtype=torch.int32),
        batch_idx=torch.tensor([0]*5 + [1]*5 + [2]*5, dtype=torch.int64),
        ptr=torch.tensor([0, 5, 10, 15], dtype=torch.int64),
        edge_indices=[
            torch.randint(0, 15, (2, 20), dtype=torch.int64),
            torch.randint(0, 15, (2, 15), dtype=torch.int64),
        ],
        num_molecules=3,
    )

    # Run forward pass - should not raise graph break errors
    output = compiled_model(batch)

    assert output.shape == (3, 1)
    assert not torch.isnan(output).any()
```

**Step 2: Run test**

Run: `pytest tests/core/test_model.py::test_model_torch_compile_compatible -v`

**Step 3: Add compile helper method**

In `src/core/model.py`, add method to SimplifiedGNN:

```python
def compile(self, **kwargs) -> "SimplifiedGNN":
    """
    Compile the model with torch.compile() for optimized execution.

    Args:
        **kwargs: Arguments passed to torch.compile()
                  Default mode is "reduce-overhead" for inference.

    Returns:
        Compiled model
    """
    if not hasattr(torch, 'compile'):
        logger.warning("torch.compile not available, returning uncompiled model")
        return self

    mode = kwargs.pop('mode', 'reduce-overhead')
    return torch.compile(self, mode=mode, **kwargs)
```

**Step 4: Run all model tests**

Run: `pytest tests/core/test_model.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/model.py tests/core/test_model.py
git commit -m "feat: add torch.compile() support with helper method"
```

---

## Task 5.9: Update __init__.py Exports

**Files:**
- Modify: `src/core/__init__.py`
- Test: Run import test

**Step 1: Check current exports**

Run: `python -c "from src.core import *; print('OK')"`

**Step 2: Update __init__.py**

Create/update `src/core/__init__.py`:

```python
"""
Core module for GPU-native molecular GNN.

This module provides the refactored architecture with:
- MolecularGraphBatch: Batched molecular data
- BatchFeaturizer: SMILES to batch conversion
- SimplifiedGNN: Main GNN model
- ModelConfig: Model configuration
- Preprocessing: SAE and scaling transforms
- BatchAdapter: Format conversion utilities
"""

from .batch import MolecularGraphBatch
from .batch_adapter import BatchAdapter
from .featurizer import BatchFeaturizer
from .model import SimplifiedGNN
from .model_config import ModelConfig
from .preprocessing import (
    SAETransform,
    StandardScaler,
    PreprocessingPipeline,
)
from .layers import (
    scatter_add,
    ShellConvBlock,
    AttentionPooling,
    FeedForwardNetwork,
    StereochemistryEncoder,
)

__all__ = [
    # Data
    "MolecularGraphBatch",
    "BatchFeaturizer",
    "BatchAdapter",
    # Model
    "SimplifiedGNN",
    "ModelConfig",
    # Preprocessing
    "SAETransform",
    "StandardScaler",
    "PreprocessingPipeline",
    # Layers
    "scatter_add",
    "ShellConvBlock",
    "AttentionPooling",
    "FeedForwardNetwork",
    "StereochemistryEncoder",
]
```

**Step 3: Verify imports**

Run: `python -c "from src.core import SimplifiedGNN, BatchAdapter, StereochemistryEncoder; print('All imports OK')"`
Expected: "All imports OK"

**Step 4: Commit**

```bash
git add src/core/__init__.py
git commit -m "chore: update core module exports with new components"
```

---

## Final Verification

**Run all core tests:**

```bash
pytest tests/core/ -v --tb=short
```

Expected: All tests PASS

**Run integration test:**

```bash
pytest tests/core/test_integration.py -v
```

Expected: All integration tests PASS
