# Phase 2: Code Quality Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve code quality by extracting magic numbers to constants, deduplicating preprocessing logic, and splitting overly long functions.

**Architecture:** Three focused improvements that reduce technical debt and improve maintainability without changing behavior.

**Tech Stack:** Python 3.12, pytest for verification

---

## Summary

| Task | Description | Files | Impact |
|------|-------------|-------|--------|
| 2.1 | Extract magic numbers to constants | 3 files | Configurability |
| 2.2 | Deduplicate preprocessing code | 1 file | DRY principle |
| 2.3 | Split long functions | 2 files | Readability |

---

## Task 2.1: Extract Magic Numbers to Constants

**Files:**
- Create: `src/config/constants.py`
- Modify: `src/models/gnn.py`
- Modify: `src/training/trainer.py`

### Step 1: Create constants file

Create `/home/olexandr/AIMNet-X2D/src/config/constants.py`:

```python
"""
Model and training constants.

These values are extracted from inline magic numbers to improve
configurability and documentation.
"""

# Model Architecture
MESSAGE_PASSING_RATIO = 0.3  # Fraction of hidden_dim for message passing (x_other)
DEFAULT_ATTENTION_TEMPERATURE = 1.0  # Softmax temperature for attention pooling

# Stereochemistry
TETRAHEDRAL_MAGNITUDE_SCALE = 3.0  # Divisor for tanh scaling of tetrahedral features

# Training
GRADIENT_CLIP_MAX_NORM = 1.0  # Maximum gradient norm for clipping
DEFAULT_EVIDENTIAL_LAMBDA = 1.0  # Default regularization for evidential loss
```

### Step 2: Update gnn.py to use constants

In `/home/olexandr/AIMNet-X2D/src/models/gnn.py`:

```python
# Add import
from config.constants import MESSAGE_PASSING_RATIO, TETRAHEDRAL_MAGNITUDE_SCALE

# Line ~101: Replace magic number
# Before
self.x_other_dim = int(0.3 * hidden_dim)
# After
self.x_other_dim = int(MESSAGE_PASSING_RATIO * hidden_dim)

# Line ~386: Replace magic number
# Before
magnitude_scale = torch.tanh(avg_magnitude / 3.0)
# After
magnitude_scale = torch.tanh(avg_magnitude / TETRAHEDRAL_MAGNITUDE_SCALE)
```

### Step 3: Update trainer.py to use constants

In `/home/olexandr/AIMNet-X2D/src/training/trainer.py`:

```python
# Add import
from config.constants import GRADIENT_CLIP_MAX_NORM, DEFAULT_EVIDENTIAL_LAMBDA

# Line ~52: Replace magic number
# Before
lambda_reg = getattr(current_args, 'evidential_lambda', 1.0)
# After
lambda_reg = getattr(current_args, 'evidential_lambda', DEFAULT_EVIDENTIAL_LAMBDA)

# Lines ~167, ~184: Replace magic numbers
# Before
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# After
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
```

### Step 4: Update config/__init__.py

Add export in `/home/olexandr/AIMNet-X2D/src/config/__init__.py`:

```python
from .constants import (
    MESSAGE_PASSING_RATIO,
    DEFAULT_ATTENTION_TEMPERATURE,
    TETRAHEDRAL_MAGNITUDE_SCALE,
    GRADIENT_CLIP_MAX_NORM,
    DEFAULT_EVIDENTIAL_LAMBDA,
)
```

### Step 5: Run tests and commit

```bash
python -m pytest tests/ -q --tb=short
git add src/config/constants.py src/config/__init__.py src/models/gnn.py src/training/trainer.py
git commit -m "Extract magic numbers to src/config/constants.py"
```

---

## Task 2.2: Deduplicate Preprocessing Code in Evaluator

**File:** `/home/olexandr/AIMNet-X2D/src/training/evaluator.py`

**Problem:** SAE inverse transform logic is duplicated in 4 functions:
- `_compute_multitask_metrics` (lines 217-280)
- `_compute_single_task_metrics` (lines 281-351)
- `_combine_multitask_ddp_metrics` (lines 352-421)
- `_combine_single_task_ddp_metrics` (lines 422-477)

### Step 1: Create shared helper function

Add after imports (around line 25):

```python
def _apply_inverse_preprocessing(
    predictions: np.ndarray,
    targets: np.ndarray,
    smiles_list: list[str],
    preprocessing_pipeline: Any | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply inverse preprocessing to predictions and targets.

    Handles SAE denormalization and standard scaling inverse transform.
    Returns predictions and targets on the original scale.
    """
    if preprocessing_pipeline is None:
        return predictions, targets

    # Ensure 2D shape for preprocessing pipeline
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    # Apply inverse transform (handles both SAE and scaling)
    predictions = preprocessing_pipeline.inverse_transform(
        smiles_list=smiles_list,
        transformed_targets=predictions
    )
    targets = preprocessing_pipeline.inverse_transform(
        smiles_list=smiles_list,
        transformed_targets=targets
    )

    return predictions, targets
```

### Step 2: Refactor _compute_multitask_metrics

Replace the preprocessing logic (around lines 240-260) with:

```python
# Apply inverse preprocessing
preds_np, targets_np = _apply_inverse_preprocessing(
    preds_np, targets_np, smiles_list, preprocessing_pipeline
)
```

### Step 3: Refactor _compute_single_task_metrics

Replace similar preprocessing logic with the helper call.

### Step 4: Refactor _combine_multitask_ddp_metrics

Replace similar preprocessing logic with the helper call.

### Step 5: Refactor _combine_single_task_ddp_metrics

Replace similar preprocessing logic with the helper call.

### Step 6: Run tests and commit

```bash
python -m pytest tests/ -q --tb=short
git add src/training/evaluator.py
git commit -m "Deduplicate preprocessing logic in evaluator.py"
```

---

## Task 2.3: Split Long Functions

### Task 2.3a: Split _handle_epoch_end_fixed in trainer.py

**File:** `/home/olexandr/AIMNet-X2D/src/training/trainer.py`
**Function:** `_handle_epoch_end_fixed` (lines 358-487, ~130 lines)

**Extract these helpers:**

```python
def _update_learning_rate(
    scheduler: Any | None,
    val_loss: float,
    scheduler_name: str
) -> None:
    """Step the learning rate scheduler based on validation loss."""
    if scheduler is None:
        return
    if scheduler_name == 'reduce_on_plateau':
        scheduler.step(val_loss)
    else:
        scheduler.step()


def _check_early_stopping(
    val_loss: float,
    best_val_loss: float,
    patience_counter: int,
    patience: int,
    min_delta: float
) -> tuple[float, int, bool, bool]:
    """
    Check early stopping condition.

    Returns: (new_best_loss, new_patience_counter, is_best, should_stop)
    """
    is_best = val_loss < best_val_loss - min_delta
    if is_best:
        return val_loss, 0, True, False
    else:
        new_counter = patience_counter + 1
        should_stop = new_counter >= patience
        return best_val_loss, new_counter, False, should_stop


def _broadcast_early_stopping_decision(
    should_stop: bool,
    is_distributed: bool,
    device: torch.device
) -> bool:
    """Broadcast early stopping decision across DDP ranks."""
    if not is_distributed:
        return should_stop

    stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
    dist.broadcast(stop_tensor, src=0)
    return stop_tensor.item() == 1
```

### Task 2.3b: Split _verify_hdf5_model_compatibility in pipeline.py

**File:** `/home/olexandr/AIMNet-X2D/src/inference/pipeline.py`
**Function:** `_verify_hdf5_model_compatibility` (lines 161-289, ~130 lines)

**Extract these helpers:**

```python
def _check_hdf5_max_hops(
    hdf5_max_hops: int,
    model_num_shells: int
) -> str | None:
    """Check max_hops compatibility. Returns error message or None."""
    if hdf5_max_hops != model_num_shells:
        return (
            f"HDF5 max_hops ({hdf5_max_hops}) != model num_shells ({model_num_shells}). "
            f"Regenerate HDF5 with matching max_hops."
        )
    return None


def _check_hdf5_preprocessing(
    hdf5_preprocessing: bool,
    model_has_preprocessing: bool
) -> str | None:
    """Check preprocessing compatibility. Returns error message or None."""
    if hdf5_preprocessing and not model_has_preprocessing:
        return (
            "HDF5 has preprocessing applied but model has no preprocessing pipeline. "
            "This will produce incorrect results."
        )
    return None
```

Then refactor `_verify_hdf5_model_compatibility` to use these helpers.

### Step: Run tests and commit

```bash
python -m pytest tests/ -q --tb=short
git add src/training/trainer.py src/inference/pipeline.py
git commit -m "Split long functions into focused helpers"
```

---

## Final Verification

```bash
# Run full test suite
python -m pytest tests/ -v --tb=short

# Verify no regressions
git diff --stat HEAD~3
```

---

## Estimated Scope

| Task | Files | Complexity | Est. Time |
|------|-------|------------|-----------|
| 2.1 Constants | 4 files | Low | 15 min |
| 2.2 Dedup | 1 file | Medium | 20 min |
| 2.3 Split | 2 files | Medium | 25 min |
| **Total** | **7 files** | | **~1 hour** |
