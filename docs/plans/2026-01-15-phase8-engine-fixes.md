# Phase 8: Engine Critical Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical issues in Engine identified by AI/GPU/Architecture reviews - AMP consistency, metric aggregation, code duplication, and extensibility.

**Architecture:** Focus on correctness first (AMP, metrics), then maintainability (DRY refactor), then extensibility (loss registry). Each fix is isolated and testable.

**Tech Stack:** PyTorch 2.5.1, pytest, dataclasses

---

## Task 8.1: Add AMP Autocast to evaluate() Method

**Files:**
- Modify: `src/core/engine.py:330-361`
- Test: `tests/core/test_engine.py`

**Step 1: Write test for AMP consistency**

Add to `tests/core/test_engine.py`:

```python
class TestEngineAMPConsistency:
    """Tests for AMP consistency across methods."""

    def test_evaluate_uses_amp_when_enabled(self):
        """Test that evaluate uses AMP autocast like predict."""
        import unittest.mock as mock

        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=True)
        engine = Engine.from_config(model_config, engine_config)

        # Mock scaler to simulate AMP being enabled
        engine.scaler = mock.MagicMock()

        batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
            degrees=torch.randint(0, 5, (10,), dtype=torch.int32),
            hybridizations=torch.randint(0, 6, (10,), dtype=torch.int32),
            hydrogen_counts=torch.randint(0, 5, (10,), dtype=torch.int32),
            batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
            ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
            edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
            num_molecules=2,
            targets=torch.randn(2, 1),
        )

        # Should not raise and should return valid metrics
        metrics = engine.evaluate(batch)
        assert "loss" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
```

**Step 2: Run test**

Run: `pytest tests/core/test_engine.py::TestEngineAMPConsistency -v`
Expected: PASS (but we're adding AMP for correctness)

**Step 3: Update evaluate() to use AMP**

In `src/core/engine.py`, modify the `evaluate` method (around line 348):

```python
@torch.inference_mode()
def evaluate(self, batch: MolecularGraphBatch) -> dict[str, float]:
    """
    Evaluate model on batch.

    Args:
        batch: MolecularGraphBatch with targets

    Returns:
        Dict with 'loss', 'mae', 'rmse'
    """
    if batch.num_molecules == 0:
        return {"loss": 0.0, "mae": 0.0, "rmse": 0.0}
    if batch.targets is None:
        raise ValueError("Evaluation batch must have targets")

    self.model.eval()
    batch = batch.to(self.device)

    # Use AMP autocast for consistency with predict() and train_step()
    if self.scaler is not None:
        with torch.amp.autocast("cuda"):
            predictions = self.model(batch)
            loss = self.loss_fn(predictions, batch.targets)
    else:
        predictions = self.model(batch)
        loss = self.loss_fn(predictions, batch.targets)

    # Compute metrics
    diff = predictions - batch.targets
    mae = diff.abs().mean().item()
    rmse = diff.pow(2).mean().sqrt().item()

    return {
        "loss": loss.item(),
        "mae": mae,
        "rmse": rmse,
    }
```

**Step 4: Run tests**

Run: `pytest tests/core/test_engine.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "fix: add AMP autocast to evaluate() for consistency"
```

---

## Task 8.2: Fix Metric Aggregation for Multi-Output Models

**Files:**
- Modify: `src/core/engine.py:330-361` (evaluate)
- Modify: `src/core/engine.py:570-610` (evaluate_batches)
- Test: `tests/core/test_engine.py`

**Step 1: Write test for multi-output metric aggregation**

Add to `tests/core/test_engine.py`:

```python
class TestEngineMultiOutputMetrics:
    """Tests for multi-output metric computation."""

    def test_evaluate_returns_element_counts(self):
        """Test that evaluate returns element counts for proper aggregation."""
        model_config = ModelConfig(hidden_dim=32, output_dim=3, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False, warmup_epochs=0)
        engine = Engine.from_config(model_config, engine_config)

        batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
            degrees=torch.randint(0, 5, (10,), dtype=torch.int32),
            hybridizations=torch.randint(0, 6, (10,), dtype=torch.int32),
            hydrogen_counts=torch.randint(0, 5, (10,), dtype=torch.int32),
            batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
            ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
            edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
            num_molecules=2,
            targets=torch.randn(2, 3),  # 3 outputs per molecule
        )

        metrics = engine.evaluate(batch)

        # Should have raw sums for proper aggregation
        assert "abs_errors" in metrics
        assert "squared_errors" in metrics
        assert "num_elements" in metrics
        assert metrics["num_elements"] == 6  # 2 molecules * 3 outputs

    def test_evaluate_batches_aggregates_correctly(self):
        """Test weighted aggregation across batches with different sizes."""
        model_config = ModelConfig(hidden_dim=32, output_dim=2, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False, warmup_epochs=0)
        engine = Engine.from_config(model_config, engine_config)

        # Small batch: 1 molecule, 2 outputs = 2 elements
        small_batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (5,), dtype=torch.int32),
            degrees=torch.randint(0, 5, (5,), dtype=torch.int32),
            hybridizations=torch.randint(0, 6, (5,), dtype=torch.int32),
            hydrogen_counts=torch.randint(0, 5, (5,), dtype=torch.int32),
            batch_idx=torch.zeros(5, dtype=torch.int64),
            ptr=torch.tensor([0, 5], dtype=torch.int64),
            edge_indices=[torch.randint(0, 5, (2, 8), dtype=torch.int64)],
            num_molecules=1,
            targets=torch.randn(1, 2),
        )

        # Large batch: 5 molecules, 2 outputs = 10 elements
        large_batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (25,), dtype=torch.int32),
            degrees=torch.randint(0, 5, (25,), dtype=torch.int32),
            hybridizations=torch.randint(0, 6, (25,), dtype=torch.int32),
            hydrogen_counts=torch.randint(0, 5, (25,), dtype=torch.int32),
            batch_idx=torch.cat([torch.full((5,), i, dtype=torch.int64) for i in range(5)]),
            ptr=torch.tensor([0, 5, 10, 15, 20, 25], dtype=torch.int64),
            edge_indices=[torch.randint(0, 25, (2, 40), dtype=torch.int64)],
            num_molecules=5,
            targets=torch.randn(5, 2),
        )

        metrics = engine.evaluate_batches([small_batch, large_batch])

        assert metrics["total_molecules"] == 6
        assert metrics["total_elements"] == 12  # 2 + 10
        assert "mae" in metrics
        assert "rmse" in metrics
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_engine.py::TestEngineMultiOutputMetrics -v`
Expected: FAIL (no abs_errors/num_elements in metrics)

**Step 3: Update evaluate() to return raw sums**

In `src/core/engine.py`, update the `evaluate` method:

```python
@torch.inference_mode()
def evaluate(self, batch: MolecularGraphBatch) -> dict[str, float]:
    """
    Evaluate model on batch.

    Args:
        batch: MolecularGraphBatch with targets

    Returns:
        Dict with 'loss', 'mae', 'rmse', plus raw sums for aggregation
    """
    if batch.num_molecules == 0:
        return {
            "loss": 0.0, "mae": 0.0, "rmse": 0.0,
            "abs_errors": 0.0, "squared_errors": 0.0, "num_elements": 0,
        }
    if batch.targets is None:
        raise ValueError("Evaluation batch must have targets")

    self.model.eval()
    batch = batch.to(self.device)

    # Use AMP autocast for consistency
    if self.scaler is not None:
        with torch.amp.autocast("cuda"):
            predictions = self.model(batch)
            loss = self.loss_fn(predictions, batch.targets)
    else:
        predictions = self.model(batch)
        loss = self.loss_fn(predictions, batch.targets)

    # Compute metrics - return raw sums for proper aggregation
    diff = predictions - batch.targets
    abs_errors = diff.abs().sum().item()
    squared_errors = diff.pow(2).sum().item()
    num_elements = diff.numel()

    return {
        "loss": loss.item(),
        "mae": abs_errors / num_elements,
        "rmse": (squared_errors / num_elements) ** 0.5,
        # Raw values for aggregation
        "abs_errors": abs_errors,
        "squared_errors": squared_errors,
        "num_elements": num_elements,
    }
```

**Step 4: Update evaluate_batches() to use raw sums**

In `src/core/engine.py`, update the `evaluate_batches` method:

```python
def evaluate_batches(
    self,
    batches: list[MolecularGraphBatch],
) -> dict[str, float]:
    """
    Evaluate model on multiple batches with proper weighting.

    Args:
        batches: List of batches to evaluate

    Returns:
        Weighted aggregated metrics
    """
    total_loss = 0.0
    total_abs_errors = 0.0
    total_squared_errors = 0.0
    total_elements = 0
    total_molecules = 0

    for batch in batches:
        n = batch.num_molecules
        metrics = self.evaluate(batch)

        total_loss += metrics["loss"] * n
        total_abs_errors += metrics["abs_errors"]
        total_squared_errors += metrics["squared_errors"]
        total_elements += metrics["num_elements"]
        total_molecules += n

    if total_molecules == 0:
        return {
            "loss": 0.0, "mae": 0.0, "rmse": 0.0,
            "total_molecules": 0, "total_elements": 0,
        }

    return {
        "loss": total_loss / total_molecules,
        "mae": total_abs_errors / total_elements if total_elements > 0 else 0.0,
        "rmse": (total_squared_errors / total_elements) ** 0.5 if total_elements > 0 else 0.0,
        "total_molecules": total_molecules,
        "total_elements": total_elements,
    }
```

**Step 5: Run tests**

Run: `pytest tests/core/test_engine.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "fix: correct metric aggregation for multi-output models"
```

---

## Task 8.3: Extract Shared Training Logic (DRY Refactor)

**Files:**
- Modify: `src/core/engine.py`
- Test: `tests/core/test_engine.py`

**Step 1: Write test for refactored training methods**

Add to `tests/core/test_engine.py`:

```python
class TestEngineTrainingRefactor:
    """Tests for refactored training methods."""

    def test_train_step_uses_forward_backward(self):
        """Test that train_step still works after refactor."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False, warmup_epochs=0)
        engine = Engine.from_config(model_config, engine_config)

        batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
            degrees=torch.randint(0, 5, (10,), dtype=torch.int32),
            hybridizations=torch.randint(0, 6, (10,), dtype=torch.int32),
            hydrogen_counts=torch.randint(0, 5, (10,), dtype=torch.int32),
            batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
            ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
            edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
            num_molecules=2,
            targets=torch.randn(2, 1),
        )

        initial_weight = engine.model.output_layer.weight.clone()
        loss = engine.train_step(batch)

        assert isinstance(loss, float)
        assert not torch.allclose(initial_weight, engine.model.output_layer.weight)

    def test_accumulated_uses_forward_backward(self):
        """Test that train_step_accumulated still works after refactor."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False, warmup_epochs=0)
        engine = Engine.from_config(model_config, engine_config)

        batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
            degrees=torch.randint(0, 5, (10,), dtype=torch.int32),
            hybridizations=torch.randint(0, 6, (10,), dtype=torch.int32),
            hydrogen_counts=torch.randint(0, 5, (10,), dtype=torch.int32),
            batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
            ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
            edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
            num_molecules=2,
            targets=torch.randn(2, 1),
        )

        initial_weight = engine.model.output_layer.weight.clone()

        for step in range(4):
            engine.train_step_accumulated(batch, step, 4)

        assert not torch.allclose(initial_weight, engine.model.output_layer.weight)
```

**Step 2: Run tests (should pass - we're refactoring, not changing behavior)**

Run: `pytest tests/core/test_engine.py::TestEngineTrainingRefactor -v`
Expected: PASS

**Step 3: Extract _validate_training_batch helper**

In `src/core/engine.py`, add after the featurizer property (around line 130):

```python
def _validate_training_batch(self, batch: MolecularGraphBatch) -> None:
    """Validate batch for training operations.

    Args:
        batch: Batch to validate

    Raises:
        ValueError: If batch is empty or missing targets
    """
    if batch.num_molecules == 0:
        raise ValueError("Cannot train on empty batch")
    if batch.targets is None:
        raise ValueError("Training batch must have targets")
```

**Step 4: Extract _forward_backward helper**

In `src/core/engine.py`, add after _validate_training_batch:

```python
def _forward_backward(
    self,
    batch: MolecularGraphBatch,
    loss_scale: float = 1.0,
) -> torch.Tensor:
    """Perform forward pass and backward pass.

    Args:
        batch: Training batch (already on device)
        loss_scale: Scale factor for loss (for gradient accumulation)

    Returns:
        Unscaled loss tensor
    """
    if self.scaler is not None:
        with torch.amp.autocast("cuda"):
            predictions = self.model(batch)
            loss = self.loss_fn(predictions, batch.targets)
            scaled_loss = loss * loss_scale

        self.scaler.scale(scaled_loss).backward()
    else:
        predictions = self.model(batch)
        loss = self.loss_fn(predictions, batch.targets)
        scaled_loss = loss * loss_scale

        scaled_loss.backward()

    return loss

def _optimizer_step(self) -> None:
    """Perform optimizer step with gradient clipping."""
    if self.scaler is not None:
        if self.config.gradient_clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        if self.config.gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )
        self.optimizer.step()
```

**Step 5: Refactor train_step to use helpers**

Replace the existing `train_step` method:

```python
def train_step(self, batch: MolecularGraphBatch) -> float:
    """
    Perform single training step.

    Args:
        batch: MolecularGraphBatch with targets

    Returns:
        Loss value as float
    """
    self._validate_training_batch(batch)

    self.model.train()
    batch = batch.to(self.device)

    self.optimizer.zero_grad(set_to_none=True)
    loss = self._forward_backward(batch, loss_scale=1.0)
    self._optimizer_step()

    self.global_step += 1

    return loss.item()
```

**Step 6: Refactor train_step_accumulated to use helpers**

Replace the existing `train_step_accumulated` method:

```python
def train_step_accumulated(
    self,
    batch: MolecularGraphBatch,
    accumulation_step: int = 0,
    accumulation_steps: int = 1,
) -> float:
    """
    Perform training step with gradient accumulation.

    Args:
        batch: MolecularGraphBatch with targets
        accumulation_step: Current step in accumulation (0-indexed)
        accumulation_steps: Total number of accumulation steps

    Returns:
        Loss value as float (unscaled)

    Raises:
        ValueError: If batch is empty or missing targets
    """
    self._validate_training_batch(batch)

    if accumulation_steps <= 0:
        raise ValueError(f"accumulation_steps must be positive, got {accumulation_steps}")
    if accumulation_step < 0 or accumulation_step >= accumulation_steps:
        raise ValueError(
            f"accumulation_step must be in [0, {accumulation_steps - 1}], got {accumulation_step}"
        )

    self.model.train()
    batch = batch.to(self.device)

    # Only zero gradients at start of accumulation
    if accumulation_step == 0:
        self.optimizer.zero_grad(set_to_none=True)

    # Forward and backward with scaled loss
    loss = self._forward_backward(batch, loss_scale=1.0 / accumulation_steps)

    # Only step optimizer at end of accumulation
    if accumulation_step == accumulation_steps - 1:
        self._optimizer_step()
        self.global_step += 1

    return loss.item()
```

**Step 7: Run all tests**

Run: `pytest tests/core/test_engine.py -v`
Expected: All PASS

**Step 8: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "refactor: extract shared training logic to reduce duplication"
```

---

## Task 8.4: Add Loss Registry for Extensibility

**Files:**
- Create: `src/core/losses.py`
- Modify: `src/core/engine.py`
- Modify: `src/core/__init__.py`
- Test: `tests/core/test_losses.py`

**Step 1: Write test for loss registry**

Create `tests/core/test_losses.py`:

```python
"""Tests for loss registry."""
import pytest
import torch
import torch.nn as nn

from src.core.losses import LOSS_REGISTRY, register_loss, create_loss


class TestLossRegistry:
    """Tests for the loss registry system."""

    def test_builtin_losses_registered(self):
        """Test that built-in losses are in registry."""
        assert "mse" in LOSS_REGISTRY
        assert "mae" in LOSS_REGISTRY
        assert "huber" in LOSS_REGISTRY

    def test_create_mse_loss(self):
        """Test creating MSE loss."""
        loss_fn = create_loss("mse")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_create_mae_loss(self):
        """Test creating MAE loss."""
        loss_fn = create_loss("mae")
        assert isinstance(loss_fn, nn.L1Loss)

    def test_create_huber_loss(self):
        """Test creating Huber loss."""
        loss_fn = create_loss("huber")
        assert isinstance(loss_fn, nn.HuberLoss)

    def test_create_unknown_loss_raises(self):
        """Test that unknown loss raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss"):
            create_loss("unknown_loss")

    def test_register_custom_loss(self):
        """Test registering a custom loss."""
        @register_loss("custom_test")
        class CustomLoss(nn.Module):
            def forward(self, pred, target):
                return (pred - target).abs().sum()

        assert "custom_test" in LOSS_REGISTRY
        loss_fn = create_loss("custom_test")
        assert isinstance(loss_fn, CustomLoss)

        # Cleanup
        del LOSS_REGISTRY["custom_test"]
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_losses.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Create loss registry module**

Create `src/core/losses.py`:

```python
"""Loss function registry for extensible loss creation.

Provides a registry pattern for loss functions, allowing external
code to register custom losses without modifying the Engine class.
"""

from __future__ import annotations
from typing import Any

import torch.nn as nn


# Registry of loss function classes
LOSS_REGISTRY: dict[str, type[nn.Module]] = {}


def register_loss(name: str):
    """Decorator to register a loss function class.

    Args:
        name: Name to register the loss under

    Returns:
        Decorator function

    Example:
        @register_loss("custom")
        class CustomLoss(nn.Module):
            def forward(self, pred, target):
                return (pred - target).abs().mean()
    """
    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def create_loss(name: str, **kwargs: Any) -> nn.Module:
    """Create a loss function by name.

    Args:
        name: Name of the registered loss
        **kwargs: Arguments to pass to loss constructor

    Returns:
        Instantiated loss function

    Raises:
        ValueError: If loss name is not registered
    """
    if name not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY.keys()))
        raise ValueError(f"Unknown loss: {name}. Available: {available}")
    return LOSS_REGISTRY[name](**kwargs)


# Register built-in losses
@register_loss("mse")
class MSELoss(nn.MSELoss):
    """Mean Squared Error loss."""
    pass


@register_loss("mae")
class MAELoss(nn.L1Loss):
    """Mean Absolute Error loss (L1)."""
    pass


@register_loss("huber")
class HuberLoss(nn.HuberLoss):
    """Huber loss (smooth L1)."""
    pass
```

**Step 4: Run tests**

Run: `pytest tests/core/test_losses.py -v`
Expected: All PASS

**Step 5: Update Engine to use registry**

In `src/core/engine.py`, add import at top:

```python
from .losses import create_loss
```

Replace `_create_loss_function` method:

```python
def _create_loss_function(self) -> nn.Module:
    """Create loss function from registry based on config."""
    return create_loss(self.config.loss_function)
```

**Step 6: Update __init__.py exports**

In `src/core/__init__.py`, add:

```python
from .losses import (
    LOSS_REGISTRY,
    register_loss,
    create_loss,
)
```

And update `__all__`:

```python
__all__ = [
    # ... existing exports ...
    # Losses
    "LOSS_REGISTRY",
    "register_loss",
    "create_loss",
]
```

**Step 7: Run all tests**

Run: `pytest tests/core/ -v`
Expected: All PASS

**Step 8: Commit**

```bash
git add src/core/losses.py src/core/engine.py src/core/__init__.py tests/core/test_losses.py
git commit -m "feat: add loss registry for extensible loss functions"
```

---

## Task 8.5: Add Evidential Loss to Registry

**Files:**
- Modify: `src/core/losses.py`
- Modify: `src/core/engine_config.py`
- Test: `tests/core/test_losses.py`

**Step 1: Write test for evidential loss**

Add to `tests/core/test_losses.py`:

```python
def test_create_evidential_loss(self):
    """Test creating evidential loss."""
    loss_fn = create_loss("evidential")

    # Test forward pass
    pred = torch.randn(10, 4)  # mu, v, alpha, beta
    target = torch.randn(10, 1)

    loss = loss_fn(pred, target)
    assert loss.shape == ()
    assert not torch.isnan(loss)
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_losses.py::TestLossRegistry::test_create_evidential_loss -v`
Expected: FAIL (evidential not registered)

**Step 3: Add evidential loss to registry**

In `src/core/losses.py`, add:

```python
import torch
from torch import Tensor


@register_loss("evidential")
class EvidentialLoss(nn.Module):
    """Evidential regression loss for uncertainty quantification.

    Expects predictions of shape [batch, 4] containing:
    - mu: Mean prediction
    - v: Variance of mean (epistemic uncertainty)
    - alpha: Shape parameter (alpha > 1)
    - beta: Scale parameter (beta > 0)

    Based on "Deep Evidential Regression" (Amini et al., 2020).
    """

    def __init__(self, coeff: float = 0.01):
        """Initialize evidential loss.

        Args:
            coeff: Regularization coefficient for evidence
        """
        super().__init__()
        self.coeff = coeff

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute evidential loss.

        Args:
            pred: Predictions [batch, 4] - (mu, v, alpha, beta)
            target: Targets [batch, 1]

        Returns:
            Scalar loss
        """
        # Unpack predictions
        mu = pred[:, 0:1]
        v = torch.nn.functional.softplus(pred[:, 1:2]) + 1e-6
        alpha = torch.nn.functional.softplus(pred[:, 2:3]) + 1.0
        beta = torch.nn.functional.softplus(pred[:, 3:4]) + 1e-6

        # NLL loss
        twoBlambda = 2 * beta * (1 + v)
        nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * torch.log(twoBlambda)
            + (alpha + 0.5) * torch.log(v * (target - mu) ** 2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # Regularization on evidence
        reg = (2 * v + alpha) * torch.abs(target - mu)

        loss = nll + self.coeff * reg
        return loss.mean()
```

**Step 4: Update EngineConfig to accept evidential**

In `src/core/engine_config.py`, update the loss_function type:

```python
loss_function: Literal["mse", "mae", "huber", "evidential"] = "mse"
```

**Step 5: Run tests**

Run: `pytest tests/core/test_losses.py tests/core/test_engine_config.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/core/losses.py src/core/engine_config.py tests/core/test_losses.py
git commit -m "feat: add evidential loss for uncertainty quantification"
```

---

## Task 8.6: Add MC Dropout Inference Support

**Files:**
- Modify: `src/core/engine.py`
- Test: `tests/core/test_engine.py`

**Step 1: Write test for MC Dropout**

Add to `tests/core/test_engine.py`:

```python
class TestEngineMCDropout:
    """Tests for MC Dropout inference."""

    def test_predict_mc_dropout(self):
        """Test MC Dropout inference returns multiple samples."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2, dropout=0.1)
        engine_config = EngineConfig(device="cpu", use_amp=False, warmup_epochs=0)
        engine = Engine.from_config(model_config, engine_config)

        batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
            degrees=torch.randint(0, 5, (10,), dtype=torch.int32),
            hybridizations=torch.randint(0, 6, (10,), dtype=torch.int32),
            hydrogen_counts=torch.randint(0, 5, (10,), dtype=torch.int32),
            batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
            ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
            edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
            num_molecules=2,
        )

        # MC Dropout with 10 samples
        predictions = engine.predict_mc_dropout(batch, num_samples=10)

        assert predictions.shape == (10, 2, 1)  # [samples, molecules, output_dim]

        # Different samples should give different results (due to dropout)
        # Note: With very small dropout or lucky seeds, this could rarely fail
        std = predictions.std(dim=0)
        assert std.mean() > 0  # Should have some variance

    def test_predict_mc_dropout_returns_stats(self):
        """Test MC Dropout can return mean and std."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2, dropout=0.1)
        engine_config = EngineConfig(device="cpu", use_amp=False, warmup_epochs=0)
        engine = Engine.from_config(model_config, engine_config)

        batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
            degrees=torch.randint(0, 5, (10,), dtype=torch.int32),
            hybridizations=torch.randint(0, 6, (10,), dtype=torch.int32),
            hydrogen_counts=torch.randint(0, 5, (10,), dtype=torch.int32),
            batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
            ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
            edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
            num_molecules=2,
        )

        mean, std = engine.predict_mc_dropout(batch, num_samples=10, return_stats=True)

        assert mean.shape == (2, 1)
        assert std.shape == (2, 1)
        assert (std >= 0).all()
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_engine.py::TestEngineMCDropout -v`
Expected: FAIL (no predict_mc_dropout method)

**Step 3: Add predict_mc_dropout method**

In `src/core/engine.py`, add after the `predict` method:

```python
def predict_mc_dropout(
    self,
    batch: MolecularGraphBatch,
    num_samples: int = 30,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Run MC Dropout inference for uncertainty estimation.

    Performs multiple forward passes with dropout enabled to estimate
    epistemic uncertainty.

    Args:
        batch: MolecularGraphBatch (targets optional)
        num_samples: Number of MC samples
        return_stats: If True, return (mean, std) instead of all samples

    Returns:
        If return_stats=False: Predictions [num_samples, num_molecules, output_dim]
        If return_stats=True: Tuple of (mean, std), each [num_molecules, output_dim]
    """
    batch = batch.to(self.device)

    # Keep model in train mode to enable dropout
    self.model.train()

    samples = []
    with torch.no_grad():  # No gradients needed for inference
        for _ in range(num_samples):
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    pred = self.model(batch)
            else:
                pred = self.model(batch)
            samples.append(pred.cpu())

    # Restore eval mode
    self.model.eval()

    # Stack samples: [num_samples, num_molecules, output_dim]
    all_samples = torch.stack(samples, dim=0)

    if return_stats:
        mean = all_samples.mean(dim=0)
        std = all_samples.std(dim=0)
        return mean, std

    return all_samples
```

**Step 4: Run tests**

Run: `pytest tests/core/test_engine.py::TestEngineMCDropout -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "feat: add MC Dropout inference for uncertainty estimation"
```

---

## Final Verification

Run all core tests to ensure everything works:

```bash
pytest tests/core/ -v --tb=short
```

Expected: All tests PASS (should be 120+ tests)

---

## Summary of Changes

| Task | Type | Description |
|------|------|-------------|
| 8.1 | Fix | AMP autocast in evaluate() |
| 8.2 | Fix | Metric aggregation for multi-output |
| 8.3 | Refactor | Extract shared training logic |
| 8.4 | Feat | Loss registry for extensibility |
| 8.5 | Feat | Evidential loss support |
| 8.6 | Feat | MC Dropout inference |
