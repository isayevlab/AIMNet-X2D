# Phase 7: Engine Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix architecture issues and add performance optimizations to the Phase 6 Engine implementation based on AI engineer and performance reviews.

**Architecture:** Apply quick-win performance fixes first, then address architecture issues (preprocessing integration, metric aggregation, input validation), then add advanced features (pluggable loss functions, warmup scheduler, gradient accumulation).

**Tech Stack:** PyTorch 2.x, torch.inference_mode, torch.compile, GradScaler

---

## Task 7.1: Quick Performance Wins - Engine

**Files:**
- Modify: `src/core/engine.py`
- Test: `tests/core/test_engine.py`

**Step 1: Write test for inference_mode behavior**

Add to `tests/core/test_engine.py`:

```python
class TestEnginePerformance:
    """Tests for Engine performance optimizations."""

    def test_predict_uses_inference_mode(self):
        """Test that predict uses inference mode (no grad tracking)."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu")
        engine = Engine.from_config(model_config, engine_config)

        batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
            batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
            ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
            edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
            num_molecules=2,
        )

        # Should not raise and should not require grad
        predictions = engine.predict(batch)
        assert not predictions.requires_grad
```

**Step 2: Run test**

Run: `pytest tests/core/test_engine.py::TestEnginePerformance -v`
Expected: PASS (current @torch.no_grad() also works, but we'll improve it)

**Step 3: Apply performance improvements to engine.py**

Update `src/core/engine.py`:

1. Change line 150 from `self.optimizer.zero_grad()` to:
```python
self.optimizer.zero_grad(set_to_none=True)
```

2. Change line 187 from `@torch.no_grad()` to:
```python
@torch.inference_mode()
```

3. Change line 209 from `@torch.no_grad()` to:
```python
@torch.inference_mode()
```

**Step 4: Run all engine tests**

Run: `pytest tests/core/test_engine.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "perf: use inference_mode and set_to_none=True in Engine"
```

---

## Task 7.2: Quick Performance Wins - Batch Device Check

**Files:**
- Modify: `src/core/batch.py`
- Test: `tests/core/test_batch.py`

**Step 1: Write test for device early return**

Add to `tests/core/test_batch.py`:

```python
def test_to_same_device_returns_self():
    """Test that .to() returns same object if already on target device."""
    batch = MolecularGraphBatch(
        atom_types=torch.tensor([6, 6, 8], dtype=torch.int32),
        batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
        ptr=torch.tensor([0, 3], dtype=torch.int64),
        num_molecules=1,
    )

    # Should return same object (identity check)
    batch2 = batch.to("cpu")
    assert batch2 is batch


def test_to_uses_non_blocking_for_pinned():
    """Test that .to() uses non_blocking for pinned tensors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch = MolecularGraphBatch(
        atom_types=torch.tensor([6, 6, 8], dtype=torch.int32),
        batch_idx=torch.tensor([0, 0, 0], dtype=torch.int64),
        ptr=torch.tensor([0, 3], dtype=torch.int64),
        num_molecules=1,
    )
    pinned = batch.pin_memory()

    # Should not raise
    gpu_batch = pinned.to("cuda")
    assert gpu_batch.device.type == "cuda"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_batch.py::test_to_same_device_returns_self -v`
Expected: FAIL (current implementation always creates new object)

**Step 3: Update batch.py .to() method**

Replace the `to()` method in `src/core/batch.py` (lines 68-105):

```python
def to(self, device: torch.device | str) -> MolecularGraphBatch:
    """
    Move all tensors to the specified device.

    Args:
        device: Target device (e.g., 'cuda', 'cpu', torch.device)

    Returns:
        New MolecularGraphBatch with tensors on the target device,
        or self if already on target device
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Early return if already on target device
    if self.device == device:
        return self

    def move_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        # Use non_blocking if tensor is pinned (for async GPU transfer)
        return t.to(device, non_blocking=t.is_pinned())

    def move_tensor_list(
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [t.to(device, non_blocking=t.is_pinned()) for t in tensors]

    return MolecularGraphBatch(
        atom_types=self.atom_types.to(device, non_blocking=self.atom_types.is_pinned()),
        batch_idx=self.batch_idx.to(device, non_blocking=self.batch_idx.is_pinned()),
        ptr=self.ptr.to(device, non_blocking=self.ptr.is_pinned()),
        num_molecules=self.num_molecules,
        degrees=move_tensor(self.degrees),
        hybridizations=move_tensor(self.hybridizations),
        hydrogen_counts=move_tensor(self.hydrogen_counts),
        edge_indices=move_tensor_list(self.edge_indices),
        targets=move_tensor(self.targets),
        total_charges=move_tensor(self.total_charges),
        smiles=self.smiles.copy(),
        chiral_indices=move_tensor(self.chiral_indices),
        cis_bond_indices=move_tensor(self.cis_bond_indices),
        trans_bond_indices=move_tensor(self.trans_bond_indices),
        atomic_numbers=move_tensor(self.atomic_numbers),
    )
```

**Step 4: Run tests**

Run: `pytest tests/core/test_batch.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/batch.py tests/core/test_batch.py
git commit -m "perf: add device early-return and non_blocking transfers to batch"
```

---

## Task 7.3: Fix AttentionPooling .item() Call

**Files:**
- Modify: `src/core/layers.py`
- Modify: `src/core/model.py`
- Test: `tests/core/test_layers.py`

**Step 1: Write test for AttentionPooling with num_molecules param**

Add to `tests/core/test_layers.py`:

```python
class TestAttentionPoolingPerformance:
    """Performance tests for AttentionPooling."""

    def test_pooling_accepts_num_molecules_param(self):
        """Test that pooling can accept num_molecules to avoid .item() call."""
        pooling = AttentionPooling(input_dim=64, num_heads=4)
        x = torch.randn(10, 64)
        batch_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])

        # Should work with explicit num_molecules
        output = pooling(x, batch_idx, num_molecules=3)
        assert output.shape == (3, 64)

    def test_pooling_backward_compatible(self):
        """Test that pooling still works without num_molecules."""
        pooling = AttentionPooling(input_dim=64, num_heads=4)
        x = torch.randn(10, 64)
        batch_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])

        # Should still work without num_molecules (backward compatible)
        output = pooling(x, batch_idx)
        assert output.shape == (3, 64)
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/core/test_layers.py::TestAttentionPoolingPerformance -v`
Expected: FAIL (TypeError: forward() got unexpected keyword argument)

**Step 3: Update AttentionPooling.forward signature**

In `src/core/layers.py`, update the `forward` method (around line 245):

```python
def forward(
    self,
    x: Tensor,
    batch_idx: Tensor,
    num_molecules: int | None = None,
) -> Tensor:
    """
    Forward pass through attention pooling.

    Args:
        x: Atom features [num_atoms, input_dim]
        batch_idx: Molecule index for each atom [num_atoms]
        num_molecules: Number of molecules (optional, avoids .item() call)

    Returns:
        Molecule features [num_molecules, input_dim]
    """
    num_atoms = x.shape[0]

    # Use provided num_molecules or compute it (slower due to .item())
    if num_molecules is None:
        num_molecules = int(batch_idx.max().item()) + 1

    # ... rest of method unchanged (use num_molecules variable)
```

**Step 4: Update model.py to pass num_molecules**

In `src/core/model.py`, update line 178:

```python
# Before:
x = self.pooling(x, batch.batch_idx)

# After:
x = self.pooling(x, batch.batch_idx, num_molecules=batch.num_molecules)
```

**Step 5: Run tests**

Run: `pytest tests/core/test_layers.py tests/core/test_model.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/core/layers.py src/core/model.py tests/core/test_layers.py
git commit -m "perf: add num_molecules param to AttentionPooling to avoid .item()"
```

---

## Task 7.4: Fix Weighted Metric Aggregation

**Files:**
- Modify: `src/core/engine.py`
- Modify: `tests/core/test_engine.py`

**Step 1: Write test for weighted metrics**

Add to `tests/core/test_engine.py`:

```python
def test_evaluate_batches_weights_by_size():
    """Test that evaluate_batches weights metrics by batch size."""
    model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
    engine_config = EngineConfig(device="cpu")
    engine = Engine.from_config(model_config, engine_config)

    # Create batches with different sizes
    small_batch = MolecularGraphBatch(
        atom_types=torch.randint(0, 10, (5,), dtype=torch.int32),
        batch_idx=torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64),
        ptr=torch.tensor([0, 5], dtype=torch.int64),
        edge_indices=[torch.randint(0, 5, (2, 8), dtype=torch.int64)],
        num_molecules=1,
        targets=torch.tensor([[0.0]]),
    )

    large_batch = MolecularGraphBatch(
        atom_types=torch.randint(0, 10, (50,), dtype=torch.int32),
        batch_idx=torch.cat([torch.full((5,), i, dtype=torch.int64) for i in range(10)]),
        ptr=torch.tensor([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=torch.int64),
        edge_indices=[torch.randint(0, 50, (2, 80), dtype=torch.int64)],
        num_molecules=10,
        targets=torch.randn(10, 1),
    )

    metrics = engine.evaluate_batches([small_batch, large_batch])

    # Metrics should exist
    assert "loss" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "total_molecules" in metrics
    assert metrics["total_molecules"] == 11
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_engine.py::test_evaluate_batches_weights_by_size -v`
Expected: FAIL (no total_molecules in metrics)

**Step 3: Update evaluate_batches in engine.py**

Replace the `evaluate_batches` method (lines 448-476):

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
        Weighted aggregated metrics ('loss', 'mae', 'rmse', 'total_molecules')
    """
    total_loss = 0.0
    total_mae = 0.0
    total_squared_error = 0.0
    total_molecules = 0

    for batch in batches:
        n = batch.num_molecules
        metrics = self.evaluate(batch)

        total_loss += metrics["loss"] * n
        total_mae += metrics["mae"] * n
        # For RMSE, accumulate squared errors (not averaged RMSEs)
        total_squared_error += (metrics["rmse"] ** 2) * n
        total_molecules += n

    if total_molecules == 0:
        return {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "total_molecules": 0}

    return {
        "loss": total_loss / total_molecules,
        "mae": total_mae / total_molecules,
        "rmse": (total_squared_error / total_molecules) ** 0.5,
        "total_molecules": total_molecules,
    }
```

**Step 4: Run tests**

Run: `pytest tests/core/test_engine.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "fix: weight evaluate_batches metrics by batch size"
```

---

## Task 7.5: Add Input Validation

**Files:**
- Modify: `src/core/engine.py`
- Modify: `tests/core/test_engine.py`

**Step 1: Write tests for input validation**

Add to `tests/core/test_engine.py`:

```python
class TestEngineValidation:
    """Tests for Engine input validation."""

    def test_train_step_requires_targets(self):
        """Test that train_step raises for batch without targets."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False)
        engine = Engine.from_config(model_config, engine_config)

        batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
            batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
            ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
            edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
            num_molecules=2,
            targets=None,  # No targets!
        )

        with pytest.raises(ValueError, match="targets"):
            engine.train_step(batch)

    def test_train_step_rejects_empty_batch(self):
        """Test that train_step raises for empty batch."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False)
        engine = Engine.from_config(model_config, engine_config)

        batch = MolecularGraphBatch(
            atom_types=torch.tensor([], dtype=torch.int32),
            batch_idx=torch.tensor([], dtype=torch.int64),
            ptr=torch.tensor([0], dtype=torch.int64),
            edge_indices=[torch.zeros(2, 0, dtype=torch.int64)],
            num_molecules=0,
            targets=torch.zeros(0, 1),
        )

        with pytest.raises(ValueError, match="empty"):
            engine.train_step(batch)

    def test_evaluate_requires_targets(self):
        """Test that evaluate raises for batch without targets."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu")
        engine = Engine.from_config(model_config, engine_config)

        batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
            batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
            ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
            edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
            num_molecules=2,
            targets=None,
        )

        with pytest.raises(ValueError, match="targets"):
            engine.evaluate(batch)

    def test_evaluate_empty_batch_returns_zeros(self):
        """Test that evaluate returns zeros for empty batch."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu")
        engine = Engine.from_config(model_config, engine_config)

        batch = MolecularGraphBatch(
            atom_types=torch.tensor([], dtype=torch.int32),
            batch_idx=torch.tensor([], dtype=torch.int64),
            ptr=torch.tensor([0], dtype=torch.int64),
            edge_indices=[torch.zeros(2, 0, dtype=torch.int64)],
            num_molecules=0,
            targets=torch.zeros(0, 1),
        )

        metrics = engine.evaluate(batch)
        assert metrics["loss"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/core/test_engine.py::TestEngineValidation -v`
Expected: FAIL

**Step 3: Add validation to train_step**

In `src/core/engine.py`, add validation at the start of `train_step` (after line 147):

```python
def train_step(self, batch: MolecularGraphBatch) -> float:
    """
    Perform single training step.

    Args:
        batch: MolecularGraphBatch with targets

    Returns:
        Loss value as float

    Raises:
        ValueError: If batch is empty or missing targets
    """
    if batch.num_molecules == 0:
        raise ValueError("Cannot train on empty batch")
    if batch.targets is None:
        raise ValueError("Training batch must have targets")

    self.model.train()
    # ... rest unchanged
```

**Step 4: Add validation to evaluate**

In `src/core/engine.py`, add validation at the start of `evaluate` (after line 220):

```python
@torch.inference_mode()
def evaluate(self, batch: MolecularGraphBatch) -> dict[str, float]:
    """
    Evaluate model on batch.

    Args:
        batch: MolecularGraphBatch with targets

    Returns:
        Dict with 'loss', 'mae', 'rmse'

    Raises:
        ValueError: If batch is missing targets
    """
    if batch.num_molecules == 0:
        return {"loss": 0.0, "mae": 0.0, "rmse": 0.0}
    if batch.targets is None:
        raise ValueError("Evaluation batch must have targets")

    self.model.eval()
    # ... rest unchanged
```

**Step 5: Run tests**

Run: `pytest tests/core/test_engine.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "feat: add input validation to Engine train_step and evaluate"
```

---

## Task 7.6: Add Pluggable Loss Functions

**Files:**
- Modify: `src/core/engine_config.py`
- Modify: `src/core/engine.py`
- Test: `tests/core/test_engine_config.py`
- Test: `tests/core/test_engine.py`

**Step 1: Write tests for loss function config**

Add to `tests/core/test_engine_config.py`:

```python
def test_loss_function_config():
    """Test that loss_function can be configured."""
    config = EngineConfig(loss_function="mae")
    assert config.loss_function == "mae"

    config2 = EngineConfig(loss_function="mse")
    assert config2.loss_function == "mse"


def test_loss_function_in_to_dict():
    """Test that loss_function is serialized."""
    config = EngineConfig(loss_function="huber")
    d = config.to_dict()
    assert d["loss_function"] == "huber"
```

Add to `tests/core/test_engine.py`:

```python
def test_engine_with_mae_loss():
    """Test Engine with MAE loss function."""
    model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
    engine_config = EngineConfig(device="cpu", use_amp=False, loss_function="mae")
    engine = Engine.from_config(model_config, engine_config)

    batch = MolecularGraphBatch(
        atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
        batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
        ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
        edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
        num_molecules=2,
        targets=torch.randn(2, 1),
    )

    loss = engine.train_step(batch)
    assert isinstance(loss, float)
    assert loss >= 0  # MAE is always non-negative


def test_engine_with_huber_loss():
    """Test Engine with Huber loss function."""
    model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
    engine_config = EngineConfig(device="cpu", use_amp=False, loss_function="huber")
    engine = Engine.from_config(model_config, engine_config)

    batch = MolecularGraphBatch(
        atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
        batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
        ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
        edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
        num_molecules=2,
        targets=torch.randn(2, 1),
    )

    loss = engine.train_step(batch)
    assert isinstance(loss, float)
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/core/test_engine_config.py::test_loss_function_config -v`
Expected: FAIL (no loss_function field)

**Step 3: Add loss_function to EngineConfig**

In `src/core/engine_config.py`, add after line 47:

```python
# Loss
loss_function: Literal["mse", "mae", "huber"] = "mse"
```

Update `to_dict()` to include loss_function (add to the dict):
```python
"loss_function": self.loss_function,
```

**Step 4: Add _create_loss_function to Engine**

In `src/core/engine.py`, replace line 79 (`self.loss_fn = nn.MSELoss()`) with:

```python
# Loss function
self.loss_fn = self._create_loss_function()
```

Add the method:

```python
def _create_loss_function(self) -> nn.Module:
    """Create loss function based on config."""
    loss_type = self.config.loss_function

    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")
```

**Step 5: Run tests**

Run: `pytest tests/core/test_engine.py tests/core/test_engine_config.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/core/engine.py src/core/engine_config.py tests/core/test_engine.py tests/core/test_engine_config.py
git commit -m "feat: add pluggable loss functions (mse, mae, huber)"
```

---

## Task 7.7: Implement Learning Rate Warmup

**Files:**
- Modify: `src/core/engine.py`
- Test: `tests/core/test_engine.py`

**Step 1: Write test for warmup**

Add to `tests/core/test_engine.py`:

```python
def test_warmup_scheduler():
    """Test that warmup epochs work correctly."""
    model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
    engine_config = EngineConfig(
        device="cpu",
        scheduler="cosine",
        warmup_epochs=3,
        epochs=10,
        learning_rate=1e-3,
    )
    engine = Engine.from_config(model_config, engine_config)

    # First step should have reduced LR due to warmup
    initial_lr = engine.get_lr()
    assert initial_lr < engine_config.learning_rate  # Should start lower

    # Step through warmup
    for _ in range(3):
        engine.step_scheduler()

    # After warmup, should be at or near base LR
    after_warmup_lr = engine.get_lr()
    assert after_warmup_lr >= initial_lr  # Should have increased
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_engine.py::test_warmup_scheduler -v`
Expected: FAIL (warmup not implemented)

**Step 3: Update _create_scheduler to support warmup**

In `src/core/engine.py`, replace the `_create_scheduler` method:

```python
def _create_scheduler(self):
    """Create learning rate scheduler based on config with optional warmup."""
    if self.config.scheduler == "none":
        return None

    # Create main scheduler
    if self.config.scheduler == "cosine":
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.epochs - self.config.warmup_epochs),
            eta_min=self.config.learning_rate * 0.01,
        )
    elif self.config.scheduler == "plateau":
        main_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )
    else:
        return None

    # Add warmup if configured
    if self.config.warmup_epochs > 0 and self.config.scheduler != "plateau":
        from torch.optim.lr_scheduler import LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs,
        )

        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.warmup_epochs],
        )

    return main_scheduler
```

**Step 4: Run tests**

Run: `pytest tests/core/test_engine.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "feat: implement learning rate warmup in Engine scheduler"
```

---

## Task 7.8: Add Gradient Accumulation Support

**Files:**
- Modify: `src/core/engine.py`
- Test: `tests/core/test_engine.py`

**Step 1: Write test for gradient accumulation**

Add to `tests/core/test_engine.py`:

```python
def test_gradient_accumulation():
    """Test gradient accumulation over multiple steps."""
    model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
    engine_config = EngineConfig(device="cpu", use_amp=False)
    engine = Engine.from_config(model_config, engine_config)

    batch = MolecularGraphBatch(
        atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
        batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
        ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
        edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
        num_molecules=2,
        targets=torch.randn(2, 1),
    )

    # Get initial weights
    initial_weight = engine.model.output_layer.weight.clone()

    # Accumulate gradients over 4 steps
    accumulation_steps = 4
    for step in range(accumulation_steps):
        engine.train_step_accumulated(
            batch,
            accumulation_step=step,
            accumulation_steps=accumulation_steps,
        )

    # Weights should have changed after accumulation completes
    final_weight = engine.model.output_layer.weight
    assert not torch.allclose(initial_weight, final_weight)
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_engine.py::test_gradient_accumulation -v`
Expected: FAIL (no train_step_accumulated method)

**Step 3: Add train_step_accumulated method**

In `src/core/engine.py`, add after train_step:

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
    if batch.num_molecules == 0:
        raise ValueError("Cannot train on empty batch")
    if batch.targets is None:
        raise ValueError("Training batch must have targets")

    self.model.train()
    batch = batch.to(self.device)

    # Only zero gradients at start of accumulation
    if accumulation_step == 0:
        self.optimizer.zero_grad(set_to_none=True)

    # Scale loss for accumulation
    scale_factor = 1.0 / accumulation_steps

    if self.scaler is not None:
        with torch.amp.autocast("cuda"):
            predictions = self.model(batch)
            loss = self.loss_fn(predictions, batch.targets)
            scaled_loss = loss * scale_factor

        self.scaler.scale(scaled_loss).backward()

        # Only step optimizer at end of accumulation
        if accumulation_step == accumulation_steps - 1:
            if self.config.gradient_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1
    else:
        predictions = self.model(batch)
        loss = self.loss_fn(predictions, batch.targets)
        scaled_loss = loss * scale_factor

        scaled_loss.backward()

        # Only step optimizer at end of accumulation
        if accumulation_step == accumulation_steps - 1:
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
            self.optimizer.step()
            self.global_step += 1

    return loss.item()  # Return unscaled loss for logging
```

**Step 4: Run tests**

Run: `pytest tests/core/test_engine.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "feat: add gradient accumulation support to Engine"
```

---

## Task 7.9: Optimize StereochemistryEncoder

**Files:**
- Modify: `src/core/layers.py`
- Test: `tests/core/test_layers.py`

**Step 1: Write test for optimized stereo encoder**

Add to `tests/core/test_layers.py`:

```python
def test_stereo_encoder_no_clone_when_empty():
    """Test that StereochemistryEncoder doesn't clone when no stereo info."""
    encoder = StereochemistryEncoder(hidden_dim=64)
    x = torch.randn(10, 64)

    # No stereochemistry - should not clone
    output = encoder(x, None, None, None)

    # Output should be same object when no stereo info
    assert output is x


def test_stereo_encoder_clones_when_needed():
    """Test that StereochemistryEncoder clones when stereo info present."""
    encoder = StereochemistryEncoder(hidden_dim=64)
    x = torch.randn(10, 64)

    # Has chiral centers - should clone
    chiral_indices = torch.tensor([[0, 1, 2, 3]])
    output = encoder(x, chiral_indices, None, None)

    # Output should be different object
    assert output is not x
    # But values at non-chiral atoms should be same
    assert torch.allclose(output[4:], x[4:])
```

**Step 2: Run tests to verify behavior**

Run: `pytest tests/core/test_layers.py::test_stereo_encoder_no_clone_when_empty -v`
Expected: FAIL (current implementation always clones)

**Step 3: Optimize StereochemistryEncoder**

In `src/core/layers.py`, update the `forward` method of `StereochemistryEncoder`:

```python
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
    # Early return if no stereochemistry info
    has_chiral = chiral_indices is not None and chiral_indices.shape[0] > 0
    has_cis = cis_bond_indices is not None and cis_bond_indices.shape[0] > 0
    has_trans = trans_bond_indices is not None and trans_bond_indices.shape[0] > 0

    if not (has_chiral or has_cis or has_trans):
        return x

    # Only clone if we need to modify
    output = x.clone()

    # Add chiral center embeddings
    if has_chiral:
        center_atoms = chiral_indices[:, 0]
        output[center_atoms] = output[center_atoms] + self.chiral_center_embed

        neighbor_atoms = chiral_indices[:, 1:4].flatten()
        valid_neighbors = neighbor_atoms[neighbor_atoms < x.shape[0]]
        if valid_neighbors.numel() > 0:
            output[valid_neighbors] = output[valid_neighbors] + self.chiral_neighbor_embed

    # Add cis bond embeddings
    if has_cis:
        cis_atoms = cis_bond_indices[:, :2].flatten()
        valid_cis = cis_atoms[cis_atoms < x.shape[0]]
        if valid_cis.numel() > 0:
            output[valid_cis] = output[valid_cis] + self.cis_bond_embed

    # Add trans bond embeddings
    if has_trans:
        trans_atoms = trans_bond_indices[:, :2].flatten()
        valid_trans = trans_atoms[trans_atoms < x.shape[0]]
        if valid_trans.numel() > 0:
            output[valid_trans] = output[valid_trans] + self.trans_bond_embed

    return output
```

**Step 4: Run tests**

Run: `pytest tests/core/test_layers.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/layers.py tests/core/test_layers.py
git commit -m "perf: optimize StereochemistryEncoder to avoid clone when no stereo"
```

---

## Task 7.10: Clean Up _message_passing Redundant Ops

**Files:**
- Modify: `src/core/layers.py`
- Test: `tests/core/test_layers.py`

**Step 1: Write test for message passing efficiency**

Add to `tests/core/test_layers.py`:

```python
def test_shell_conv_empty_edges():
    """Test ShellConvBlock handles empty edges without redundant ops."""
    block = ShellConvBlock(input_dim=64, hidden_dim=64, num_shells=2)
    x = torch.randn(10, 64)

    # Empty edges for both shells
    edge_indices = [
        torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(2, 0, dtype=torch.long),
    ]

    # Should work without errors
    output = block(x, edge_indices)
    assert output.shape == (10, 64)
    assert not torch.isnan(output).any()
```

**Step 2: Run test**

Run: `pytest tests/core/test_layers.py::test_shell_conv_empty_edges -v`
Expected: PASS (should already work, but we're optimizing)

**Step 3: Optimize _message_passing**

In `src/core/layers.py`, replace the `_message_passing` method:

```python
def _message_passing(
    self,
    x: Tensor,
    edge_index: Tensor,
    transform: nn.Linear,
    num_atoms: int,
) -> Tensor:
    """
    Perform message passing for a single shell.

    Args:
        x: Atom features [num_atoms, input_dim]
        edge_index: Edge indices [2, num_edges]
        transform: Linear transform for this shell
        num_atoms: Number of atoms

    Returns:
        Aggregated features [num_atoms, hidden_dim]
    """
    # Transform all features (needed for gradient flow even with empty edges)
    transformed = transform(x)

    # Handle empty edges - return zeros
    if edge_index.shape[1] == 0:
        return torch.zeros(
            num_atoms, self.hidden_dim,
            device=x.device, dtype=x.dtype
        )

    # Gather and aggregate
    source_idx = edge_index[0]
    target_idx = edge_index[1]
    source_features = transformed[source_idx]

    return scatter_add(
        source_features,
        target_idx,
        dim=0,
        dim_size=num_atoms,
    )
```

**Step 4: Run tests**

Run: `pytest tests/core/test_layers.py tests/core/test_model.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/layers.py tests/core/test_layers.py
git commit -m "perf: remove redundant tensor ops in _message_passing"
```

---

## Final Verification

Run all core tests to ensure everything works:

```bash
pytest tests/core/ -v --tb=short
```

Expected: All tests PASS (should be 95+ tests)

---

## Summary of Changes

| Task | Type | Description |
|------|------|-------------|
| 7.1 | Perf | inference_mode + set_to_none |
| 7.2 | Perf | Batch device check + non_blocking |
| 7.3 | Perf | AttentionPooling num_molecules param |
| 7.4 | Fix | Weighted metric aggregation |
| 7.5 | Fix | Input validation |
| 7.6 | Feat | Pluggable loss functions |
| 7.7 | Feat | Learning rate warmup |
| 7.8 | Feat | Gradient accumulation |
| 7.9 | Perf | StereochemistryEncoder optimization |
| 7.10 | Perf | _message_passing cleanup |
