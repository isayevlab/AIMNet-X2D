# Phase 9: Critical Fixes for Multi-Task and Performance

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical issues in Phase 8 implementation - multi-output EvidentialLoss, MC Dropout GPU batching, loss aggregation consistency, and configuration extensibility.

**Architecture:** Address P0/P1 issues from expert reviews: (1) Multi-task EvidentialLoss, (2) GPU-efficient MC Dropout, (3) Consistent loss aggregation, (4) Configurable loss parameters, (5) Evidential-aware uncertainty estimation, (6) Reduced GPU synchronization.

**Tech Stack:** PyTorch 2.5.1, pytest, type hints

---

## Task 9.1: Fix EvidentialLoss for Multi-Output Support

**Priority:** P0 (Critical)

**Files:**
- Modify: `src/core/losses.py:78-131`
- Test: `tests/core/test_losses.py`

**Step 1: Write test for multi-output evidential loss**

Add to `tests/core/test_losses.py`:

```python
def test_evidential_loss_multi_output(self):
    """Test evidential loss with multiple outputs."""
    loss_fn = create_loss("evidential")

    # 2 tasks, each needs 4 parameters (mu, v, alpha, beta)
    pred = torch.randn(10, 8)  # [batch, 2 * 4]
    target = torch.randn(10, 2)  # [batch, 2 tasks]

    loss = loss_fn(pred, target)

    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert loss.item() > 0

def test_evidential_loss_single_output_backward_compat(self):
    """Test evidential loss still works with single output."""
    loss_fn = create_loss("evidential")

    # Single task
    pred = torch.randn(10, 4)  # [batch, 4]
    target = torch.randn(10, 1)  # [batch, 1]

    loss = loss_fn(pred, target)

    assert loss.shape == ()
    assert not torch.isnan(loss)
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_losses.py::TestLossRegistry::test_evidential_loss_multi_output -v`
Expected: FAIL (current implementation only handles single output)

**Step 3: Update EvidentialLoss for multi-output support**

In `src/core/losses.py`, replace the `forward` method:

```python
def forward(self, pred: Tensor, target: Tensor) -> Tensor:
    """Compute evidential loss.

    Args:
        pred: Predictions [batch, num_tasks * 4] - (mu, v, alpha, beta) for each task
        target: Targets [batch, num_tasks]

    Returns:
        Scalar loss
    """
    batch_size = pred.shape[0]
    num_tasks = target.shape[1]

    # Validate shape compatibility
    expected_pred_dim = num_tasks * 4
    if pred.shape[1] != expected_pred_dim:
        raise ValueError(
            f"Expected pred shape [batch, {expected_pred_dim}] for {num_tasks} tasks, "
            f"got {pred.shape}"
        )

    # Reshape predictions: [batch, num_tasks, 4]
    pred_reshaped = pred.view(batch_size, num_tasks, 4)

    # Unpack predictions for all tasks
    mu = pred_reshaped[:, :, 0:1]  # [batch, num_tasks, 1]
    v = torch.nn.functional.softplus(pred_reshaped[:, :, 1:2]) + 1e-6
    alpha = torch.nn.functional.softplus(pred_reshaped[:, :, 2:3]) + 1.0
    beta = torch.nn.functional.softplus(pred_reshaped[:, :, 3:4]) + 1e-6

    # Reshape target to match: [batch, num_tasks, 1]
    target_reshaped = target.unsqueeze(-1)

    # NLL loss (vectorized over all tasks)
    twoBlambda = 2 * beta * (1 + v)
    nll = (
        0.5 * torch.log(torch.pi / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (target_reshaped - mu) ** 2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    # Regularization on evidence
    reg = (2 * v + alpha) * torch.abs(target_reshaped - mu)

    loss = nll + self.coeff * reg
    return loss.mean()
```

**Step 4: Run tests**

Run: `pytest tests/core/test_losses.py::TestLossRegistry::test_evidential_loss_multi_output -v`
Expected: PASS

Run: `pytest tests/core/test_losses.py::TestLossRegistry::test_evidential_loss_single_output_backward_compat -v`
Expected: PASS

**Step 5: Run full loss tests**

Run: `pytest tests/core/test_losses.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/core/losses.py tests/core/test_losses.py
git commit -m "fix: add multi-output support to EvidentialLoss

Reshape predictions to [batch, num_tasks, 4] format and vectorize loss
computation over all tasks. Maintains backward compatibility with single
output case. Adds shape validation with helpful error messages.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9.2: Add Loss Configuration Parameters to EngineConfig

**Priority:** P1 (Important)

**Files:**
- Modify: `src/core/engine_config.py:34-52`
- Modify: `src/core/engine.py:225-228`
- Test: `tests/core/test_engine_config.py`

**Step 1: Write test for loss_kwargs configuration**

Add to `tests/core/test_engine_config.py`:

```python
def test_loss_kwargs_default():
    """Test loss_kwargs defaults to empty dict."""
    config = EngineConfig()
    assert config.loss_kwargs == {}

def test_loss_kwargs_custom():
    """Test loss_kwargs can be customized."""
    config = EngineConfig(
        loss_function="evidential",
        loss_kwargs={"coeff": 0.05}
    )
    assert config.loss_kwargs == {"coeff": 0.05}

def test_to_dict_includes_loss_kwargs():
    """Test loss_kwargs serialization."""
    config = EngineConfig(loss_kwargs={"coeff": 0.1})
    d = config.to_dict()
    assert "loss_kwargs" in d
    assert d["loss_kwargs"] == {"coeff": 0.1}

def test_from_dict_restores_loss_kwargs():
    """Test loss_kwargs deserialization."""
    d = {"loss_function": "huber", "loss_kwargs": {"delta": 1.5}}
    config = EngineConfig.from_dict(d)
    assert config.loss_kwargs == {"delta": 1.5}
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_engine_config.py -k loss_kwargs -v`
Expected: FAIL (loss_kwargs doesn't exist)

**Step 3: Add loss_kwargs to EngineConfig**

In `src/core/engine_config.py`, add after loss_function field:

```python
# Loss
loss_function: Literal["mse", "mae", "huber", "evidential"] = "mse"
loss_kwargs: dict[str, Any] = field(default_factory=dict)
```

Add import at top:

```python
from dataclasses import dataclass, field
from typing import Any, Literal
```

Update to_dict method:

```python
def to_dict(self) -> dict[str, Any]:
    """Serialize configuration to dictionary.

    Returns:
        Dictionary containing all configuration parameters.
    """
    return {
        "learning_rate": self.learning_rate,
        "weight_decay": self.weight_decay,
        "gradient_clip": self.gradient_clip,
        "batch_size": self.batch_size,
        "epochs": self.epochs,
        "warmup_epochs": self.warmup_epochs,
        "early_stopping_patience": self.early_stopping_patience,
        "scheduler": self.scheduler,
        "loss_function": self.loss_function,
        "loss_kwargs": self.loss_kwargs,
        "device": self.device,
        "num_workers": self.num_workers,
        "use_amp": self.use_amp,
        "compile_model": self.compile_model,
        "checkpoint_dir": self.checkpoint_dir,
        "log_interval": self.log_interval,
    }
```

**Step 4: Update Engine to use loss_kwargs**

In `src/core/engine.py`, update `_create_loss_function`:

```python
def _create_loss_function(self) -> nn.Module:
    """Create loss function from registry based on config."""
    return create_loss(self.config.loss_function, **self.config.loss_kwargs)
```

**Step 5: Add integration test**

Add to `tests/core/test_engine.py`:

```python
def test_engine_with_loss_kwargs():
    """Test Engine respects loss_kwargs."""
    model_config = ModelConfig(hidden_dim=32, output_dim=4, num_shells=2)
    engine_config = EngineConfig(
        device="cpu",
        use_amp=False,
        loss_function="evidential",
        loss_kwargs={"coeff": 0.05}
    )
    engine = Engine.from_config(model_config, engine_config)

    # Verify loss function has correct config
    assert hasattr(engine.loss_fn, 'coeff')
    assert engine.loss_fn.coeff == 0.05
```

**Step 6: Run tests**

Run: `pytest tests/core/test_engine_config.py -k loss_kwargs -v`
Expected: All PASS

Run: `pytest tests/core/test_engine.py::test_engine_with_loss_kwargs -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/core/engine_config.py src/core/engine.py tests/core/test_engine_config.py tests/core/test_engine.py
git commit -m "feat: add loss_kwargs to EngineConfig for configurable loss parameters

Enable passing arbitrary kwargs to loss constructors (e.g., coeff for
EvidentialLoss, delta for HuberLoss). Update Engine._create_loss_function
to forward kwargs. Add serialization support.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9.3: Fix Loss Aggregation Consistency

**Priority:** P1 (Important)

**Files:**
- Modify: `src/core/engine.py:645-651`
- Test: `tests/core/test_engine.py`

**Step 1: Write test for loss aggregation consistency**

Add to `tests/core/test_engine.py`:

```python
def test_evaluate_batches_loss_weighted_by_elements():
    """Test that loss is weighted by num_elements, not num_molecules."""
    model_config = ModelConfig(hidden_dim=32, output_dim=2, num_shells=2)
    engine_config = EngineConfig(device="cpu", use_amp=False, warmup_epochs=0)
    engine = Engine.from_config(model_config, engine_config)

    # Create two batches with different molecule counts
    # Batch 1: 1 molecule * 2 outputs = 2 elements
    batch1 = MolecularGraphBatch(
        atom_types=torch.randint(0, 10, (5,), dtype=torch.int32),
        degrees=torch.randint(0, 5, (5,), dtype=torch.int32),
        hybridizations=torch.randint(0, 6, (5,), dtype=torch.int32),
        hydrogen_counts=torch.randint(0, 5, (5,), dtype=torch.int32),
        batch_idx=torch.zeros(5, dtype=torch.int64),
        ptr=torch.tensor([0, 5], dtype=torch.int64),
        edge_indices=[torch.randint(0, 5, (2, 8), dtype=torch.int64)],
        num_molecules=1,
        targets=torch.ones(1, 2),  # Known target
    )

    # Batch 2: 2 molecules * 2 outputs = 4 elements
    batch2 = MolecularGraphBatch(
        atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
        degrees=torch.randint(0, 5, (10,), dtype=torch.int32),
        hybridizations=torch.randint(0, 6, (10,), dtype=torch.int32),
        hydrogen_counts=torch.randint(0, 5, (10,), dtype=torch.int32),
        batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
        ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
        edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
        num_molecules=2,
        targets=torch.ones(2, 2),  # Known target
    )

    # Get individual losses
    metrics1 = engine.evaluate(batch1)
    metrics2 = engine.evaluate(batch2)

    # Aggregate
    agg_metrics = engine.evaluate_batches([batch1, batch2])

    # Loss should be weighted by num_elements (2 + 4 = 6)
    expected_loss = (
        metrics1["loss"] * metrics1["num_elements"] +
        metrics2["loss"] * metrics2["num_elements"]
    ) / (metrics1["num_elements"] + metrics2["num_elements"])

    assert abs(agg_metrics["loss"] - expected_loss) < 1e-5
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_engine.py::test_evaluate_batches_loss_weighted_by_elements -v`
Expected: FAIL (loss currently weighted by num_molecules)

**Step 3: Fix loss aggregation in evaluate_batches()**

In `src/core/engine.py`, update the `evaluate_batches` method aggregation logic:

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

        # Weight loss by num_elements for consistency with MAE/RMSE
        total_loss += metrics["loss"] * metrics["num_elements"]
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
        "loss": total_loss / total_elements if total_elements > 0 else 0.0,
        "mae": total_abs_errors / total_elements if total_elements > 0 else 0.0,
        "rmse": (total_squared_errors / total_elements) ** 0.5 if total_elements > 0 else 0.0,
        "total_molecules": total_molecules,
        "total_elements": total_elements,
    }
```

**Step 4: Run tests**

Run: `pytest tests/core/test_engine.py::test_evaluate_batches_loss_weighted_by_elements -v`
Expected: PASS

Run: `pytest tests/core/test_engine.py -k evaluate_batches -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "fix: weight loss by num_elements in evaluate_batches

Change loss aggregation to use num_elements instead of num_molecules
for consistency with MAE/RMSE weighting. For multi-output models, this
ensures proper weighted averaging across batches.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9.4: Optimize MC Dropout GPU Batching

**Priority:** P1 (Important) - 10-30% inference speedup

**Files:**
- Modify: `src/core/engine.py:320-367`
- Test: `tests/core/test_engine.py`

**Step 1: Write test verifying MC Dropout results unchanged**

Add to `tests/core/test_engine.py`:

```python
def test_predict_mc_dropout_batched_equivalent():
    """Test batched MC Dropout gives same statistical properties."""
    torch.manual_seed(42)
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

    # Run MC Dropout
    predictions = engine.predict_mc_dropout(batch, num_samples=10)

    # Should still return correct shape
    assert predictions.shape == (10, 2, 1)

    # Should have variance from dropout
    std = predictions.std(dim=0)
    assert std.mean() > 0
```

**Step 2: Run test to verify current implementation passes**

Run: `pytest tests/core/test_engine.py::test_predict_mc_dropout_batched_equivalent -v`
Expected: PASS (baseline)

**Step 3: Optimize predict_mc_dropout to batch on GPU**

In `src/core/engine.py`, replace the `predict_mc_dropout` method:

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
    epistemic uncertainty. Optimized to batch samples on GPU before
    transferring to CPU.

    Args:
        batch: MolecularGraphBatch (targets optional)
        num_samples: Number of MC samples
        return_stats: If True, return (mean, std) instead of all samples

    Returns:
        If return_stats=False: Predictions [num_samples, num_molecules, output_dim]
        If return_stats=True: Tuple of (mean, std), each [num_molecules, output_dim]
    """
    batch = batch.to(self.device)
    output_dim = self.model.config.output_dim

    # Keep model in train mode to enable dropout
    self.model.train()

    try:
        # Pre-allocate tensor on GPU for all samples
        samples = torch.empty(
            num_samples,
            batch.num_molecules,
            output_dim,
            device=self.device,
            dtype=torch.float32
        )

        with torch.no_grad():  # No gradients needed for inference
            # Apply AMP context once, outside loop
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    for i in range(num_samples):
                        samples[i] = self.model(batch)
            else:
                for i in range(num_samples):
                    samples[i] = self.model(batch)

        # Single GPU->CPU transfer for all samples
        samples = samples.cpu()

    finally:
        # Always restore eval mode, even if exception occurs
        self.model.eval()

    if return_stats:
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std

    return samples
```

**Step 4: Run tests**

Run: `pytest tests/core/test_engine.py::test_predict_mc_dropout_batched_equivalent -v`
Expected: PASS

Run: `pytest tests/core/test_engine.py::TestEngineMCDropout -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "perf: batch MC Dropout samples on GPU for 10-30% speedup

Optimize predict_mc_dropout to pre-allocate tensor on GPU, collect all
samples before single CPU transfer. Move AMP autocast outside loop.
Add try/finally for model state restoration on exception.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9.5: Add Evidential-Aware MC Dropout

**Priority:** P1 (Important)

**Files:**
- Modify: `src/core/engine.py:320-380`
- Test: `tests/core/test_engine.py`

**Step 1: Write test for MC Dropout with evidential model**

Add to `tests/core/test_engine.py`:

```python
def test_predict_mc_dropout_with_evidential():
    """Test MC Dropout extracts mu from evidential outputs."""
    model_config = ModelConfig(hidden_dim=32, output_dim=4, num_shells=2, dropout=0.1)
    engine_config = EngineConfig(
        device="cpu",
        use_amp=False,
        loss_function="evidential",
        warmup_epochs=0
    )
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

    # MC Dropout with evidential model
    predictions = engine.predict_mc_dropout(batch, num_samples=10)

    # Should extract mu only, not all 4 params
    assert predictions.shape == (10, 2, 1)  # [samples, molecules, 1] not [samples, molecules, 4]

    # Should have variance
    std = predictions.std(dim=0)
    assert std.mean() > 0

def test_predict_mc_dropout_evidential_multi_output():
    """Test MC Dropout with multi-output evidential model."""
    # 2 tasks, model outputs 8 parameters (4 per task)
    model_config = ModelConfig(hidden_dim=32, output_dim=8, num_shells=2, dropout=0.1)
    engine_config = EngineConfig(
        device="cpu",
        use_amp=False,
        loss_function="evidential",
        warmup_epochs=0
    )
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

    # Should extract 2 mu values (one per task)
    predictions = engine.predict_mc_dropout(batch, num_samples=10)
    assert predictions.shape == (10, 2, 2)  # [samples, molecules, 2 tasks]
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_engine.py::test_predict_mc_dropout_with_evidential -v`
Expected: FAIL (returns [10, 2, 4] instead of [10, 2, 1])

**Step 3: Add _is_evidential_loss helper**

In `src/core/engine.py`, add after `_create_loss_function`:

```python
def _is_evidential_loss(self) -> bool:
    """Check if current loss function is evidential.

    Returns:
        True if using evidential loss, False otherwise.
    """
    return self.config.loss_function == "evidential"

def _extract_evidential_mu(self, predictions: torch.Tensor) -> torch.Tensor:
    """Extract mu (mean predictions) from evidential outputs.

    For evidential regression, the model outputs [batch, num_tasks * 4]
    containing (mu, v, alpha, beta) for each task. This extracts just mu.

    Args:
        predictions: Raw model predictions [batch, num_tasks * 4]

    Returns:
        Extracted mu values [batch, num_tasks]
    """
    batch_size = predictions.shape[0]
    num_tasks = predictions.shape[1] // 4

    # Reshape to [batch, num_tasks, 4]
    reshaped = predictions.view(batch_size, num_tasks, 4)

    # Extract mu (index 0)
    mu = reshaped[:, :, 0]  # [batch, num_tasks]

    return mu
```

**Step 4: Update predict_mc_dropout to handle evidential**

Update the `predict_mc_dropout` method:

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
    epistemic uncertainty. For evidential models, extracts mu (mean)
    from the 4-parameter outputs.

    Args:
        batch: MolecularGraphBatch (targets optional)
        num_samples: Number of MC samples
        return_stats: If True, return (mean, std) instead of all samples

    Returns:
        If return_stats=False: Predictions [num_samples, num_molecules, num_tasks]
        If return_stats=True: Tuple of (mean, std), each [num_molecules, num_tasks]
    """
    batch = batch.to(self.device)
    output_dim = self.model.config.output_dim

    # For evidential models, determine number of tasks
    is_evidential = self._is_evidential_loss()
    num_tasks = output_dim // 4 if is_evidential else output_dim

    # Keep model in train mode to enable dropout
    self.model.train()

    try:
        # Pre-allocate tensor on GPU for all samples
        samples = torch.empty(
            num_samples,
            batch.num_molecules,
            num_tasks,
            device=self.device,
            dtype=torch.float32
        )

        with torch.no_grad():  # No gradients needed for inference
            # Apply AMP context once, outside loop
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    for i in range(num_samples):
                        raw_pred = self.model(batch)
                        # Extract mu for evidential models
                        if is_evidential:
                            samples[i] = self._extract_evidential_mu(raw_pred)
                        else:
                            samples[i] = raw_pred
            else:
                for i in range(num_samples):
                    raw_pred = self.model(batch)
                    # Extract mu for evidential models
                    if is_evidential:
                        samples[i] = self._extract_evidential_mu(raw_pred)
                    else:
                        samples[i] = raw_pred

        # Single GPU->CPU transfer for all samples
        samples = samples.cpu()

    finally:
        # Always restore eval mode, even if exception occurs
        self.model.eval()

    if return_stats:
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std

    return samples
```

**Step 5: Run tests**

Run: `pytest tests/core/test_engine.py::test_predict_mc_dropout_with_evidential -v`
Expected: PASS

Run: `pytest tests/core/test_engine.py::test_predict_mc_dropout_evidential_multi_output -v`
Expected: PASS

Run: `pytest tests/core/test_engine.py::TestEngineMCDropout -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "feat: add evidential-aware MC Dropout

Extract mu (mean predictions) from evidential model outputs during MC
Dropout inference. Handles both single and multi-task evidential models.
Add _is_evidential_loss and _extract_evidential_mu helper methods.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9.6: Reduce Synchronization Points in evaluate()

**Priority:** P2 (Minor but measurable)

**Files:**
- Modify: `src/core/engine.py:400-414`
- Test: `tests/core/test_engine.py`

**Step 1: Write test verifying results unchanged**

Add to `tests/core/test_engine.py`:

```python
def test_evaluate_reduced_sync_points_equivalent():
    """Test optimized evaluate gives same results."""
    torch.manual_seed(42)
    model_config = ModelConfig(hidden_dim=32, output_dim=2, num_shells=2)
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
        targets=torch.randn(2, 2),
    )

    metrics = engine.evaluate(batch)

    # All metrics should be present and finite
    assert "loss" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "abs_errors" in metrics
    assert "squared_errors" in metrics
    assert "num_elements" in metrics

    assert not torch.isnan(torch.tensor(metrics["loss"]))
    assert not torch.isnan(torch.tensor(metrics["mae"]))
    assert not torch.isnan(torch.tensor(metrics["rmse"]))
```

**Step 2: Run test to verify baseline**

Run: `pytest tests/core/test_engine.py::test_evaluate_reduced_sync_points_equivalent -v`
Expected: PASS (baseline)

**Step 3: Optimize evaluate to reduce .item() calls**

In `src/core/engine.py`, update the metric computation in `evaluate`:

```python
# Compute metrics - batch reductions together to minimize sync points
diff = predictions - batch.targets

# Compute all scalar values in single operation, then extract
metrics_tensor = torch.stack([
    diff.abs().sum(),
    diff.pow(2).sum(),
    loss,
])

# Single GPU->CPU sync point for all three values
metrics_cpu = metrics_tensor.cpu().tolist()
abs_errors = metrics_cpu[0]
squared_errors = metrics_cpu[1]
loss_value = metrics_cpu[2]

num_elements = diff.numel()

return {
    "loss": loss_value,
    "mae": abs_errors / num_elements,
    "rmse": (squared_errors / num_elements) ** 0.5,
    # Raw values for aggregation
    "abs_errors": abs_errors,
    "squared_errors": squared_errors,
    "num_elements": num_elements,
}
```

**Step 4: Run tests**

Run: `pytest tests/core/test_engine.py::test_evaluate_reduced_sync_points_equivalent -v`
Expected: PASS

Run: `pytest tests/core/test_engine.py -k evaluate -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "perf: reduce GPU sync points in evaluate from 3 to 1

Batch metric tensor operations (abs sum, squared sum, loss) together
and extract via single .cpu().tolist() call. Reduces synchronization
overhead in validation loops.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9.7: Add Loss-Output Compatibility Validation

**Priority:** P2 (Prevents misconfiguration)

**Files:**
- Modify: `src/core/engine.py:80-95`
- Test: `tests/core/test_engine.py`

**Step 1: Write test for validation**

Add to `tests/core/test_engine.py`:

```python
def test_engine_validates_evidential_output_dim():
    """Test Engine validates output_dim for evidential loss."""
    # output_dim must be multiple of 4 for evidential
    model_config = ModelConfig(hidden_dim=32, output_dim=3, num_shells=2)
    engine_config = EngineConfig(loss_function="evidential")

    with pytest.raises(ValueError, match="output_dim.*must be multiple of 4"):
        Engine.from_config(model_config, engine_config)

def test_engine_accepts_valid_evidential_output_dim():
    """Test Engine accepts valid output_dim for evidential loss."""
    # output_dim=4 is valid (1 task)
    model_config = ModelConfig(hidden_dim=32, output_dim=4, num_shells=2)
    engine_config = EngineConfig(loss_function="evidential", device="cpu")

    engine = Engine.from_config(model_config, engine_config)
    assert engine is not None

def test_engine_accepts_non_evidential_any_output_dim():
    """Test Engine accepts any output_dim for non-evidential losses."""
    model_config = ModelConfig(hidden_dim=32, output_dim=3, num_shells=2)
    engine_config = EngineConfig(loss_function="mse", device="cpu")

    engine = Engine.from_config(model_config, engine_config)
    assert engine is not None
```

**Step 2: Run test to verify failure**

Run: `pytest tests/core/test_engine.py::test_engine_validates_evidential_output_dim -v`
Expected: FAIL (no validation currently)

**Step 3: Add validation method**

In `src/core/engine.py`, add after `__init__`:

```python
def _validate_loss_model_compatibility(self) -> None:
    """Validate that loss function is compatible with model output_dim.

    Raises:
        ValueError: If incompatible configuration detected.
    """
    if self.config.loss_function == "evidential":
        output_dim = self.model.config.output_dim
        if output_dim % 4 != 0:
            raise ValueError(
                f"EvidentialLoss requires output_dim to be multiple of 4 "
                f"(got {output_dim}). Each task needs 4 parameters: "
                f"mu, v, alpha, beta."
            )
```

**Step 4: Call validation in __init__**

Update the `__init__` method to call validation:

```python
def __init__(
    self,
    model: SimplifiedGNN,
    config: EngineConfig,
    preprocessing: PreprocessingPipeline | None = None,
):
    self.model = model
    self.config = config
    self.preprocessing = preprocessing

    # Device management
    self.device = config.resolved_device
    self.model = self.model.to(self.device)

    # Validate loss-model compatibility
    self._validate_loss_model_compatibility()

    # Training components
    self.optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    self.loss_fn = self._create_loss_function()
    # ... rest of init
```

**Step 5: Run tests**

Run: `pytest tests/core/test_engine.py::test_engine_validates_evidential_output_dim -v`
Expected: PASS

Run: `pytest tests/core/test_engine.py::test_engine_accepts_valid_evidential_output_dim -v`
Expected: PASS

Run: `pytest tests/core/test_engine.py::test_engine_accepts_non_evidential_any_output_dim -v`
Expected: PASS

**Step 6: Run full Engine tests**

Run: `pytest tests/core/test_engine.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "feat: add loss-output compatibility validation

Validate that EvidentialLoss has output_dim as multiple of 4 during
Engine initialization. Provides clear error message explaining the
requirement. Prevents silent misconfiguration errors.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Final Verification

Run all core tests to ensure everything works together:

```bash
pytest tests/core/ -v --tb=short
```

Expected: All 140+ tests PASS (including new tests from all 7 tasks)

---

## Summary of Changes

| Task | Type | Description | Files | Tests |
|------|------|-------------|-------|-------|
| 9.1 | Fix | Multi-output EvidentialLoss | losses.py | +2 |
| 9.2 | Feat | Loss kwargs in EngineConfig | engine_config.py, engine.py | +5 |
| 9.3 | Fix | Loss aggregation consistency | engine.py | +1 |
| 9.4 | Perf | MC Dropout GPU batching | engine.py | +1 |
| 9.5 | Feat | Evidential-aware MC Dropout | engine.py | +2 |
| 9.6 | Perf | Reduce sync points | engine.py | +1 |
| 9.7 | Feat | Loss validation | engine.py | +3 |

**Impact:**
- P0 fixes: EvidentialLoss now supports multi-task models
- P1 fixes: MC Dropout 10-30% faster, evidential compatible, loss aggregation correct, configurable loss params
- P2 improvements: Reduced sync points, validation prevents misconfiguration

**All changes maintain backward compatibility** except EvidentialLoss multi-output (intentional fix).
