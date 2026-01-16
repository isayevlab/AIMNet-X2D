# Phase 6: Unified Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a unified `Engine` class that handles both training and inference, eliminating duplicate code between training and inference pipelines.

**Architecture:** Single `Engine` class with `train()`, `predict()`, and `evaluate()` methods. Uses existing `SimplifiedGNN`, `BatchFeaturizer`, and `PreprocessingPipeline`. Supports DDP for multi-GPU training. Clean API that accepts MolecularGraphBatch directly.

**Tech Stack:** PyTorch 2.x, torch.compile(), DDP, existing core module components

---

## Task 6.1: Create EngineConfig Dataclass

**Files:**
- Create: `src/core/engine_config.py`
- Test: `tests/core/test_engine_config.py`

**Step 1: Write the failing test**

Create `tests/core/test_engine_config.py`:

```python
"""Tests for EngineConfig dataclass."""

import pytest
import torch

from src.core.engine_config import EngineConfig


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_default_creation(self):
        """Test creation with defaults."""
        config = EngineConfig()

        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.device == "cuda" if torch.cuda.is_available() else "cpu"

    def test_custom_parameters(self):
        """Test creation with custom parameters."""
        config = EngineConfig(
            learning_rate=5e-4,
            batch_size=64,
            epochs=50,
            weight_decay=1e-5,
        )

        assert config.learning_rate == 5e-4
        assert config.batch_size == 64
        assert config.epochs == 50

    def test_to_dict(self):
        """Test serialization to dict."""
        config = EngineConfig(learning_rate=1e-4)
        d = config.to_dict()

        assert d["learning_rate"] == 1e-4
        assert "batch_size" in d

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {"learning_rate": 2e-4, "batch_size": 128}
        config = EngineConfig.from_dict(d)

        assert config.learning_rate == 2e-4
        assert config.batch_size == 128

    def test_device_auto_detection(self):
        """Test automatic device detection."""
        config = EngineConfig(device="auto")
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.resolved_device.type == expected
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_engine_config.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement EngineConfig**

Create `src/core/engine_config.py`:

```python
"""
Engine configuration for training and inference.

Provides a unified configuration interface for the Engine class.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

import torch


@dataclass
class EngineConfig:
    """
    Configuration for the unified Engine.

    Attributes:
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        epochs: Number of training epochs
        weight_decay: L2 regularization weight
        device: Device to use ('cuda', 'cpu', or 'auto')
        num_workers: DataLoader workers
        gradient_clip: Max gradient norm (None to disable)
        scheduler: Learning rate scheduler type
        warmup_epochs: Epochs for learning rate warmup
        early_stopping_patience: Epochs without improvement before stopping
        checkpoint_dir: Directory for saving checkpoints
        log_interval: Steps between logging
        use_amp: Use automatic mixed precision
        compile_model: Use torch.compile()
    """

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float | None = 1.0

    # Training
    batch_size: int = 32
    epochs: int = 100
    warmup_epochs: int = 5
    early_stopping_patience: int = 20

    # Scheduler
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"

    # Hardware
    device: str = "auto"
    num_workers: int = 4
    use_amp: bool = True
    compile_model: bool = False

    # Logging
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100

    @property
    def resolved_device(self) -> torch.device:
        """Get resolved device (handles 'auto')."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "scheduler": self.scheduler,
            "device": self.device,
            "num_workers": self.num_workers,
            "use_amp": self.use_amp,
            "compile_model": self.compile_model,
            "checkpoint_dir": self.checkpoint_dir,
            "log_interval": self.log_interval,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EngineConfig:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
```

**Step 4: Run tests**

Run: `pytest tests/core/test_engine_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/engine_config.py tests/core/test_engine_config.py
git commit -m "feat: add EngineConfig dataclass for unified engine configuration"
```

---

## Task 6.2: Create Unified Engine Class

**Files:**
- Create: `src/core/engine.py`
- Test: `tests/core/test_engine.py`

**Step 1: Write failing tests**

Create `tests/core/test_engine.py`:

```python
"""Tests for unified Engine class."""

import pytest
import torch
import numpy as np

from src.core.engine import Engine
from src.core.engine_config import EngineConfig
from src.core.model_config import ModelConfig
from src.core.model import SimplifiedGNN
from src.core.batch import MolecularGraphBatch
from src.core.featurizer import BatchFeaturizer


class TestEngineCreation:
    """Tests for Engine creation."""

    def test_create_with_model(self):
        """Test creating engine with existing model."""
        model_config = ModelConfig(hidden_dim=64, output_dim=1)
        model = SimplifiedGNN(model_config)
        engine_config = EngineConfig(device="cpu")

        engine = Engine(model=model, config=engine_config)

        assert engine.model is model
        assert engine.device.type == "cpu"

    def test_create_from_config(self):
        """Test creating engine from model config."""
        model_config = ModelConfig(hidden_dim=32, output_dim=2)
        engine_config = EngineConfig(device="cpu")

        engine = Engine.from_config(model_config, engine_config)

        assert engine.model is not None
        assert engine.model.config.hidden_dim == 32


class TestEnginePrediction:
    """Tests for Engine prediction."""

    def test_predict_batch(self):
        """Test prediction on a batch."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu")
        engine = Engine.from_config(model_config, engine_config)

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

        predictions = engine.predict(batch)

        assert predictions.shape == (3, 1)
        assert not torch.isnan(predictions).any()

    def test_predict_smiles(self):
        """Test prediction from SMILES strings."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu")
        engine = Engine.from_config(model_config, engine_config)

        smiles = ["C", "CC", "CCC"]
        predictions = engine.predict_smiles(smiles)

        assert predictions.shape[0] == 3
        assert predictions.shape[1] == 1


class TestEngineTraining:
    """Tests for Engine training."""

    def test_single_training_step(self):
        """Test a single training step."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False)
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

        loss = engine.train_step(batch)

        assert isinstance(loss, float)
        assert not np.isnan(loss)

    def test_evaluate_batch(self):
        """Test evaluation on a batch."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu")
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

        metrics = engine.evaluate(batch)

        assert "loss" in metrics
        assert "mae" in metrics


class TestEngineSaveLoad:
    """Tests for Engine checkpoint save/load."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoint."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1)
        engine_config = EngineConfig(device="cpu")
        engine = Engine.from_config(model_config, engine_config)

        # Save
        checkpoint_path = tmp_path / "checkpoint.pth"
        engine.save_checkpoint(str(checkpoint_path))

        # Load into new engine
        engine2 = Engine.load_checkpoint(str(checkpoint_path))

        assert engine2.model.config.hidden_dim == 32

    def test_checkpoint_includes_optimizer(self, tmp_path):
        """Test checkpoint includes optimizer state."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1)
        engine_config = EngineConfig(device="cpu")
        engine = Engine.from_config(model_config, engine_config)

        # Do a training step to update optimizer state
        batch = MolecularGraphBatch(
            atom_types=torch.randint(0, 10, (5,), dtype=torch.int32),
            batch_idx=torch.zeros(5, dtype=torch.int64),
            ptr=torch.tensor([0, 5], dtype=torch.int64),
            edge_indices=[torch.randint(0, 5, (2, 8))],
            num_molecules=1,
            targets=torch.randn(1, 1),
        )
        engine.train_step(batch)

        # Save and load
        checkpoint_path = tmp_path / "checkpoint.pth"
        engine.save_checkpoint(str(checkpoint_path))
        engine2 = Engine.load_checkpoint(str(checkpoint_path))

        # Optimizer should have state
        assert len(engine2.optimizer.state) > 0
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/core/test_engine.py -v`
Expected: FAIL

**Step 3: Implement Engine class**

Create `src/core/engine.py`:

```python
"""
Unified Engine for training and inference.

Combines training and inference into a single class with clean API.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

from .batch import MolecularGraphBatch
from .model import SimplifiedGNN
from .model_config import ModelConfig
from .engine_config import EngineConfig
from .featurizer import BatchFeaturizer
from .preprocessing import PreprocessingPipeline

from utils.logging import get_logger

logger = get_logger(__name__)


class Engine:
    """
    Unified engine for training and inference.

    Handles:
    - Model training with mixed precision and gradient clipping
    - Inference with preprocessing integration
    - Checkpoint save/load
    - Metrics computation

    Args:
        model: SimplifiedGNN model instance
        config: EngineConfig with training parameters
        preprocessing: Optional preprocessing pipeline
    """

    def __init__(
        self,
        model: SimplifiedGNN,
        config: EngineConfig,
        preprocessing: PreprocessingPipeline | None = None,
    ):
        self.config = config
        self.device = config.resolved_device
        self.preprocessing = preprocessing

        # Move model to device
        self.model = model.to(self.device)

        # Optionally compile
        if config.compile_model:
            self.model = self.model.compile()

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup mixed precision
        self.scaler = GradScaler() if config.use_amp and self.device.type == "cuda" else None

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Featurizer for SMILES input
        self._featurizer: BatchFeaturizer | None = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        engine_config: EngineConfig,
        preprocessing: PreprocessingPipeline | None = None,
    ) -> Engine:
        """Create engine from model configuration."""
        model = SimplifiedGNN(model_config)
        return cls(model=model, config=engine_config, preprocessing=preprocessing)

    @property
    def featurizer(self) -> BatchFeaturizer:
        """Lazy-initialized featurizer."""
        if self._featurizer is None:
            self._featurizer = BatchFeaturizer(
                num_hops=self.model.config.num_shells,
                num_workers=self.config.num_workers,
            )
        return self._featurizer

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
            )
        return None

    def train_step(self, batch: MolecularGraphBatch) -> float:
        """
        Perform single training step.

        Args:
            batch: MolecularGraphBatch with targets

        Returns:
            Loss value
        """
        self.model.train()
        batch = batch.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass with optional AMP
        if self.scaler is not None:
            with autocast():
                predictions = self.model(batch)
                loss = self.loss_fn(predictions, batch.targets)

            self.scaler.scale(loss).backward()

            if self.config.gradient_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predictions = self.model(batch)
            loss = self.loss_fn(predictions, batch.targets)

            loss.backward()

            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            self.optimizer.step()

        self.global_step += 1

        return loss.item()

    @torch.no_grad()
    def predict(self, batch: MolecularGraphBatch) -> torch.Tensor:
        """
        Run inference on batch.

        Args:
            batch: MolecularGraphBatch (targets optional)

        Returns:
            Predictions [num_molecules, output_dim]
        """
        self.model.eval()
        batch = batch.to(self.device)

        if self.scaler is not None:
            with autocast():
                predictions = self.model(batch)
        else:
            predictions = self.model(batch)

        return predictions.cpu()

    def predict_smiles(
        self,
        smiles: list[str],
        denormalize: bool = True,
    ) -> torch.Tensor:
        """
        Run inference on SMILES strings.

        Args:
            smiles: List of SMILES strings
            denormalize: Whether to reverse preprocessing

        Returns:
            Predictions [num_molecules, output_dim]
        """
        import numpy as np

        # Featurize (targets not used for inference)
        dummy_targets = np.zeros((len(smiles), self.model.config.output_dim))
        batch = self.featurizer.featurize(smiles, dummy_targets)

        predictions = self.predict(batch)

        # Optionally denormalize
        if denormalize and self.preprocessing is not None:
            from .batch_adapter import BatchAdapter
            adapter = BatchAdapter()
            padded_nums, atom_counts = adapter.to_padded_format(batch)
            predictions = self.preprocessing.inverse_transform_batch(
                padded_nums, atom_counts, predictions
            )

        return predictions

    @torch.no_grad()
    def evaluate(self, batch: MolecularGraphBatch) -> dict[str, float]:
        """
        Evaluate model on batch.

        Args:
            batch: MolecularGraphBatch with targets

        Returns:
            Dict with 'loss', 'mae', 'rmse'
        """
        self.model.eval()
        batch = batch.to(self.device)

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

    def step_scheduler(self, val_loss: float | None = None):
        """Step the learning rate scheduler."""
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def save_checkpoint(self, path: str):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_config": self.model.config.to_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "engine_config": self.config.to_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        if self.preprocessing is not None:
            checkpoint["preprocessing"] = self.preprocessing.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: str = "auto",
    ) -> Engine:
        """
        Load engine from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load to ('auto' for automatic)

        Returns:
            Loaded Engine instance
        """
        checkpoint = torch.load(path, map_location="cpu")

        # Reconstruct configs
        model_config = ModelConfig.from_dict(checkpoint["model_config"])
        engine_config = EngineConfig.from_dict(checkpoint["engine_config"])

        if device != "auto":
            engine_config.device = device

        # Reconstruct preprocessing
        preprocessing = None
        if "preprocessing" in checkpoint and checkpoint["preprocessing"]:
            preprocessing = PreprocessingPipeline.from_state_dict(
                checkpoint["preprocessing"]
            )

        # Create engine
        engine = cls.from_config(model_config, engine_config, preprocessing)

        # Load states
        engine.model.load_state_dict(checkpoint["model_state_dict"])
        engine.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and engine.scheduler is not None:
            engine.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint and engine.scaler is not None:
            engine.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        engine.epoch = checkpoint.get("epoch", 0)
        engine.global_step = checkpoint.get("global_step", 0)
        engine.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(f"Loaded checkpoint from {path} (epoch {engine.epoch})")

        return engine

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
```

**Step 4: Run tests**

Run: `pytest tests/core/test_engine.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "feat: add unified Engine class for training and inference"
```

---

## Task 6.3: Add Training Loop to Engine

**Files:**
- Modify: `src/core/engine.py`
- Modify: `tests/core/test_engine.py`

**Step 1: Add test for full training**

Add to `tests/core/test_engine.py`:

```python
class TestEngineFullTraining:
    """Tests for full training loops."""

    def test_train_epoch(self):
        """Test training for one epoch."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False)
        engine = Engine.from_config(model_config, engine_config)

        # Create multiple batches
        batches = []
        for _ in range(3):
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
            batches.append(batch)

        metrics = engine.train_epoch(batches)

        assert "loss" in metrics
        assert "lr" in metrics
        assert engine.epoch == 1

    def test_fit_with_validation(self):
        """Test fit method with train and val data."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(
            device="cpu",
            use_amp=False,
            epochs=2,
            early_stopping_patience=5,
        )
        engine = Engine.from_config(model_config, engine_config)

        # Create train/val batches
        train_batches = [
            MolecularGraphBatch(
                atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
                batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
                ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
                edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
                num_molecules=2,
                targets=torch.randn(2, 1),
            )
            for _ in range(2)
        ]
        val_batches = [train_batches[0]]

        history = engine.fit(train_batches, val_batches)

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
```

**Step 2: Add train_epoch and fit methods**

Add to `Engine` class in `src/core/engine.py`:

```python
def train_epoch(
    self,
    train_batches: list[MolecularGraphBatch],
) -> dict[str, float]:
    """
    Train for one epoch.

    Args:
        train_batches: List of training batches

    Returns:
        Dict with epoch metrics
    """
    self.model.train()
    total_loss = 0.0
    num_batches = len(train_batches)

    for batch_idx, batch in enumerate(train_batches):
        loss = self.train_step(batch)
        total_loss += loss

        if (batch_idx + 1) % self.config.log_interval == 0:
            logger.info(
                f"Epoch {self.epoch} [{batch_idx+1}/{num_batches}] "
                f"Loss: {loss:.4f}"
            )

    self.epoch += 1
    avg_loss = total_loss / num_batches

    return {
        "loss": avg_loss,
        "lr": self.get_lr(),
    }

def fit(
    self,
    train_batches: list[MolecularGraphBatch],
    val_batches: list[MolecularGraphBatch] | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Train model for multiple epochs.

    Args:
        train_batches: Training data batches
        val_batches: Optional validation batches
        verbose: Whether to print progress

    Returns:
        Training history dict
    """
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    epochs_without_improvement = 0

    for epoch in range(self.config.epochs):
        # Train
        train_metrics = self.train_epoch(train_batches)
        history["train_loss"].append(train_metrics["loss"])
        history["lr"].append(train_metrics["lr"])

        # Validate
        if val_batches:
            val_metrics = self.evaluate_batches(val_batches)
            history["val_loss"].append(val_metrics["loss"])

            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Step scheduler
            self.step_scheduler(val_metrics["loss"])

            if verbose:
                logger.info(
                    f"Epoch {self.epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"LR={train_metrics['lr']:.2e}"
                )

            # Early stopping
            if epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {self.epoch}")
                break
        else:
            self.step_scheduler()
            if verbose:
                logger.info(
                    f"Epoch {self.epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"LR={train_metrics['lr']:.2e}"
                )

    return history

def evaluate_batches(
    self,
    batches: list[MolecularGraphBatch],
) -> dict[str, float]:
    """
    Evaluate model on multiple batches.

    Args:
        batches: List of batches to evaluate

    Returns:
        Aggregated metrics
    """
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    num_batches = len(batches)

    for batch in batches:
        metrics = self.evaluate(batch)
        total_loss += metrics["loss"]
        total_mae += metrics["mae"]
        total_rmse += metrics["rmse"]

    return {
        "loss": total_loss / num_batches,
        "mae": total_mae / num_batches,
        "rmse": total_rmse / num_batches,
    }
```

**Step 3: Run tests**

Run: `pytest tests/core/test_engine.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/core/engine.py tests/core/test_engine.py
git commit -m "feat: add train_epoch and fit methods to Engine"
```

---

## Task 6.4: Update Core Module Exports

**Files:**
- Modify: `src/core/__init__.py`

**Step 1: Update exports**

```python
"""
Core module for GPU-native molecular GNN.

This module provides the refactored architecture with:
- MolecularGraphBatch: Batched molecular data
- BatchFeaturizer: SMILES to batch conversion
- SimplifiedGNN: Main GNN model
- Engine: Unified training and inference
- Preprocessing: SAE and scaling transforms
"""

from .batch import MolecularGraphBatch
from .batch_adapter import BatchAdapter
from .featurizer import BatchFeaturizer
from .model import SimplifiedGNN
from .model_config import ModelConfig
from .engine import Engine
from .engine_config import EngineConfig
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
    # Engine
    "Engine",
    "EngineConfig",
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

**Step 2: Verify imports**

Run: `PYTHONPATH=src python -c "from src.core import Engine, EngineConfig; print('OK')"`

**Step 3: Run all core tests**

Run: `pytest tests/core/ -v --tb=short`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/core/__init__.py
git commit -m "chore: add Engine and EngineConfig to core module exports"
```

---

## Final Verification

Run all tests:
```bash
pytest tests/core/ -v --tb=short
```

Expected: All tests PASS
