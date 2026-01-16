"""Tests for unified Engine class."""

import pytest
import torch
import numpy as np

from src.core.engine import Engine
from src.core.engine_config import EngineConfig
from src.core.model_config import ModelConfig
from src.core.model import SimplifiedGNN
from src.core.batch import MolecularGraphBatch


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


class TestEngineScheduler:
    """Tests for Engine scheduler functionality."""

    def test_cosine_scheduler(self):
        """Test cosine scheduler creation."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1)
        engine_config = EngineConfig(device="cpu", scheduler="cosine", epochs=100)
        engine = Engine.from_config(model_config, engine_config)

        assert engine.scheduler is not None
        initial_lr = engine.get_lr()
        engine.step_scheduler()
        # After one step, LR should have changed
        assert engine.get_lr() != initial_lr or engine.config.epochs == 1

    def test_plateau_scheduler(self):
        """Test plateau scheduler creation."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1)
        engine_config = EngineConfig(device="cpu", scheduler="plateau")
        engine = Engine.from_config(model_config, engine_config)

        assert engine.scheduler is not None
        # Plateau scheduler needs val_loss
        engine.step_scheduler(val_loss=1.0)

    def test_no_scheduler(self):
        """Test no scheduler option."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1)
        engine_config = EngineConfig(device="cpu", scheduler="none")
        engine = Engine.from_config(model_config, engine_config)

        assert engine.scheduler is None
        # Should not raise when stepping
        engine.step_scheduler()


class TestEngineGradientClipping:
    """Tests for Engine gradient clipping."""

    def test_gradient_clipping_applied(self):
        """Test that gradient clipping is applied during training."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False, gradient_clip=1.0)
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
            targets=torch.randn(2, 1) * 100,  # Large targets to cause large gradients
        )

        # Should not raise even with large targets
        loss = engine.train_step(batch)
        assert not np.isnan(loss)


class TestEngineTrainingState:
    """Tests for Engine training state tracking."""

    def test_global_step_increments(self):
        """Test that global_step increments with each training step."""
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

        assert engine.global_step == 0
        engine.train_step(batch)
        assert engine.global_step == 1
        engine.train_step(batch)
        assert engine.global_step == 2

    def test_checkpoint_preserves_state(self, tmp_path):
        """Test that checkpoint preserves training state."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False)
        engine = Engine.from_config(model_config, engine_config)

        # Modify state
        engine.epoch = 10
        engine.global_step = 500
        engine.best_val_loss = 0.5

        # Save and load
        checkpoint_path = tmp_path / "checkpoint.pth"
        engine.save_checkpoint(str(checkpoint_path))
        engine2 = Engine.load_checkpoint(str(checkpoint_path))

        assert engine2.epoch == 10
        assert engine2.global_step == 500
        assert engine2.best_val_loss == 0.5


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
            warmup_epochs=0,
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

    def test_evaluate_batches(self):
        """Test evaluating multiple batches."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu")
        engine = Engine.from_config(model_config, engine_config)

        batches = [
            MolecularGraphBatch(
                atom_types=torch.randint(0, 10, (10,), dtype=torch.int32),
                batch_idx=torch.tensor([0]*5 + [1]*5, dtype=torch.int64),
                ptr=torch.tensor([0, 5, 10], dtype=torch.int64),
                edge_indices=[torch.randint(0, 10, (2, 15), dtype=torch.int64)],
                num_molecules=2,
                targets=torch.randn(2, 1),
            )
            for _ in range(3)
        ]

        metrics = engine.evaluate_batches(batches)

        assert "loss" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics

    def test_early_stopping(self):
        """Test that early stopping works."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(
            device="cpu",
            use_amp=False,
            epochs=100,  # High epoch count
            early_stopping_patience=2,  # Short patience
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

        # Should stop before 100 epochs (patience 2 means stop after 3 epochs without improvement)
        assert len(history["train_loss"]) < 100


class TestEngineWeightedMetrics:
    """Tests for weighted metric aggregation."""

    def test_evaluate_batches_weights_by_size(self):
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


class TestEngineWarmup:
    """Tests for Engine learning rate warmup."""

    def test_warmup_scheduler(self):
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

    def test_no_warmup_scheduler(self):
        """Test that warmup_epochs=0 doesn't add warmup."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(
            device="cpu",
            scheduler="cosine",
            warmup_epochs=0,
            epochs=10,
        )
        engine = Engine.from_config(model_config, engine_config)

        # Should be CosineAnnealingLR, not SequentialLR
        assert isinstance(engine.scheduler, CosineAnnealingLR)

    def test_warmup_invalid_config(self):
        """Test that warmup_epochs >= epochs raises error."""
        with pytest.raises(ValueError, match="warmup_epochs"):
            EngineConfig(warmup_epochs=10, epochs=5)


class TestEngineLossFunctions:
    """Tests for Engine loss function configuration."""

    def test_engine_with_mae_loss(self):
        """Test Engine with MAE loss function."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False, loss_function="mae")
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
        assert loss >= 0  # MAE is always non-negative

    def test_engine_with_huber_loss(self):
        """Test Engine with Huber loss function."""
        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=False, loss_function="huber")
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


class TestEngineGradientAccumulation:
    """Tests for Engine gradient accumulation."""

    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple steps."""
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

    def test_gradient_accumulation_invalid_zero_steps(self):
        """Test that zero accumulation_steps raises error."""
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

        with pytest.raises(ValueError, match="accumulation_steps must be positive"):
            engine.train_step_accumulated(batch, accumulation_step=0, accumulation_steps=0)

    def test_gradient_accumulation_invalid_step(self):
        """Test that invalid accumulation_step raises error."""
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

        with pytest.raises(ValueError, match="accumulation_step must be in"):
            engine.train_step_accumulated(batch, accumulation_step=5, accumulation_steps=4)

    def test_gradient_accumulation_optimizer_steps_once(self):
        """Test optimizer only steps at the end of accumulation."""
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

        initial_global_step = engine.global_step
        accumulation_steps = 4

        for step in range(accumulation_steps):
            engine.train_step_accumulated(batch, step, accumulation_steps)

        # Global step should only increment once (at the final accumulation step)
        assert engine.global_step == initial_global_step + 1


class TestEngineAMPConsistency:
    """Tests for AMP consistency across methods."""

    def test_evaluate_uses_amp_when_enabled(self):
        """Test that evaluate uses AMP autocast like predict."""
        import unittest.mock as mock

        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=True, warmup_epochs=0)
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

    def test_evaluate_calls_autocast_when_scaler_present(self):
        """Test that evaluate() calls torch.amp.autocast when scaler is present."""
        import unittest.mock as mock

        model_config = ModelConfig(hidden_dim=32, output_dim=1, num_shells=2)
        engine_config = EngineConfig(device="cpu", use_amp=True, warmup_epochs=0)
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

        # Patch torch.amp.autocast to track if it's called
        with mock.patch("torch.amp.autocast") as mock_autocast:
            # Set up the mock to work as a context manager
            mock_autocast.return_value.__enter__ = mock.MagicMock()
            mock_autocast.return_value.__exit__ = mock.MagicMock(return_value=False)

            engine.evaluate(batch)

            # Verify autocast was called with "cuda"
            mock_autocast.assert_called_once_with("cuda")


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
