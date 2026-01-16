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
