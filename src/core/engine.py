"""
Unified Engine for training and inference.

Combines training and inference into a single class with clean API.
"""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .batch import MolecularGraphBatch
from .model import SimplifiedGNN
from .model_config import ModelConfig
from .engine_config import EngineConfig
from .featurizer import BatchFeaturizer
from .preprocessing import PreprocessingPipeline

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Engine:
    """
    Unified engine for training and inference.

    Handles:
    - Model training with mixed precision and gradient clipping
    - Inference with preprocessing integration
    - Checkpoint save/load
    - Metrics computation
    """

    def __init__(
        self,
        model: SimplifiedGNN,
        config: EngineConfig,
        preprocessing: PreprocessingPipeline | None = None,
    ):
        """
        Initialize Engine with model and configuration.

        Args:
            model: SimplifiedGNN model instance
            config: EngineConfig with training/inference settings
            preprocessing: Optional PreprocessingPipeline for target normalization
        """
        self.config = config
        self.device = config.resolved_device
        self.preprocessing = preprocessing

        # Move model to device
        self.model = model.to(self.device)

        # Optionally compile
        if config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup mixed precision (only for CUDA)
        self.scaler = None
        if config.use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Featurizer for SMILES input (lazy)
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
        """
        Create engine from model configuration.

        Args:
            model_config: ModelConfig with architecture settings
            engine_config: EngineConfig with training/inference settings
            preprocessing: Optional PreprocessingPipeline

        Returns:
            Engine instance with newly created model
        """
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
        """Create learning rate scheduler based on config."""
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
            Loss value as float
        """
        self.model.train()
        batch = batch.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass with optional AMP
        if self.scaler is not None:
            with torch.amp.autocast("cuda"):
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
            with torch.amp.autocast("cuda"):
                predictions = self.model(batch)
        else:
            predictions = self.model(batch)

        return predictions.cpu()

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

    def step_scheduler(self, val_loss: float | None = None) -> None:
        """
        Step the learning rate scheduler.

        Args:
            val_loss: Validation loss (required for ReduceLROnPlateau)
        """
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def save_checkpoint(self, path: str) -> None:
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
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

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
        """
        Get current learning rate.

        Returns:
            Current learning rate from optimizer
        """
        return self.optimizer.param_groups[0]["lr"]
