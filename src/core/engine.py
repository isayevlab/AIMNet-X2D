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
from .losses import create_loss

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
        self.loss_fn = self._create_loss_function()

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

    def _create_loss_function(self) -> nn.Module:
        """Create loss function from registry based on config."""
        return create_loss(self.config.loss_function, **self.config.loss_kwargs)

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

        return loss.item()  # Return unscaled loss for logging

    @torch.inference_mode()
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

        # Use AMP autocast for consistency with predict() and train_step()
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

    def train_epoch(
        self,
        train_batches: list[MolecularGraphBatch],
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_batches: List of training batches

        Returns:
            Dict with epoch metrics ('loss', 'lr')
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
            verbose: Whether to log progress

        Returns:
            Training history dict with 'train_loss', 'val_loss', 'lr' lists
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
