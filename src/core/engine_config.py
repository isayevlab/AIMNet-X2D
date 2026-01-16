"""Engine configuration for training and inference.

Provides a unified configuration interface for the Engine class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass
class EngineConfig:
    """Configuration for the unified Engine.

    Attributes:
        learning_rate: Learning rate for optimizer.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        weight_decay: L2 regularization weight.
        device: Device to use ('cuda', 'cpu', or 'auto').
        num_workers: DataLoader workers.
        gradient_clip: Max gradient norm (None to disable).
        scheduler: Learning rate scheduler type.
        warmup_epochs: Epochs for learning rate warmup.
        early_stopping_patience: Epochs without improvement before stopping.
        checkpoint_dir: Directory for saving checkpoints.
        log_interval: Steps between logging.
        use_amp: Use automatic mixed precision.
        compile_model: Use torch.compile().
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

    # Loss
    loss_function: Literal["mse", "mae", "huber"] = "mse"

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
        """Get resolved device (handles 'auto').

        Returns:
            torch.device for 'cuda' if available when device is 'auto',
            otherwise the specified device.
        """
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

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
            "device": self.device,
            "num_workers": self.num_workers,
            "use_amp": self.use_amp,
            "compile_model": self.compile_model,
            "checkpoint_dir": self.checkpoint_dir,
            "log_interval": self.log_interval,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EngineConfig:
        """Deserialize configuration from dictionary.

        Args:
            d: Dictionary containing configuration parameters.

        Returns:
            EngineConfig instance with the specified parameters.
        """
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
