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
