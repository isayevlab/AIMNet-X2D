"""Loss function registry for extensible loss creation.

Provides a registry pattern for loss functions, allowing external
code to register custom losses without modifying the Engine class.
"""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


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
