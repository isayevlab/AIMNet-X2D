"""Tests for loss registry."""
import pytest
import torch
import torch.nn as nn

from src.core.losses import LOSS_REGISTRY, register_loss, create_loss


class TestLossRegistry:
    """Tests for the loss registry system."""

    def test_builtin_losses_registered(self):
        """Test that built-in losses are in registry."""
        assert "mse" in LOSS_REGISTRY
        assert "mae" in LOSS_REGISTRY
        assert "huber" in LOSS_REGISTRY

    def test_create_mse_loss(self):
        """Test creating MSE loss."""
        loss_fn = create_loss("mse")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_create_mae_loss(self):
        """Test creating MAE loss."""
        loss_fn = create_loss("mae")
        assert isinstance(loss_fn, nn.L1Loss)

    def test_create_huber_loss(self):
        """Test creating Huber loss."""
        loss_fn = create_loss("huber")
        assert isinstance(loss_fn, nn.HuberLoss)

    def test_create_unknown_loss_raises(self):
        """Test that unknown loss raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss"):
            create_loss("unknown_loss")

    def test_register_custom_loss(self):
        """Test registering a custom loss."""
        @register_loss("custom_test")
        class CustomLoss(nn.Module):
            def forward(self, pred, target):
                return (pred - target).abs().sum()

        assert "custom_test" in LOSS_REGISTRY
        loss_fn = create_loss("custom_test")
        assert isinstance(loss_fn, CustomLoss)

        # Cleanup
        del LOSS_REGISTRY["custom_test"]

    def test_loss_functions_compute_correctly(self):
        """Test that loss functions compute values correctly."""
        pred = torch.tensor([[1.0], [2.0], [3.0]])
        target = torch.tensor([[1.5], [2.5], [3.5]])

        # MSE
        mse_loss = create_loss("mse")
        mse_value = mse_loss(pred, target)
        expected_mse = ((pred - target) ** 2).mean()
        assert torch.allclose(mse_value, expected_mse)

        # MAE
        mae_loss = create_loss("mae")
        mae_value = mae_loss(pred, target)
        expected_mae = (pred - target).abs().mean()
        assert torch.allclose(mae_value, expected_mae)

        # Huber
        huber_loss = create_loss("huber")
        huber_value = huber_loss(pred, target)
        assert huber_value > 0  # Just check it runs and returns positive

    def test_create_loss_with_kwargs(self):
        """Test that kwargs are passed to loss constructor."""
        # HuberLoss accepts delta parameter
        loss_fn = create_loss("huber", delta=2.0)
        assert isinstance(loss_fn, nn.HuberLoss)
        assert loss_fn.delta == 2.0

    def test_error_message_lists_available_losses(self):
        """Test that error message lists available losses."""
        with pytest.raises(ValueError) as exc_info:
            create_loss("nonexistent")

        error_msg = str(exc_info.value)
        assert "huber" in error_msg
        assert "mae" in error_msg
        assert "mse" in error_msg

    def test_register_loss_returns_class(self):
        """Test that register_loss decorator returns the class unchanged."""
        @register_loss("test_return")
        class TestLoss(nn.Module):
            def forward(self, pred, target):
                return torch.tensor(0.0)

        # The decorated class should be the same as TestLoss
        assert LOSS_REGISTRY["test_return"] is TestLoss

        # Cleanup
        del LOSS_REGISTRY["test_return"]

    def test_create_evidential_loss(self):
        """Test creating evidential loss."""
        loss_fn = create_loss("evidential")

        # Test forward pass
        pred = torch.randn(10, 4)  # mu, v, alpha, beta
        target = torch.randn(10, 1)

        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_evidential_loss_with_custom_coeff(self):
        """Test evidential loss with custom regularization coefficient."""
        loss_fn = create_loss("evidential", coeff=0.1)
        assert loss_fn.coeff == 0.1

        pred = torch.randn(10, 4)
        target = torch.randn(10, 1)

        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_evidential_loss_positive_output(self):
        """Test that evidential loss produces positive values for typical inputs."""
        loss_fn = create_loss("evidential")

        # Use reasonable prediction values
        pred = torch.tensor([
            [0.5, 1.0, 1.5, 0.5],
            [1.0, 0.5, 2.0, 1.0],
            [0.0, 0.8, 1.2, 0.3],
        ])
        target = torch.tensor([[0.6], [0.9], [0.1]])

        loss = loss_fn(pred, target)
        assert loss > 0  # Loss should be positive
