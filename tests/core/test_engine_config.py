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
        # Should resolve to either cuda or cpu
        device = config.resolved_device
        assert device.type in ("cuda", "cpu")

    def test_loss_function_config(self):
        """Test that loss_function can be configured."""
        config = EngineConfig(loss_function="mae")
        assert config.loss_function == "mae"

        config2 = EngineConfig(loss_function="mse")
        assert config2.loss_function == "mse"

    def test_loss_function_in_to_dict(self):
        """Test that loss_function is serialized."""
        config = EngineConfig(loss_function="huber")
        d = config.to_dict()
        assert d["loss_function"] == "huber"

    def test_loss_kwargs_default(self):
        """Test loss_kwargs defaults to empty dict."""
        config = EngineConfig()
        assert config.loss_kwargs == {}

    def test_loss_kwargs_custom(self):
        """Test loss_kwargs can be customized."""
        config = EngineConfig(
            loss_function="evidential",
            loss_kwargs={"coeff": 0.05}
        )
        assert config.loss_kwargs == {"coeff": 0.05}

    def test_to_dict_includes_loss_kwargs(self):
        """Test loss_kwargs serialization."""
        config = EngineConfig(loss_kwargs={"coeff": 0.1})
        d = config.to_dict()
        assert "loss_kwargs" in d
        assert d["loss_kwargs"] == {"coeff": 0.1}

    def test_from_dict_restores_loss_kwargs(self):
        """Test loss_kwargs deserialization."""
        d = {"loss_function": "huber", "loss_kwargs": {"delta": 1.5}}
        config = EngineConfig.from_dict(d)
        assert config.loss_kwargs == {"delta": 1.5}
