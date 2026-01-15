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
