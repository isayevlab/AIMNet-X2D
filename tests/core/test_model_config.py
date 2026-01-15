"""Unit tests for ModelConfig dataclass."""

import pytest

from src.core.model_config import ModelConfig


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_creation(self) -> None:
        """Create ModelConfig with only required parameters."""
        config = ModelConfig(hidden_dim=128, output_dim=1)

        assert config.hidden_dim == 128
        assert config.output_dim == 1
        assert config.num_shells == 3
        assert config.num_message_passing_layers == 3
        assert config.embedding_dim == 64
        assert config.dropout == 0.05
        assert config.ffn_num_layers == 3
        assert config.attention_num_heads == 4

    def test_custom_parameters(self) -> None:
        """Create ModelConfig with all custom values."""
        config = ModelConfig(
            hidden_dim=256,
            output_dim=12,
            num_shells=5,
            num_message_passing_layers=6,
            embedding_dim=128,
            dropout=0.1,
            ffn_num_layers=4,
            ffn_hidden_dim=512,
            attention_num_heads=8,
        )

        assert config.hidden_dim == 256
        assert config.output_dim == 12
        assert config.num_shells == 5
        assert config.num_message_passing_layers == 6
        assert config.embedding_dim == 128
        assert config.dropout == 0.1
        assert config.ffn_num_layers == 4
        assert config.ffn_hidden_dim == 512
        assert config.attention_num_heads == 8

    def test_ffn_hidden_dim_defaults_to_hidden_dim(self) -> None:
        """ffn_hidden_dim defaults to hidden_dim when not specified."""
        config = ModelConfig(hidden_dim=256, output_dim=1)

        assert config.ffn_hidden_dim == 256

    def test_ffn_hidden_dim_can_be_overridden(self) -> None:
        """ffn_hidden_dim can be explicitly set to a different value."""
        config = ModelConfig(hidden_dim=256, output_dim=1, ffn_hidden_dim=512)

        assert config.ffn_hidden_dim == 512
        assert config.hidden_dim == 256

    def test_to_dict(self) -> None:
        """Verify serialization to dictionary."""
        config = ModelConfig(
            hidden_dim=128,
            output_dim=5,
            num_shells=4,
            dropout=0.2,
        )

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["hidden_dim"] == 128
        assert result["output_dim"] == 5
        assert result["num_shells"] == 4
        assert result["dropout"] == 0.2
        # Check defaults are also serialized
        assert result["num_message_passing_layers"] == 3
        assert result["embedding_dim"] == 64
        assert result["ffn_num_layers"] == 3
        assert result["ffn_hidden_dim"] == 128  # Should be hidden_dim
        assert result["attention_num_heads"] == 4

    def test_from_dict(self) -> None:
        """Verify deserialization from dictionary."""
        data = {
            "hidden_dim": 256,
            "output_dim": 12,
            "num_shells": 5,
            "num_message_passing_layers": 6,
            "embedding_dim": 128,
            "dropout": 0.1,
            "ffn_num_layers": 4,
            "ffn_hidden_dim": 512,
            "attention_num_heads": 8,
        }

        config = ModelConfig.from_dict(data)

        assert config.hidden_dim == 256
        assert config.output_dim == 12
        assert config.num_shells == 5
        assert config.num_message_passing_layers == 6
        assert config.embedding_dim == 128
        assert config.dropout == 0.1
        assert config.ffn_num_layers == 4
        assert config.ffn_hidden_dim == 512
        assert config.attention_num_heads == 8

    def test_feature_sizes_computed(self) -> None:
        """Verify get_feature_sizes returns correct vocabulary sizes."""
        config = ModelConfig(hidden_dim=128, output_dim=1)

        feature_sizes = config.get_feature_sizes()

        assert isinstance(feature_sizes, dict)
        assert feature_sizes["atom_type"] == 119
        assert feature_sizes["degree"] == 7
        assert feature_sizes["hybridization"] == 8
        assert feature_sizes["hydrogen_count"] == 9
