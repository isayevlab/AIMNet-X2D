"""Model configuration dataclass for GNN models.

This module provides a simplified configuration interface that reduces
the number of constructor parameters from 22 to 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for GNN model architecture.

    This dataclass encapsulates all model hyperparameters, reducing the
    number of constructor arguments and providing sensible defaults.

    Attributes:
        hidden_dim: Hidden dimension size for model layers.
        output_dim: Output dimension (number of targets).
        num_shells: Number of shells for message passing. Default: 3.
        num_message_passing_layers: Number of message passing layers. Default: 3.
        embedding_dim: Dimension for atom embeddings. Default: 64.
        dropout: Dropout rate. Default: 0.05.
        ffn_num_layers: Number of layers in feed-forward networks. Default: 3.
        ffn_hidden_dim: Hidden dimension for FFN. Defaults to hidden_dim if None.
        attention_num_heads: Number of attention heads. Default: 4.
    """

    # Required parameters
    hidden_dim: int
    output_dim: int

    # Optional parameters with defaults
    num_shells: int = 3
    num_message_passing_layers: int = 3
    embedding_dim: int = 64
    dropout: float = 0.05
    ffn_num_layers: int = 3
    ffn_hidden_dim: int | None = field(default=None)
    attention_num_heads: int = 4

    def __post_init__(self) -> None:
        """Set ffn_hidden_dim to hidden_dim if not specified."""
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = self.hidden_dim

    def get_feature_sizes(self) -> dict[str, int]:
        """Get vocabulary sizes for categorical features.

        Returns:
            Dictionary mapping feature names to their vocabulary sizes:
            - atom_type: 119 (elements in periodic table + padding)
            - degree: 7 (0-6 bonds)
            - hybridization: 8 (sp, sp2, sp3, etc.)
            - hydrogen_count: 9 (0-8 hydrogens)
        """
        return {
            "atom_type": 119,
            "degree": 7,
            "hybridization": 8,
            "hydrogen_count": 9,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary.

        Returns:
            Dictionary containing all configuration parameters.
        """
        return {
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_shells": self.num_shells,
            "num_message_passing_layers": self.num_message_passing_layers,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "ffn_num_layers": self.ffn_num_layers,
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "attention_num_heads": self.attention_num_heads,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Deserialize configuration from dictionary.

        Args:
            d: Dictionary containing configuration parameters.

        Returns:
            ModelConfig instance with the specified parameters.
        """
        return cls(
            hidden_dim=d["hidden_dim"],
            output_dim=d["output_dim"],
            num_shells=d.get("num_shells", 3),
            num_message_passing_layers=d.get("num_message_passing_layers", 3),
            embedding_dim=d.get("embedding_dim", 64),
            dropout=d.get("dropout", 0.05),
            ffn_num_layers=d.get("ffn_num_layers", 3),
            ffn_hidden_dim=d.get("ffn_hidden_dim"),
            attention_num_heads=d.get("attention_num_heads", 4),
        )
