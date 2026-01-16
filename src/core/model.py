"""
SimplifiedGNN model with batch-native forward pass.

This module contains the main GNN model that accepts MolecularGraphBatch
directly instead of multiple separate arguments.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from .model_config import ModelConfig
from .batch import MolecularGraphBatch
from .layers import ShellConvBlock, AttentionPooling, FeedForwardNetwork, StereochemistryEncoder

logger = logging.getLogger(__name__)


class SimplifiedGNN(nn.Module):
    """
    Simplified Graph Neural Network for molecular property prediction.

    This model accepts MolecularGraphBatch directly, eliminating the need
    for 7 separate arguments in the forward pass. The architecture consists of:

    1. Embeddings: One nn.Embedding per feature type
    2. Projection: Concatenate embeddings -> Linear -> SiLU
    3. Message Passing: num_message_passing_layers ShellConvBlocks
    4. Pooling: AttentionPooling to get graph-level representation
    5. FFN: FeedForwardNetwork for final processing
    6. Output: Linear layer to output_dim

    Args:
        config: ModelConfig containing all architecture hyperparameters
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Get feature vocabulary sizes
        feature_sizes = config.get_feature_sizes()

        # 1. Embeddings for each feature type
        self.embeddings = nn.ModuleDict({
            "atom_type": nn.Embedding(
                feature_sizes["atom_type"], config.embedding_dim
            ),
            "degree": nn.Embedding(
                feature_sizes["degree"], config.embedding_dim
            ),
            "hybridization": nn.Embedding(
                feature_sizes["hybridization"], config.embedding_dim
            ),
            "hydrogen_count": nn.Embedding(
                feature_sizes["hydrogen_count"], config.embedding_dim
            ),
        })

        # Store vocabulary sizes for index clamping
        self._vocab_sizes = feature_sizes

        # 2. Projection layer: concatenated embeddings -> hidden_dim
        concat_dim = config.embedding_dim * len(self.embeddings)
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, config.hidden_dim),
            nn.SiLU(),
        )

        # Stereochemistry encoder (adds to atom features)
        self.stereo_encoder = StereochemistryEncoder(config.hidden_dim)

        # Charge embedding (added to molecule features after pooling)
        self.charge_embedding = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # 3. Message passing layers
        self.message_passing_layers = nn.ModuleList()
        for layer_idx in range(config.num_message_passing_layers):
            # First layer takes hidden_dim input, all layers output hidden_dim
            input_dim = config.hidden_dim
            self.message_passing_layers.append(
                ShellConvBlock(
                    input_dim=input_dim,
                    hidden_dim=config.hidden_dim,
                    num_shells=config.num_shells,
                    dropout=config.dropout,
                )
            )

        # 4. Attention pooling
        self.pooling = AttentionPooling(
            input_dim=config.hidden_dim,
            num_heads=config.attention_num_heads,
        )

        # 5. Feed-forward network
        self.ffn = FeedForwardNetwork(
            input_dim=config.hidden_dim,
            hidden_dim=config.ffn_hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.ffn_num_layers,
            dropout=config.dropout,
        )

        # 6. Output layer
        self.output_layer = nn.Linear(config.hidden_dim, config.output_dim)

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            f"SimplifiedGNN created with {total_params:,} parameters "
            f"({trainable_params:,} trainable)"
        )

    def forward(self, batch: MolecularGraphBatch) -> Tensor:
        """
        Forward pass through the model.

        Args:
            batch: MolecularGraphBatch containing molecular data

        Returns:
            Output tensor of shape [num_molecules, output_dim]
        """
        # Get atom features as dictionary
        features = batch.atom_features_dict()

        # Compute embeddings for each feature, clamping indices to valid range
        embedded_features = []
        for feature_name, embedding_layer in self.embeddings.items():
            if feature_name in features:
                indices = features[feature_name]
            else:
                # If feature not present, use zeros
                indices = torch.zeros(
                    batch.total_atoms,
                    dtype=torch.long,
                    device=batch.device,
                )

            # Clamp indices to valid range [0, vocab_size - 1]
            vocab_size = self._vocab_sizes[feature_name]
            indices = indices.clamp(0, vocab_size - 1)

            embedded = embedding_layer(indices)
            embedded_features.append(embedded)

        # Concatenate all embeddings [total_atoms, concat_dim]
        x = torch.cat(embedded_features, dim=-1)

        # Project to hidden dimension [total_atoms, hidden_dim]
        x = self.projection(x)

        # Add stereochemistry information
        x = self.stereo_encoder(
            x,
            batch.chiral_indices,
            batch.cis_bond_indices,
            batch.trans_bond_indices,
        )

        # Message passing layers
        for mp_layer in self.message_passing_layers:
            x = mp_layer(x, batch.edge_indices)

        # Pooling to molecule level [num_molecules, hidden_dim]
        x = self.pooling(x, batch.batch_idx, num_molecules=batch.num_molecules)

        # Add charge information if available
        if batch.total_charges is not None:
            charge_features = self.charge_embedding(
                batch.total_charges.unsqueeze(-1)
            )
            x = x + charge_features

        # Feed-forward network [num_molecules, hidden_dim]
        x = self.ffn(x)

        # Output layer [num_molecules, output_dim]
        output = self.output_layer(x)

        return output

    def compile(self, **kwargs) -> "SimplifiedGNN":
        """
        Compile the model with torch.compile() for optimized execution.

        Args:
            **kwargs: Arguments passed to torch.compile()
                      Default mode is "reduce-overhead" for inference.

        Returns:
            Compiled model (or self if torch.compile unavailable)
        """
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available, returning uncompiled model")
            return self

        mode = kwargs.pop('mode', 'reduce-overhead')
        return torch.compile(self, mode=mode, **kwargs)
