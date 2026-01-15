"""
Simplified model layers for GPU-native molecular GNN.

This module contains core neural network layers for the refactored architecture:
- ShellConvBlock: Multi-hop message passing with shell convolution
- AttentionPooling: Multi-head attention pooling for molecule-level features
- FeedForwardNetwork: Simple FFN with SiLU activation
"""

import torch
import torch.nn as nn
from torch import Tensor


def scatter_add(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> Tensor:
    """
    Scatter add operation using PyTorch native operations.

    Optimized for torch.compile() compatibility by avoiding
    dynamic control flow.

    Args:
        src: Source tensor to scatter
        index: Index tensor specifying where to scatter
        dim: Dimension along which to scatter (default: 0)
        dim_size: Size of output dimension (default: inferred from index)

    Returns:
        Output tensor with scattered values summed
    """
    # Handle empty input
    if src.numel() == 0:
        out_size = dim_size if dim_size is not None else 0
        out_shape = list(src.shape)
        out_shape[dim] = out_size
        return torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    # Determine output size
    out_size = dim_size if dim_size is not None else (index.max().item() + 1)

    # Build output shape
    out_shape = list(src.shape)
    out_shape[dim] = out_size

    # Create output tensor
    output = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    # Expand index if needed to match src shape
    if index.dim() == 1 and src.dim() > 1:
        # Expand 1D index to match src dimensions
        expand_shape = [1] * src.dim()
        expand_shape[dim] = -1
        index = index.view(*expand_shape).expand_as(src)

    # Use scatter_add_ for in-place accumulation (faster than scatter_reduce)
    output.scatter_add_(dim, index, src)

    return output


class ShellConvBlock(nn.Module):
    """
    Shell convolution block for multi-hop message passing.

    Performs message passing between atoms connected at different hop distances,
    with learned per-shell transforms and combination weights.

    Args:
        input_dim: Input dimension of atom features
        hidden_dim: Output hidden dimension
        num_shells: Number of shells (hops) for message passing
        dropout: Dropout probability (default: 0.05)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_shells: int,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_shells = num_shells

        # Per-shell linear transforms
        self.shell_transforms = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_shells)
        ])

        # Self-connection transform
        self.self_transform = nn.Linear(input_dim, hidden_dim)

        # Learnable combination weights for shells (including self)
        self.combination_weights = nn.Parameter(torch.ones(num_shells + 1))

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Activation
        self.activation = nn.SiLU()

        # Residual projection if dimensions differ
        self.residual_proj = nn.Linear(input_dim, hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_indices: list[Tensor],
    ) -> Tensor:
        """
        Forward pass through shell convolution block.

        Args:
            x: Atom features [num_atoms, input_dim]
            edge_indices: List of edge index tensors [2, E] for each shell

        Returns:
            Updated atom features [num_atoms, hidden_dim]
        """
        num_atoms = x.shape[0]

        # Compute self-connection
        self_features = self.self_transform(x)

        # Compute shell features via message passing
        shell_features_list = []
        for shell_idx, (shell_transform, edges) in enumerate(
            zip(self.shell_transforms, edge_indices)
        ):
            shell_features = self._message_passing(x, edges, shell_transform, num_atoms)
            shell_features_list.append(shell_features)

        # Normalize combination weights
        weights = torch.softmax(self.combination_weights, dim=0)

        # Combine self and shell features
        combined = weights[0] * self_features
        for shell_idx, shell_feat in enumerate(shell_features_list):
            combined = combined + weights[shell_idx + 1] * shell_feat

        # Apply activation and dropout
        combined = self.activation(combined)
        combined = self.dropout(combined)

        # Residual connection
        residual = self.residual_proj(x)
        output = combined + residual

        # Layer normalization
        output = self.layer_norm(output)

        return output

    def _message_passing(
        self,
        x: Tensor,
        edge_index: Tensor,
        transform: nn.Linear,
        num_atoms: int,
    ) -> Tensor:
        """
        Perform message passing for a single shell.

        Args:
            x: Atom features [num_atoms, input_dim]
            edge_index: Edge indices [2, num_edges]
            transform: Linear transform for this shell
            num_atoms: Number of atoms

        Returns:
            Aggregated features [num_atoms, hidden_dim]
        """
        # Handle empty edges
        num_edges = edge_index.shape[1]

        # Transform source features
        transformed = transform(x)

        # If no edges, return zeros
        zeros = torch.zeros(num_atoms, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Handle empty edge case - always compute both paths
        source_idx = edge_index[0] if num_edges > 0 else torch.zeros(0, dtype=torch.long, device=x.device)
        target_idx = edge_index[1] if num_edges > 0 else torch.zeros(0, dtype=torch.long, device=x.device)

        # Gather source features
        source_features = transformed[source_idx] if num_edges > 0 else torch.zeros(0, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Aggregate to target nodes
        aggregated = scatter_add(
            source_features,
            target_idx,
            dim=0,
            dim_size=num_atoms,
        )

        # Blend result (always execute both paths, no early return)
        result = aggregated + zeros * 0  # zeros term ensures both paths execute

        return result


class AttentionPooling(nn.Module):
    """
    Multi-head attention pooling for molecule-level aggregation.

    Computes attention weights for each atom within a molecule,
    then performs weighted aggregation to produce molecule-level features.

    Args:
        input_dim: Input dimension of atom features
        num_heads: Number of attention heads (default: 4)
        temperature: Temperature for softmax scaling (default: 1.0)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.temperature = temperature

        # Single linear layer for all heads (efficient)
        self.attention_linear = nn.Linear(input_dim, num_heads)

        # Output projection to combine heads
        self.output_proj = nn.Linear(input_dim * num_heads, input_dim)

    def forward(
        self,
        x: Tensor,
        batch_idx: Tensor,
    ) -> Tensor:
        """
        Forward pass through attention pooling.

        Args:
            x: Atom features [num_atoms, input_dim]
            batch_idx: Molecule index for each atom [num_atoms]

        Returns:
            Molecule features [num_molecules, input_dim]
        """
        num_atoms = x.shape[0]
        num_molecules = batch_idx.max().item() + 1

        # Compute attention scores for all heads [num_atoms, num_heads]
        attention_scores = self.attention_linear(x) / self.temperature

        # Apply softmax within each molecule for each head
        # We need to compute softmax per molecule, using scatter operations

        # Get max per molecule for numerical stability [num_molecules, num_heads]
        max_scores = torch.zeros(
            num_molecules, self.num_heads,
            device=x.device, dtype=x.dtype
        )
        max_scores = max_scores.scatter_reduce(
            0,
            batch_idx.unsqueeze(1).expand(-1, self.num_heads),
            attention_scores,
            reduce="amax",
            include_self=True,
        )

        # Subtract max for stability
        attention_scores = attention_scores - max_scores[batch_idx]

        # Compute exp
        attention_exp = torch.exp(attention_scores)

        # Sum exp per molecule [num_molecules, num_heads]
        attention_sum = scatter_add(
            attention_exp,
            batch_idx.unsqueeze(1).expand(-1, self.num_heads),
            dim=0,
            dim_size=num_molecules,
        )

        # Normalize to get attention weights [num_atoms, num_heads]
        attention_weights = attention_exp / (attention_sum[batch_idx] + 1e-6)

        # Weighted aggregation for each head
        # [num_atoms, num_heads, input_dim]
        weighted_features = x.unsqueeze(1) * attention_weights.unsqueeze(2)

        # Reshape for scatter: [num_atoms, num_heads * input_dim]
        weighted_features = weighted_features.reshape(num_atoms, -1)

        # Aggregate per molecule [num_molecules, num_heads * input_dim]
        aggregated = scatter_add(
            weighted_features,
            batch_idx,
            dim=0,
            dim_size=num_molecules,
        )

        # Project back to input_dim [num_molecules, input_dim]
        output = self.output_proj(aggregated)

        return output


class FeedForwardNetwork(nn.Module):
    """
    Simple feed-forward network with SiLU activation.

    Multi-layer perceptron with configurable depth. Uses SiLU activation
    between layers, with no activation on the final layer.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of layers (default: 3)
        dropout: Dropout probability (default: 0.05)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Build layers
        layers: list[nn.Module] = []

        # Handle single layer case
        in_dim = input_dim
        for layer_idx in range(num_layers):
            is_last = layer_idx == num_layers - 1
            out_dim = output_dim if is_last else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            # Add activation and dropout for non-final layers
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))

            in_dim = out_dim

        # Remove activation and dropout from last layer
        # Pop the last dropout and activation
        layers.pop()  # Remove dropout
        layers.pop()  # Remove SiLU

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through FFN.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        return self.layers(x)


class StereochemistryEncoder(nn.Module):
    """
    Encodes stereochemistry information as atom-level features.

    Marks atoms involved in chiral centers and cis/trans bonds with
    learnable embeddings that get added to atom features.

    Args:
        hidden_dim: Dimension of atom features to match
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Learnable embeddings for stereochemistry types
        self.chiral_center_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.chiral_neighbor_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.cis_bond_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.trans_bond_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(
        self,
        x: Tensor,
        chiral_indices: Tensor | None,
        cis_bond_indices: Tensor | None,
        trans_bond_indices: Tensor | None,
    ) -> Tensor:
        """
        Add stereochemistry information to atom features.

        Args:
            x: Atom features [num_atoms, hidden_dim]
            chiral_indices: [num_chiral, 4] - center + 3 neighbors
            cis_bond_indices: [num_cis, 4] - bond atoms + neighbors
            trans_bond_indices: [num_trans, 4] - bond atoms + neighbors

        Returns:
            Updated atom features with stereochemistry encoded
        """
        output = x.clone()

        # Add chiral center embeddings
        if chiral_indices is not None and chiral_indices.shape[0] > 0:
            center_atoms = chiral_indices[:, 0]
            output[center_atoms] = output[center_atoms] + self.chiral_center_embed

            neighbor_atoms = chiral_indices[:, 1:4].flatten()
            valid_neighbors = neighbor_atoms[neighbor_atoms < x.shape[0]]
            if valid_neighbors.numel() > 0:
                output[valid_neighbors] = output[valid_neighbors] + self.chiral_neighbor_embed

        # Add cis bond embeddings
        if cis_bond_indices is not None and cis_bond_indices.shape[0] > 0:
            cis_atoms = cis_bond_indices[:, :2].flatten()
            valid_cis = cis_atoms[cis_atoms < x.shape[0]]
            if valid_cis.numel() > 0:
                output[valid_cis] = output[valid_cis] + self.cis_bond_embed

        # Add trans bond embeddings
        if trans_bond_indices is not None and trans_bond_indices.shape[0] > 0:
            trans_atoms = trans_bond_indices[:, :2].flatten()
            valid_trans = trans_atoms[trans_atoms < x.shape[0]]
            if valid_trans.numel() > 0:
                output[valid_trans] = output[valid_trans] + self.trans_bond_embed

        return output
