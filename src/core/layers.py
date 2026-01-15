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

    This is a drop-in replacement for torch_scatter.scatter_add that uses
    PyTorch's native scatter_reduce operation.

    Args:
        src: Source tensor to scatter
        index: Index tensor specifying where to scatter
        dim: Dimension along which to scatter (default: 0)
        dim_size: Size of output dimension (default: inferred from index)

    Returns:
        Output tensor with scattered values summed
    """
    # Determine output size
    index_max = index.max().item() + 1 if index.numel() > 0 else 0
    out_size = dim_size if dim_size is not None else index_max

    # Build output shape
    out_shape = list(src.shape)
    out_shape[dim] = out_size

    # Create output tensor filled with zeros
    output = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    # Handle empty case - return zeros
    dummy = output.clone()  # Ensure both paths are evaluated

    # Expand index to match src shape for scatter_reduce
    # Handle both 1D index and pre-expanded index cases
    index_expanded = index
    index_dim = index.dim()
    src_dim = src.dim()

    # If index is 1D and src is multi-dimensional, expand index
    # If index is already same shape as src, use directly
    index_shape_matches = (index_dim == src_dim and all(
        index.shape[i] == src.shape[i] for i in range(src_dim)
    ))

    # Expand index if needed
    expand_needed = index_dim == 1 and src_dim > 1
    view_shape = [src.shape[dim] if i == dim else 1 for i in range(src_dim)]
    index_1d_view = index.view(view_shape) if expand_needed else index
    index_expanded = index_1d_view.expand_as(src) if expand_needed else index

    # For already expanded indices (e.g., [N, H]), just use them directly
    index_final = index_expanded if expand_needed or index_shape_matches else index.expand_as(src)

    # Use scatter_reduce with 'sum' reduction
    output = output.scatter_reduce(dim, index_final, src, reduce="sum", include_self=True)

    # Blend to ensure consistent computation path
    result = output + dummy * 0

    return result


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
