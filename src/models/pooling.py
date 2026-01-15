"""
Graph pooling layers for molecular GNN.

This module contains various pooling mechanisms to aggregate
node-level features into graph-level representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class MeanPoolingLayer(nn.Module):
    """Mean pooling for graph-level representations."""
    
    def __init__(self):
        super(MeanPoolingLayer, self).__init__()

    def forward(self, x: torch.Tensor, batch_indices: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Perform mean pooling on node features.
        
        Args:
            x: Node feature matrix of shape [num_nodes, num_features]
            batch_indices: Batch index for each node of shape [num_nodes]
            
        Returns:
            Tuple of (pooled_features, None)
        """
        x = x.to(batch_indices.device)
        x_pooled = torch_scatter.scatter_mean(x, batch_indices, dim=0)
        return x_pooled, None


class MaxPoolingLayer(nn.Module):
    """Max pooling for graph-level representations."""
    
    def __init__(self):
        super(MaxPoolingLayer, self).__init__()

    def forward(self, x: torch.Tensor, batch_indices: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Perform max pooling on node features.

        Args:
            x: Node feature matrix of shape [num_nodes, num_features]
            batch_indices: Batch index for each node of shape [num_nodes]

        Returns:
            Tuple of (pooled_features, None)
        """
        x = x.to(batch_indices.device)
        # scatter_max returns a tuple (values, argmax_indices)
        x_pooled, _ = torch_scatter.scatter_max(x, batch_indices, dim=0)
        return x_pooled, None


class SumPoolingLayer(nn.Module):
    """Sum pooling for graph-level representations."""
    
    def __init__(self):
        super(SumPoolingLayer, self).__init__()

    def forward(self, x: torch.Tensor, batch_indices: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Perform sum pooling on node features.

        Args:
            x: Node feature matrix of shape [num_nodes, num_features]
            batch_indices: Batch index for each node of shape [num_nodes]

        Returns:
            Tuple of (pooled_features, None)
        """
        x = x.to(batch_indices.device)
        # scatter_add sums the features for each graph in the batch
        x_pooled = torch_scatter.scatter_add(x, batch_indices, dim=0)
        return x_pooled, None


class MultiHeadAttentionPoolingLayer(nn.Module):
    """
    Multi-head attention pooling for graph-level representations.
    
    This layer uses multiple attention heads to compute weighted averages
    of node features, providing a more flexible pooling mechanism.
    
    Args:
        input_dim: Input dimension
        num_heads: Number of attention heads
        initial_temperature: Initial temperature for attention scaling
        learnable_temperature: Whether temperature is learnable
        dropout_prob: Dropout probability
    """
    
    def __init__(self, 
                 input_dim: int, 
                 num_heads: int = 4, 
                 initial_temperature: float = 1.0,
                 learnable_temperature: bool = True, 
                 dropout_prob: float = 0.0):
        super(MultiHeadAttentionPoolingLayer, self).__init__()
        
        self.num_heads = num_heads
        self.input_dim = input_dim
        
        # Create separate attention weights for each head
        self.attention_weights = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_heads)
        ])
        
        # Temperature parameter for attention scaling
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        else:
            self.register_buffer('temperature', torch.tensor(initial_temperature))
            
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, batch_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention pooling.
        
        Args:
            x: Node feature matrix of shape [num_nodes, input_dim]
            batch_indices: Batch index for each node of shape [num_nodes]
            
        Returns:
            Tuple of (pooled_features, attention_weights)
        """
        # Compute attention scores for each head
        attention_scores_list = []
        for i in range(self.num_heads):
            scores = self.attention_weights[i](x).squeeze(-1) / self.temperature
            attention_scores_list.append(scores)

        # Stack attention scores: [num_heads, num_nodes]
        attention_scores = torch.stack(attention_scores_list, dim=0)
        
        # Apply softmax within each graph for each head
        if batch_indices is not None:
            expanded_batch_indices = batch_indices.unsqueeze(0).expand(self.num_heads, -1)
            attention_weights = torch_scatter.scatter_softmax(attention_scores, expanded_batch_indices, dim=1)
        else:
            attention_weights = F.softmax(attention_scores, dim=1)

        # Expand node features for each head: [num_heads, num_nodes, input_dim]
        x_expanded = x.unsqueeze(0).expand(self.num_heads, -1, -1)
        
        # Apply attention weights: [num_heads, num_nodes, input_dim]
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        weighted_x = x_expanded * attention_weights_expanded

        # Aggregate within each graph for each head
        if batch_indices is not None:
            # Sum over nodes within each graph: [num_heads, num_graphs, input_dim]
            x_pooled = torch_scatter.scatter_sum(weighted_x, batch_indices, dim=1)
            # Average across heads: [num_graphs, input_dim]
            x_pooled = x_pooled.mean(dim=0)
        else:
            # Sum over all nodes: [num_heads, 1, input_dim]
            x_pooled = torch.sum(weighted_x, dim=1, keepdim=True)
            # Average across heads: [1, input_dim]
            x_pooled = x_pooled.mean(dim=0)

        # Apply dropout
        if self.dropout.p > 0:
            x_pooled = self.dropout(x_pooled)

        return x_pooled, attention_weights


class SetAttentionPoolingLayer(nn.Module):
    """
    Set attention pooling layer.
    
    Uses the Set2Set approach for graph-level representations.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension for LSTM
        num_steps: Number of processing steps
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_steps: int = 3):
        super(SetAttentionPoolingLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim + input_dim, 1)
        
    def forward(self, x: torch.Tensor, batch_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through set attention pooling.
        
        Args:
            x: Node feature matrix
            batch_indices: Batch indices
            
        Returns:
            Tuple of (pooled_features, attention_weights)
        """
        batch_size = batch_indices.max().item() + 1
        device = x.device
        
        # Initialize LSTM hidden state
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        
        attention_weights_list = []
        
        for step in range(self.num_steps):
            # LSTM forward pass
            lstm_input = h.transpose(0, 1)  # [batch_size, 1, hidden_dim]
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Expand LSTM output to match nodes
            lstm_expanded = lstm_out.squeeze(1)[batch_indices]  # [num_nodes, hidden_dim]
            
            # Compute attention scores
            combined = torch.cat([x, lstm_expanded], dim=-1)
            attention_scores = self.attention(combined).squeeze(-1)
            
            # Apply softmax within each graph
            attention_weights = torch_scatter.scatter_softmax(attention_scores, batch_indices, dim=0)
            attention_weights_list.append(attention_weights)
            
            # Update LSTM input with attended features
            weighted_features = x * attention_weights.unsqueeze(-1)
            aggregated = torch_scatter.scatter_sum(weighted_features, batch_indices, dim=0)
            h = aggregated.unsqueeze(0)
        
        # Final aggregation
        final_attention = attention_weights_list[-1]
        final_weighted = x * final_attention.unsqueeze(-1)
        pooled = torch_scatter.scatter_sum(final_weighted, batch_indices, dim=0)
        
        return pooled, torch.stack(attention_weights_list, dim=0)


def create_pooling_layer(pooling_type: str, input_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create pooling layers.
    
    Args:
        pooling_type: Type of pooling ('attention', 'mean', 'max', 'sum', 'set_attention')
        input_dim: Input dimension for attention-based pooling
        **kwargs: Additional arguments for specific pooling types
        
    Returns:
        Initialized pooling layer
        
    Raises:
        ValueError: If pooling_type is not supported
    """
    if pooling_type == 'attention':
        return MultiHeadAttentionPoolingLayer(input_dim, **kwargs)
    elif pooling_type == 'mean':
        return MeanPoolingLayer()
    elif pooling_type == 'max':
        return MaxPoolingLayer()
    elif pooling_type == 'sum':
        return SumPoolingLayer()
    elif pooling_type == 'set_attention':
        return SetAttentionPoolingLayer(input_dim, **kwargs)
    else:
        supported = ['attention', 'mean', 'max', 'sum', 'set_attention']
        raise ValueError(f"Unsupported pooling type: {pooling_type}. Supported: {supported}")