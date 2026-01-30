"""
Main GNN model for molecular property prediction.

This module contains the primary GNN architecture that combines
shell convolution layers, pooling, and feed-forward networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ShellConvolutionLayer, MultiLayerPerceptron
from .pooling import create_pooling_layer
from config.constants import MESSAGE_PASSING_RATIO, TETRAHEDRAL_MAGNITUDE_SCALE
from utils.activation import get_activation_function
from utils.logging import get_logger

logger = get_logger(__name__)


class GNN(nn.Module):
    """
    Graph Neural Network for molecular property prediction.
    
    This model uses shell-based convolution layers for message passing,
    configurable pooling for graph-level representations, and feed-forward
    networks for final predictions.
    
    Args:
        feature_sizes: Dictionary of feature dimensions for embeddings
        hidden_dim: Hidden dimension for the model
        output_dim: Output dimension (number of tasks)
        num_shells: Number of shells/hops for message passing
        num_message_passing_layers: Number of message passing layers
        dropout: Dropout probability for message passing
        ffn_hidden_dim: Feed-forward network hidden dimension
        ffn_num_layers: Number of feed-forward layers
        pooling_type: Type of graph pooling
        task_type: Type of task ('regression', 'multitask', 'classification')
        embedding_dim: Embedding dimension for atom features
        use_partial_charges: Whether to use partial charges
        use_stereochemistry: Whether to use stereochemistry features
        ffn_dropout: Dropout rate for feed-forward layers
        activation_type: Type of activation function
        shell_conv_num_mlp_layers: Number of MLP layers in shell convolution
        shell_conv_dropout: Dropout rate for shell convolution
        attention_num_heads: Number of attention heads for attention pooling
        attention_temperature: Initial temperature for attention pooling
        loss_function: Type of loss function ('l1', 'mse', 'evidential')
    """
    
    def __init__(self,
                 feature_sizes: dict[str, int],
                 hidden_dim: int,
                 output_dim: int,
                 num_shells: int = 3,
                 num_message_passing_layers: int = 3,
                 dropout: float = 0.05,
                 ffn_hidden_dim: int | None = None,
                 ffn_num_layers: int = 3, 
                 pooling_type: str = 'attention',
                 task_type: str = 'regression',
                 embedding_dim: int = 64,
                 use_partial_charges: bool = False,
                 use_stereochemistry: bool = False,
                 ffn_dropout: float = 0.05,
                 activation_type: str = "silu",
                 shell_conv_num_mlp_layers: int = 2,
                 shell_conv_dropout: float = 0.05,
                 attention_num_heads: int = 4,
                 attention_temperature: float = 1.0,
                 loss_function: str = "l1"):
        
        super(GNN, self).__init__()

        # Store configuration
        self.hidden_dim = hidden_dim
        self.num_shells = num_shells
        self.task_type = task_type
        self.embedding_dim = embedding_dim
        self.use_partial_charges = use_partial_charges
        self.use_stereochemistry = use_stereochemistry
        self.loss_function = loss_function
        
        if ffn_hidden_dim is None:
            ffn_hidden_dim = hidden_dim

        # Log feature activation status
        logger.info(f"Partial Charges: {self.use_partial_charges}")
        logger.info(f"Stereochemistry: {self.use_stereochemistry}")
        logger.info(f"Loss Function: {self.loss_function}")

        # Embedding layers for atomic features
        self._create_embeddings(feature_sizes, embedding_dim)
        
        # Projection from concatenated embeddings to hidden dimension
        total_embedding_dim = embedding_dim * len(feature_sizes)
        self.embedding_projection = nn.Linear(total_embedding_dim, hidden_dim)
        self.activation = get_activation_function(activation_type)

        # Split hidden representation for message passing and self features
        self.x_other_dim = int(MESSAGE_PASSING_RATIO * hidden_dim)
        self.x_self_dim = hidden_dim - self.x_other_dim

        # Message passing layers
        self._create_message_passing_layers(
            num_message_passing_layers, num_shells, activation_type,
            shell_conv_dropout, shell_conv_num_mlp_layers
        )

        # Pooling layer
        self.pooling = create_pooling_layer(
            pooling_type, 
            hidden_dim,
            num_heads=attention_num_heads,
            initial_temperature=attention_temperature
        )

        # Feature combination and processing layers
        self._create_processing_layers(hidden_dim, activation_type)

        # Feed-forward network
        self.post_pooling_projection = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.ffn = MultiLayerPerceptron(
            input_dim=ffn_hidden_dim,
            hidden_dim=ffn_hidden_dim,
            output_dim=ffn_hidden_dim,
            num_layers=ffn_num_layers,
            activation_type=activation_type,
            dropout=ffn_dropout,
            use_skip=True
        )

        # Output layers
        self.skip_transform = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        
        # Determine final output dimension based on loss function
        if loss_function == "evidential":
            # For evidential learning, output 4 parameters per task
            final_output_dim = output_dim * 4
            logger.info(f"Evidential mode: outputting {final_output_dim} parameters ({output_dim} tasks x 4 params)")
        else:
            final_output_dim = output_dim
            
        self.output_layer = nn.Linear(ffn_hidden_dim * 2, final_output_dim)

        # Additional projection for long-range interactions
        self.long_range_projection = nn.Linear(hidden_dim, ffn_hidden_dim)

        # Initialize weights
        self.init_weights()

    def _create_embeddings(self, feature_sizes: dict[str, int], embedding_dim: int) -> None:
        """Create embedding layers for atomic features."""
        self.atom_type_embedding = nn.Embedding(
            num_embeddings=feature_sizes['atom_type'],
            embedding_dim=embedding_dim
        )
        
        self.hydrogen_count_embedding = nn.Embedding(
            num_embeddings=feature_sizes['hydrogen_count'],
            embedding_dim=embedding_dim
        )
        
        self.degree_embedding = nn.Embedding(
            num_embeddings=feature_sizes['degree'],
            embedding_dim=embedding_dim
        )
        
        self.hybridization_embedding = nn.Embedding(
            num_embeddings=feature_sizes['hybridization'],
            embedding_dim=embedding_dim
        )

    def _create_message_passing_layers(self, num_layers: int, num_shells: int,
                                     activation_type: str, dropout: float, num_mlp_layers: int) -> None:
        """Create message passing layers."""
        self.message_passing_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = ShellConvolutionLayer(
                atom_input_dim=self.x_other_dim, 
                output_dim=self.x_other_dim, 
                num_hops=num_shells, 
                activation_type=activation_type, 
                dropout=dropout, 
                num_mlp_layers=num_mlp_layers
            )
            self.message_passing_layers.append(layer)

    def _create_processing_layers(self, hidden_dim: int, activation_type: str) -> None:
        """Create layers for feature processing and stereochemistry."""
        self.concat_self_other = nn.Linear(hidden_dim, hidden_dim)
        
        if self.use_stereochemistry:
            # Stereochemistry processing layers
            self.stereochemical_embedding = nn.Linear(hidden_dim * 3, hidden_dim)
            self.stereochemical_embedding_2 = nn.Linear(self.x_other_dim * 3, self.x_other_dim)

    def forward(self,
                atom_features: dict[str, torch.Tensor],
                multi_hop_edge_indices: torch.Tensor,
                batch_indices: torch.Tensor,
                total_charges: torch.Tensor,
                tetrahedral_indices: torch.Tensor,
                cis_indices: torch.Tensor,
                trans_indices: torch.Tensor,
                chiral_signs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Forward pass through the GNN.

        Args:
            atom_features: Dictionary of atomic features
            multi_hop_edge_indices: Edge indices for message passing
            batch_indices: Batch indices for each atom
            total_charges: Total formal charges for each molecule
            tetrahedral_indices: Indices for tetrahedral chiral centers (shape M, 5)
            cis_indices: Indices for cis bonds
            trans_indices: Indices for trans bonds
            chiral_signs: R/S chirality signs for each chiral center (R=+1, S=-1)

        Returns:
            Tuple of (predictions, attention_weights, partial_charges)
        """
        # Embed atomic features
        atom_embeddings = self._embed_atomic_features(atom_features)
        
        # Project to hidden dimension and split
        atom_embeddings = self.embedding_projection(atom_embeddings)
        atom_embeddings = self.activation(atom_embeddings)
        
        x_self, x_other = torch.split(
            atom_embeddings, 
            [self.x_self_dim, self.x_other_dim], 
            dim=-1
        )

        # Message passing with optional features
        x_other_updated = self._message_passing_forward(
            x_other, multi_hop_edge_indices, batch_indices, total_charges,
            tetrahedral_indices, cis_indices, trans_indices, chiral_signs
        )

        # Extract partial charges if enabled
        partial_charges = None
        if self.use_partial_charges and x_other_updated.shape[-1] >= 2:
            partial_charges = x_other_updated[:, 0].clone()

        # Combine self and other features
        x_combined = torch.cat([x_self, x_other_updated], dim=-1)
        x = self.concat_self_other(x_combined)

        # Pool to graph-level representation
        x_pooled, attention_weights = self.pooling(x, batch_indices)
        
        # Feed-forward processing
        x = self.post_pooling_projection(x_pooled)
        x = self.ffn(x)
        
        # Final output with skip connection
        skip_connection = self.skip_transform(x)
        final_features = torch.cat([x, skip_connection], dim=-1)
        output = self.output_layer(final_features)

        return output, attention_weights, partial_charges

    def _embed_atomic_features(self, atom_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed and concatenate atomic features."""
        atom_type_emb = self.atom_type_embedding(atom_features['atom_type'])
        hydrogen_count_emb = self.hydrogen_count_embedding(atom_features['hydrogen_count'])
        degree_emb = self.degree_embedding(atom_features['degree'])
        hybridization_emb = self.hybridization_embedding(atom_features['hybridization'])

        return torch.cat([
            atom_type_emb,
            hydrogen_count_emb,
            degree_emb,
            hybridization_emb,
        ], dim=-1)

    def _message_passing_forward(self,
                                x_other: torch.Tensor,
                                multi_hop_edge_indices: torch.Tensor,
                                batch_indices: torch.Tensor,
                                total_charges: torch.Tensor,
                                tetrahedral_indices: torch.Tensor,
                                cis_indices: torch.Tensor,
                                trans_indices: torch.Tensor,
                                chiral_signs: torch.Tensor | None = None) -> torch.Tensor:
        """Perform message passing with optional features."""
        x_other_updated = x_other

        if multi_hop_edge_indices.numel() > 0:
            for layer in self.message_passing_layers:
                # Apply partial charge calculation if enabled
                if self.use_partial_charges:
                    x_other_updated = self._partial_charge_calculation(
                        x_other_updated, batch_indices, total_charges
                    )

                # Apply stereochemistry features if enabled
                if self.use_stereochemistry:
                    x_other_updated = self._apply_stereochemistry(
                        x_other_updated, tetrahedral_indices, cis_indices, trans_indices, chiral_signs
                    )

                # Message passing
                x_other_updated = layer(
                    x_other_updated,
                    multi_hop_edge_indices[:, 0],
                    multi_hop_edge_indices[:, 1]
                ) + x_other_updated

        return x_other_updated

    def _apply_stereochemistry(self,
                              x_other: torch.Tensor,
                              tetrahedral_indices: torch.Tensor,
                              cis_indices: torch.Tensor,
                              trans_indices: torch.Tensor,
                              chiral_signs: torch.Tensor | None = None) -> torch.Tensor:
        """Apply stereochemistry features."""
        cis_trans_features = self._cis_trans_calculation(x_other, cis_indices, trans_indices)

        tetrahedral_features = self._tetrahedral_feature_calculation_physics_inspired(
            x_other, tetrahedral_indices, chiral_signs
        )

        x_concat_stereochemistry = torch.cat([
            x_other, cis_trans_features, tetrahedral_features
        ], dim=-1)

        return self.stereochemical_embedding_2(x_concat_stereochemistry)

    def _tetrahedral_feature_calculation_physics_inspired(self,
                                                            atom_features: torch.Tensor,
                                                            tetrahedral_indices: torch.Tensor,
                                                            chiral_signs: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute chirality features using physics-inspired approach with R/S sign encoding.

        Key fixes from original:
        1. Uses chiral_signs (R=+1, S=-1) to distinguish enantiomers
        2. tetrahedral_indices now shape (M, 5): [center_idx, n1, n2, n3, n4]
        3. NO LONGER zeros out non-chiral atoms (was destroying information)
        4. Features are applied to CENTER atoms only

        Original equation preserved:
        chirality_features = (
            squares_1 * (emb_2 - emb_3) +
            squares_2 * (emb_3 - emb_1) +
            squares_3 * (emb_1 - emb_2)
        )

        Physics insight applied: Work with normalized directions first, then scale back.
        """
        if tetrahedral_indices.numel() == 0:
            return atom_features

        # Start with a copy
        updated = atom_features.clone()

        # Extract center indices and neighbor indices
        # tetrahedral_indices shape: (M, 5) where [0] is center, [1:5] are neighbors
        center_indices = tetrahedral_indices[:, 0]  # (M,)
        neighbor_indices = tetrahedral_indices[:, 1:5]  # (M, 4)

        # Get neighbor embeddings
        emb_raw = updated[neighbor_indices]  # Shape: (M, 4, D)

        # Extract magnitude information to preserve later
        emb_magnitudes = torch.norm(emb_raw, dim=-1, keepdim=True)  # (M, 4, 1)

        # Work with normalized directions (prevents explosion)
        emb = F.normalize(emb_raw, dim=-1, eps=1e-8)  # (M, 4, D) - unit vectors

        # Compute squares of normalized vectors (bounded to [0, 1])
        squares = emb ** 2
        squares_1 = torch.roll(squares, shifts=-1, dims=1)
        squares_2 = torch.roll(squares, shifts=-2, dims=1)
        squares_3 = torch.roll(squares, shifts=-3, dims=1)

        emb_1 = torch.roll(emb, shifts=-1, dims=1)
        emb_2 = torch.roll(emb, shifts=-2, dims=1)
        emb_3 = torch.roll(emb, shifts=-3, dims=1)

        # Original chirality equation
        chirality_features = (
            squares_1 * (emb_2 - emb_3) +
            squares_2 * (emb_3 - emb_1) +
            squares_3 * (emb_1 - emb_2)
        )

        # Use the average magnitude of the 4 substituents for each center
        avg_magnitude = torch.mean(emb_magnitudes, dim=1, keepdim=True)  # (M, 1, 1)

        # Apply soft magnitude scaling (bounded growth)
        magnitude_scale = torch.tanh(avg_magnitude / TETRAHEDRAL_MAGNITUDE_SCALE)

        # Scale by magnitude
        chirality_features = chirality_features * magnitude_scale

        # Apply R/S chirality sign - THIS IS THE KEY FIX FOR ENANTIOMER DISTINGUISHABILITY
        # R configuration -> +1, S configuration -> -1
        # This preserves the mirror-image relationship between enantiomers
        if chiral_signs is not None and chiral_signs.numel() > 0:
            sign_scale = chiral_signs.view(-1, 1, 1)  # (M, 1, 1)
            chirality_features = chirality_features * sign_scale

        # Sum across the 4 neighbors to get per-center features
        chirality_summed = chirality_features.sum(dim=1)  # (M, D)

        # Apply chirality features to CENTER atoms (not neighbors)
        # This is additive - we don't zero out non-chiral atoms
        updated.index_add_(0, center_indices, chirality_summed)

        # NO ZEROING OF NON-CHIRAL ATOMS
        # Previous bug: updated[~mask] = 0.0 destroyed all non-chiral atom features
        # Non-chiral atoms contain important chemical information (ring atoms, functional groups, etc.)

        return updated


    def _cis_trans_calculation(self, 
                              atom_features: torch.Tensor, 
                              cis_indices: torch.Tensor, 
                              trans_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculate cis/trans bond features efficiently.
        
        Applies cis/trans geometric constraints to bond features
        using scatter operations.
        """
        if cis_indices.numel() == 0 and trans_indices.numel() == 0:
            return atom_features

        # Get source features for cis and trans bonds
        if cis_indices.numel() > 0:
            source_cis_nodes = cis_indices[0]
            target_cis_nodes = cis_indices[1]
            source_cis_features = atom_features[source_cis_nodes]
        else:
            target_cis_nodes = torch.empty(0, dtype=torch.long, device=atom_features.device)
            source_cis_features = torch.empty(0, atom_features.shape[1], device=atom_features.device)

        if trans_indices.numel() > 0:
            source_trans_nodes = trans_indices[0]
            target_trans_nodes = trans_indices[1]
            source_trans_features = atom_features[source_trans_nodes]
        else:
            target_trans_nodes = torch.empty(0, dtype=torch.long, device=atom_features.device)
            source_trans_features = torch.empty(0, atom_features.shape[1], device=atom_features.device)

        # Combine targets and sources (cis gets negative, trans gets positive)
        all_targets = torch.cat([target_cis_nodes, target_trans_nodes], dim=0)
        all_sources = torch.cat([-source_cis_features, source_trans_features], dim=0)

        # Apply updates via scatter_add
        if all_targets.numel() > 0:
            updated_features = atom_features.scatter_add(
                dim=0,
                index=all_targets.unsqueeze(1).expand(-1, atom_features.shape[1]),
                src=all_sources
            )
        else:
            updated_features = atom_features

        return updated_features

    def _partial_charge_calculation(self, 
                                   atom_features: torch.Tensor, 
                                   batch_indices: torch.Tensor, 
                                   total_charges: torch.Tensor) -> torch.Tensor:
        """
        Calculate partial charges using charge conservation.
        
        Implements the charge equilibration method for partial charge
        computation with molecular charge constraints.
        """
        # Split features into charge, electronegativity, and others
        _q, _f, delta_a = atom_features.split([1, 1, atom_features.shape[-1] - 2], dim=-1)
        _f = torch.clamp(_f, min=1e-6)

        # Aggregate charges and electronegativities per molecule
        target_shape = (total_charges.shape[0], _q.shape[1])
        Q_u = torch.zeros(target_shape, device=_q.device).scatter_add(
            0, batch_indices.unsqueeze(1), _q
        )
        
        F_u = torch.zeros(target_shape, device=_f.device).scatter_add(
            0, batch_indices.unsqueeze(1), _f
        ) + 1e-6

        F_u = torch.clamp(F_u, min=1e-6)
        
        # Calculate charge difference
        dQ = total_charges.unsqueeze(-1) - Q_u

        # Distribute charge difference proportionally
        F_u_expanded = F_u[batch_indices]
        dQ_expanded = dQ[batch_indices]

        f_new = _f / F_u_expanded
        q_new = _q + f_new * dQ_expanded

        return torch.cat([q_new, f_new, delta_a], dim=-1)

    def init_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        # Linear layers to initialize
        linear_layers = [
            self.embedding_projection,
            self.concat_self_other,
            self.post_pooling_projection,
            self.skip_transform,
            self.output_layer,
            self.long_range_projection,
        ]

        # Add stereochemistry layers if they exist
        if hasattr(self, 'stereochemical_embedding'):
            linear_layers.extend([
                self.stereochemical_embedding,
                self.stereochemical_embedding_2
            ])

        # Initialize linear layers
        for layer in linear_layers:
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Initialize embeddings
        for embedding in [self.atom_type_embedding, self.degree_embedding, 
                         self.hybridization_embedding, self.hydrogen_count_embedding]:
            nn.init.xavier_uniform_(embedding.weight)

        # Initialize message passing layers
        for layer in self.message_passing_layers:
            if hasattr(layer, 'init_weights'):
                layer.init_weights()

        # Initialize pooling layer if it has weights
        if hasattr(self.pooling, 'attention_weights'):
            for attention_weight in self.pooling.attention_weights:
                nn.init.xavier_uniform_(attention_weight.weight)
                if attention_weight.bias is not None:
                    nn.init.zeros_(attention_weight.bias)

        logger.info("Model weights initialized")

    def get_model_info(self) -> dict[str, any]:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_dim': self.hidden_dim,
            'num_shells': self.num_shells,
            'embedding_dim': self.embedding_dim,
            'task_type': self.task_type,
            'use_partial_charges': self.use_partial_charges,
            'use_stereochemistry': self.use_stereochemistry,
            'loss_function': self.loss_function,
            'num_message_passing_layers': len(self.message_passing_layers),
            'pooling_type': type(self.pooling).__name__,
        }

    def __repr__(self) -> str:
        """String representation of the model."""
        info = self.get_model_info()
        return (f"GNN(\n"
                f"  parameters={info['total_parameters']:,}\n"
                f"  hidden_dim={info['hidden_dim']}\n"
                f"  num_shells={info['num_shells']}\n"
                f"  task_type='{info['task_type']}'\n"
                f"  loss_function='{info['loss_function']}'\n"
                f"  features=[partial_charges={info['use_partial_charges']}, "
                f"stereochemistry={info['use_stereochemistry']}]\n"
                f")")


class GNNConfig:
    """Configuration helper for GNN model creation."""
    
    @staticmethod
    def from_args(args) -> dict[str, any]:
        """Create GNN configuration from command line arguments."""
        # Extract feature sizes (this would typically come from your data module)
        feature_sizes = {
            'atom_type': 119,  # This should be imported from your molecular module
            'hydrogen_count': 9,
            'degree': 7,
            'hybridization': 7,
        }
        
        config = {
            'feature_sizes': feature_sizes,
            'hidden_dim': args.hidden_dim,
            'output_dim': getattr(args, 'output_dim', 1),
            'num_shells': args.num_shells,
            'num_message_passing_layers': args.num_message_passing_layers,
            'ffn_hidden_dim': args.ffn_hidden_dim,
            'ffn_num_layers': args.ffn_num_layers,
            'pooling_type': args.pooling_type,
            'task_type': args.task_type,
            'embedding_dim': args.embedding_dim,
            'use_partial_charges': args.use_partial_charges,
            'use_stereochemistry': args.use_stereochemistry,
            'ffn_dropout': args.ffn_dropout,
            'activation_type': args.activation_type,
            'shell_conv_num_mlp_layers': args.shell_conv_num_mlp_layers,
            'shell_conv_dropout': args.shell_conv_dropout,
            'attention_num_heads': args.attention_num_heads,
            'attention_temperature': args.attention_temperature,
            'loss_function': args.loss_function,
        }
        
        return config

    @staticmethod
    def create_model_from_args(args) -> GNN:
        """Create GNN model from command line arguments."""
        config = GNNConfig.from_args(args)
        return GNN(**config)