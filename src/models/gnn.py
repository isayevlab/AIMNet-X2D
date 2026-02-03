"""
Main GNN model for molecular property prediction.

This module contains the primary GNN architecture that combines
shell convolution layers, pooling, and feed-forward networks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ShellConvolutionLayer, MultiLayerPerceptron
from .pooling import create_pooling_layer
from config.constants import MESSAGE_PASSING_RATIO, TETRAHEDRAL_MAGNITUDE_SCALE
from utils.activation import get_activation_function
from utils.logging import get_logger

logger = get_logger(__name__)


def det_3x3_batched(m: torch.Tensor) -> torch.Tensor:
    """
    Fast explicit 3x3 determinant for batched matrices.

    Args:
        m: Tensor of shape (B, 3, 3) containing batch of 3x3 matrices

    Returns:
        Tensor of shape (B,) containing determinants
    """
    return (
        m[:, 0, 0] * (m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1]) -
        m[:, 0, 1] * (m[:, 1, 0] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 0]) +
        m[:, 0, 2] * (m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0])
    )


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
            # Stereochemistry processing layers (legacy)
            self.stereochemical_embedding = nn.Linear(hidden_dim * 3, hidden_dim)
            self.stereochemical_embedding_2 = nn.Linear(self.x_other_dim * 3, self.x_other_dim)

            # V3 tetrahedral chirality layers (oriented volume approach)
            # Pre-normalization for consistent scale
            self.tet_pre_norm = nn.LayerNorm(self.x_other_dim)

            # Projection from embeddings to 3D for determinant computation
            self.tet_W_proj = nn.Linear(self.x_other_dim, 3, bias=False)
            nn.init.xavier_uniform_(self.tet_W_proj.weight, gain=2.0)

            # Output projection from concatenated neighbors
            self.tet_U_out = nn.Linear(4 * self.x_other_dim, self.x_other_dim)
            nn.init.xavier_uniform_(self.tet_U_out.weight)
            nn.init.zeros_(self.tet_U_out.bias)

            # Learnable temperature (clamped in forward to prevent division by zero)
            self.tet_tau_V = nn.Parameter(torch.tensor(1.0))

            # Confidence thresholds for degenerate case handling
            self.tet_eps_conf = 0.01
            self.tet_delta_conf = 0.1

            # Learnable virtual lone pair embedding for pyramidal heteroatoms
            # This replaces the self-reference (center_idx as placeholder) with a
            # semantically meaningful embedding representing the LP "substituent"
            self.virtual_lp_embedding = nn.Parameter(torch.randn(self.x_other_dim) * 0.01)

            # Register constant tensors as buffers (not recreated each forward)
            # Triplet indices for parity-corrected symmetrization
            self.register_buffer('tet_triplet_indices',
                torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long))
            # Signs for antisymmetry: verified that swapping any two neighbors flips V_sym sign
            self.register_buffer('tet_triplet_signs',
                torch.tensor([1.0, -1.0, 1.0, -1.0]))

            # StereoMixer: per-dimension sigmoid gates for combining stereo contributions
            # Unlike softmax, sigmoid gates allow multiple stereo types to contribute
            # independently without competition (each dimension can be 0 to 1)
            # Output dim = D for per-dimension gating (more expressive than scalar gates)
            self.stereo_gate_tet = nn.Linear(self.x_other_dim, self.x_other_dim)
            self.stereo_gate_ez = nn.Linear(self.x_other_dim, self.x_other_dim)
            self.stereo_gate_overall = nn.Linear(self.x_other_dim, self.x_other_dim)
            # Initialize gates for ~50% initial contribution:
            # - Small random weights to break symmetry across dimensions
            # - Per-type bias = 0.5 → sigmoid(0.5) ≈ 0.62
            # - Overall bias = 0.0 → sigmoid(0) = 0.5
            # Total: 0.5 * (0.62 * g_tet + 0.62 * g_ez) ≈ 0.31 per stereo type
            nn.init.normal_(self.stereo_gate_tet.weight, std=0.01)
            nn.init.normal_(self.stereo_gate_ez.weight, std=0.01)
            nn.init.normal_(self.stereo_gate_overall.weight, std=0.01)
            nn.init.constant_(self.stereo_gate_tet.bias, 0.5)
            nn.init.constant_(self.stereo_gate_ez.bias, 0.5)
            nn.init.constant_(self.stereo_gate_overall.bias, 0.0)

            # V3 Allene/Cumulene axial chirality layers
            # Antisymmetric bilinear form: s = d1^T M d2 where M = M_raw - M_raw^T
            # The antisymmetry ensures s flips sign when swapping substituents (P/M enantiomers)
            self.allene_W_a = nn.Linear(self.x_other_dim, self.x_other_dim, bias=False)
            self.allene_W_b = nn.Linear(self.x_other_dim, self.x_other_dim, bias=False)
            # Raw matrix - antisymmetric M is computed as M_raw - M_raw.T
            self.allene_M_raw = nn.Parameter(torch.randn(self.x_other_dim, self.x_other_dim) * 0.01)
            # Output projection
            self.allene_U_out = nn.Linear(self.x_other_dim, self.x_other_dim)
            nn.init.xavier_uniform_(self.allene_W_a.weight)
            nn.init.xavier_uniform_(self.allene_W_b.weight)
            nn.init.xavier_uniform_(self.allene_U_out.weight)
            nn.init.zeros_(self.allene_U_out.bias)
            # Gate for allene contribution (per-dimension like others)
            self.stereo_gate_allene = nn.Linear(self.x_other_dim, self.x_other_dim)
            nn.init.normal_(self.stereo_gate_allene.weight, std=0.01)
            nn.init.constant_(self.stereo_gate_allene.bias, 0.5)

    def forward(self,
                atom_features: dict[str, torch.Tensor],
                multi_hop_edge_indices: torch.Tensor,
                batch_indices: torch.Tensor,
                total_charges: torch.Tensor,
                tetrahedral_indices: torch.Tensor,
                cis_indices: torch.Tensor,
                trans_indices: torch.Tensor,
                chiral_signs: torch.Tensor | None = None,
                chiral_is_virtual_lp: torch.Tensor | None = None,
                allene_centers: torch.Tensor | None = None,
                allene_subs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
            chiral_is_virtual_lp: Boolean mask (M, 4) indicating virtual lone pairs in neighbors
            allene_centers: Indices of allene/cumulene central atoms (shape M_all,)
            allene_subs: Substituent indices for allenes (shape M_all, 4) [R1, R2, R3, R4]

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
            tetrahedral_indices, cis_indices, trans_indices, chiral_signs,
            chiral_is_virtual_lp, allene_centers, allene_subs
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
                                chiral_signs: torch.Tensor | None = None,
                                chiral_is_virtual_lp: torch.Tensor | None = None,
                                allene_centers: torch.Tensor | None = None,
                                allene_subs: torch.Tensor | None = None) -> torch.Tensor:
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
                        x_other_updated, tetrahedral_indices, cis_indices, trans_indices,
                        chiral_signs, chiral_is_virtual_lp, allene_centers, allene_subs
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
                              chiral_signs: torch.Tensor | None = None,
                              chiral_is_virtual_lp: torch.Tensor | None = None,
                              allene_centers: torch.Tensor | None = None,
                              allene_subs: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply stereochemistry features using sigmoid-gated mixing.

        V3 Implementation:
        Unlike softmax which forces competition between stereo types, sigmoid gates
        allow each type (tetrahedral, E/Z, allene) to contribute independently:
        - w_tet, w_ez, w_allene in [0, 1] - each can be high or low independently
        - w_overall in [0, 1] - controls total stereo contribution

        Formula: h_new = h + w_overall * (w_tet * g_tet + w_ez * g_ez + w_allene * g_allene)
        """
        # Compute tetrahedral delta (g_tet = updated - original)
        tetrahedral_updated = self._tetrahedral_feature_calculation_physics_inspired(
            x_other, tetrahedral_indices, chiral_signs, chiral_is_virtual_lp
        )
        g_tet = tetrahedral_updated - x_other

        # Compute E/Z delta (g_ez = updated - original)
        cis_trans_updated = self._cis_trans_calculation(x_other, cis_indices, trans_indices)
        g_ez = cis_trans_updated - x_other

        # Compute allene delta (g_allene = updated - original)
        allene_updated = self._allene_feature_calculation(x_other, allene_centers, allene_subs)
        g_allene = allene_updated - x_other

        # Apply sigmoid gates - per-dimension gating (N, D)
        # Each gate learns independent per-feature weights based on current features
        w_tet = torch.sigmoid(self.stereo_gate_tet(x_other))      # (N, D)
        w_ez = torch.sigmoid(self.stereo_gate_ez(x_other))        # (N, D)
        w_allene = torch.sigmoid(self.stereo_gate_allene(x_other))  # (N, D)

        # Weighted combination of stereo deltas (element-wise gating)
        combined_delta = w_tet * g_tet + w_ez * g_ez + w_allene * g_allene  # (N, D)

        # Overall gate controls total stereo contribution per dimension
        w_overall = torch.sigmoid(self.stereo_gate_overall(x_other))  # (N, D)

        # Apply gated stereo features
        return x_other + w_overall * combined_delta

    def _tetrahedral_feature_calculation_physics_inspired(self,
                                                            atom_features: torch.Tensor,
                                                            tetrahedral_indices: torch.Tensor,
                                                            chiral_signs: torch.Tensor | None = None,
                                                            chiral_is_virtual_lp: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute chirality features using oriented volume with parity-corrected symmetrization.

        V3 Implementation:
        1. Pre-normalize embeddings with LayerNorm for consistent scale
        2. Project neighbor-relative vectors to 3D for determinant computation
        3. Compute parity-corrected oriented volume: (+det012 - det013 + det023 - det123) / 4
        4. Scale-invariant normalization: V_norm = V_sym / (prod(||v||))^(1/3)
        5. Clamped temperature tanh for bounded output
        6. Confidence weighting to downweight degenerate cases
        7. Apply R/S sign for enantiomer distinguishability

        For pyramidal heteroatoms with virtual lone pairs, the 4th neighbor position
        uses a learnable embedding instead of the center atom (which would create
        a zero vector and lose information).

        Args:
            atom_features: Node embeddings (N, D)
            tetrahedral_indices: Shape (M, 5) where [0] is center, [1:5] are neighbors
            chiral_signs: R/S signs (M,) - R=+1, S=-1
            chiral_is_virtual_lp: Boolean mask (M, 4) indicating virtual LPs in neighbor positions

        Returns:
            Updated atom features with chirality information added to center atoms
        """
        if tetrahedral_indices.numel() == 0:
            return atom_features

        # Start with a copy
        updated = atom_features.clone()
        device = atom_features.device
        M = tetrahedral_indices.shape[0]
        D = atom_features.shape[1]

        # Extract center indices and neighbor indices
        center_indices = tetrahedral_indices[:, 0]  # (M,)
        neighbor_indices = tetrahedral_indices[:, 1:5]  # (M, 4)

        # Pre-normalize embeddings for consistent scale
        h_norm = self.tet_pre_norm(atom_features)

        # Get center and neighbor embeddings
        h_center = h_norm[center_indices]  # (M, D)
        h_neigh = h_norm[neighbor_indices]  # (M, 4, D)

        # Handle virtual lone pairs: replace self-reference (center) with learnable LP embedding
        # For pyramidal heteroatoms, the 4th position is a virtual LP (h_neigh == h_center)
        # which would give a zero vector in the determinant. Instead, use a semantically
        # meaningful learnable embedding.
        if chiral_is_virtual_lp is not None and chiral_is_virtual_lp.numel() > 0:
            # chiral_is_virtual_lp: (M, 4) boolean mask
            lp_mask = chiral_is_virtual_lp.to(device)  # (M, 4)
            # Expand LP embedding for broadcasting: (D,) -> (M, 4, D)
            lp_emb_expanded = self.virtual_lp_embedding.unsqueeze(0).unsqueeze(0).expand(M, 4, -1)
            # Apply normalized LP embedding only where mask is True
            lp_emb_norm = self.tet_pre_norm.weight * self.virtual_lp_embedding + self.tet_pre_norm.bias
            lp_emb_norm_expanded = lp_emb_norm.unsqueeze(0).unsqueeze(0).expand(M, 4, -1)
            # Replace virtual LP positions with LP embedding
            h_neigh = torch.where(lp_mask.unsqueeze(-1), lp_emb_norm_expanded, h_neigh)

        # Compute neighbor-relative vectors and project to 3D
        # v_j = W_proj(h_neigh_j - h_center)
        v = self.tet_W_proj(h_neigh - h_center.unsqueeze(1))  # (M, 4, 3)

        # Compute parity-corrected oriented volume using triplet determinants
        # Signs: +det(012), -det(013), +det(023), -det(123)
        # These signs ensure V_sym flips sign under odd permutations (R/S swap)
        # Use pre-registered buffers for efficiency (not recreated each forward)

        # Extract triplets: v_triplets[m, t, :, :] is the (3, 3) matrix for center m, triplet t
        v_triplets = v[:, self.tet_triplet_indices, :]  # (M, 4, 3, 3)

        # Compute determinants for all triplets
        dets = det_3x3_batched(v_triplets.view(-1, 3, 3)).view(M, 4)  # (M, 4)

        # Parity-corrected symmetrization
        V_sym = (dets * self.tet_triplet_signs).sum(dim=1) / 4.0  # (M,)

        # Scale-invariant normalization: divide by geometric mean of edge lengths
        # Use first 3 vectors (avoiding the 4th which may be virtual LP)
        # Clamp norms to prevent numerical instability from near-zero vectors
        v_norms = torch.norm(v[:, :3, :], dim=-1).clamp(min=1e-6)  # (M, 3)
        norm_product = torch.prod(v_norms, dim=1) ** (1.0 / 3.0)  # (M,)
        V_norm = V_sym / norm_product

        # Clamped temperature for stable tanh
        tau = torch.clamp(self.tet_tau_V, min=0.01)
        s = torch.tanh(V_norm / tau)  # (M,)

        # Confidence weighting: downweight near-degenerate cases
        # confidence approaches 0 when |V_norm| is very small
        confidence = torch.sigmoid(
            (torch.abs(V_norm) - self.tet_eps_conf) / self.tet_delta_conf
        )
        s = s * confidence

        # Apply R/S chirality sign for enantiomer distinguishability
        if chiral_signs is not None and chiral_signs.numel() > 0:
            s = s * chiral_signs

        # Compute chiral feature from concatenated neighbor embeddings
        # Use original (non-normalized) embeddings for richer features
        h_neigh_orig = updated[neighbor_indices]  # (M, 4, D)
        # Also replace virtual LP positions in the original embeddings
        if chiral_is_virtual_lp is not None and chiral_is_virtual_lp.numel() > 0:
            lp_mask = chiral_is_virtual_lp.to(device)  # (M, 4)
            lp_emb_expanded = self.virtual_lp_embedding.unsqueeze(0).unsqueeze(0).expand(M, 4, -1)
            h_neigh_orig = torch.where(lp_mask.unsqueeze(-1), lp_emb_expanded, h_neigh_orig)
        h_concat = h_neigh_orig.reshape(M, 4 * D)  # (M, 4D)
        f_tet = s.unsqueeze(-1) * self.tet_U_out(h_concat)  # (M, D)

        # Scatter to center atoms (additive, preserves non-chiral atom features)
        updated.index_add_(0, center_indices, f_tet)

        return updated


    def _cis_trans_calculation(self,
                              atom_features: torch.Tensor,
                              cis_indices: torch.Tensor,
                              trans_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculate cis/trans bond features efficiently.

        Applies cis/trans geometric constraints to bond features
        using scatter operations.

        Args:
            atom_features: Node embeddings (N, D)
            cis_indices: Cis bond pairs, shape (M_cis, 2) as [source, target]
            trans_indices: Trans bond pairs, shape (M_trans, 2) as [source, target]
        """
        if cis_indices.numel() == 0 and trans_indices.numel() == 0:
            return atom_features

        # Get source features for cis and trans bonds
        # Note: tensors have shape (M, 2) where [:, 0] is source and [:, 1] is target
        if cis_indices.numel() > 0:
            source_cis_nodes = cis_indices[:, 0]
            target_cis_nodes = cis_indices[:, 1]
            source_cis_features = atom_features[source_cis_nodes]
        else:
            target_cis_nodes = torch.empty(0, dtype=torch.long, device=atom_features.device)
            source_cis_features = torch.empty(0, atom_features.shape[1], device=atom_features.device)

        if trans_indices.numel() > 0:
            source_trans_nodes = trans_indices[:, 0]
            target_trans_nodes = trans_indices[:, 1]
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

    def _allene_feature_calculation(self,
                                   atom_features: torch.Tensor,
                                   allene_centers: torch.Tensor | None,
                                   allene_subs: torch.Tensor | None) -> torch.Tensor:
        """
        Compute allene/cumulene axial chirality features using antisymmetric bilinear form.

        V3 Implementation:
        For allenes R1R2-C=C=C-R3R4, chirality arises from the perpendicular π-systems.
        We compute: s_allene = d1^T M d2 where M = M_raw - M_raw^T (antisymmetric)

        The antisymmetry ensures:
        - Swapping R1↔R2 or R3↔R4 flips the sign (P/M enantiomers)
        - The scalar s distinguishes between axial enantiomers

        Args:
            atom_features: Node embeddings (N, D)
            allene_centers: Central atom indices for each allene (M_all,)
            allene_subs: Substituent indices (M_all, 4) as [R1, R2, R3, R4]

        Returns:
            Updated atom features with allene chirality information
        """
        if allene_centers is None or allene_subs is None:
            return atom_features
        if allene_centers.numel() == 0:
            return atom_features

        # Start with a copy
        updated = atom_features.clone()
        M_all = allene_centers.shape[0]

        # Extract center indices
        center_indices = allene_centers  # (M_all,)

        # Extract substituent embeddings
        # allene_subs: (M_all, 4) as [R1, R2, R3, R4]
        h_R1 = atom_features[allene_subs[:, 0]]  # (M_all, D)
        h_R2 = atom_features[allene_subs[:, 1]]  # (M_all, D)
        h_R3 = atom_features[allene_subs[:, 2]]  # (M_all, D)
        h_R4 = atom_features[allene_subs[:, 3]]  # (M_all, D)

        # Compute difference vectors for each end
        # d1 = W_a(R1 - R2) captures the asymmetry at the first terminal carbon
        # d2 = W_b(R3 - R4) captures the asymmetry at the second terminal carbon
        d1 = self.allene_W_a(h_R1 - h_R2)  # (M_all, D)
        d2 = self.allene_W_b(h_R3 - h_R4)  # (M_all, D)

        # Compute antisymmetric M matrix: M = M_raw - M_raw^T
        M = self.allene_M_raw - self.allene_M_raw.T  # (D, D)

        # Compute bilinear form: s = d1^T M d2
        # s_allene[i] = sum_j sum_k d1[i,j] * M[j,k] * d2[i,k]
        # = sum_k (d1 @ M)[i,k] * d2[i,k]
        # Scale by sqrt(D) to prevent tanh saturation (similar to attention scaling)
        D = atom_features.shape[1]
        s_allene = torch.sum((d1 @ M) * d2, dim=-1, keepdim=True) / math.sqrt(D)  # (M_all, 1)

        # Create chirality feature from scalar
        # tanh bounds output to [-1, 1], then project to hidden dim
        chirality_sign = torch.tanh(s_allene)  # (M_all, 1)

        # Get center embeddings and combine with chirality signal
        h_center = atom_features[center_indices]  # (M_all, D)
        chirality_feature = chirality_sign * h_center  # (M_all, D)

        # Project and update center atoms
        delta = self.allene_U_out(chirality_feature)  # (M_all, D)
        updated.index_add_(0, center_indices, delta)

        return updated

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

        # Add allene layers if they exist (excluding stereo gates which have custom init)
        if hasattr(self, 'allene_W_a'):
            linear_layers.extend([
                self.allene_W_a,
                self.allene_W_b,
                self.allene_U_out,
                # NOTE: stereo_gate_allene excluded - has custom bias init (0.5) in _create_processing_layers
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