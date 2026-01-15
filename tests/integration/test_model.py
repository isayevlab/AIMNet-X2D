# tests/integration/test_model.py
"""
Integration tests for model creation and forward pass.

Note: These tests require torch_scatter to be installed.
"""

import pytest
import torch
import numpy as np

# Check if torch_scatter is available
torch_scatter_available = False
try:
    import torch_scatter
    torch_scatter_available = True
except ImportError:
    pass

# Mark all tests in this module as integration tests and skip if torch_scatter not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch_scatter_available, reason="torch_scatter not installed")
]


@pytest.fixture
def minimal_model_config():
    """Minimal model configuration for fast testing."""
    return {
        'hidden_dim': 32,
        'output_dim': 1,
        'num_shells': 2,
        'num_message_passing_layers': 1,
        'dropout': 0.0,
        'ffn_hidden_dim': 32,
        'ffn_num_layers': 1,
        'pooling_type': 'mean',
        'task_type': 'regression',
        'embedding_dim': 16,
        'use_partial_charges': False,
        'use_stereochemistry': False,
        'ffn_dropout': 0.0,
        'activation_type': 'silu',
        'shell_conv_num_mlp_layers': 1,
        'shell_conv_dropout': 0.0,
        'attention_num_heads': 2,
        'attention_temperature': 1.0,
        'loss_function': 'l1',
    }


@pytest.fixture
def feature_sizes():
    """Feature sizes for embeddings."""
    return {
        'atom_type': 119,  # Elements 0-118
        'hydrogen_count': 10,
        'degree': 6,
        'hybridization': 7,
    }


class TestModelCreation:
    """Tests for GNN model creation."""

    def test_create_gnn_model(self, minimal_model_config, feature_sizes):
        """Test creating a GNN model."""
        from models import GNN

        model = GNN(feature_sizes=feature_sizes, **minimal_model_config)

        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_parameter_count(self, minimal_model_config, feature_sizes):
        """Test that model has reasonable parameter count."""
        from models import GNN

        model = GNN(feature_sizes=feature_sizes, **minimal_model_config)

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        # With minimal config, should be relatively small
        assert param_count < 1_000_000

    def test_model_trainable_parameters(self, minimal_model_config, feature_sizes):
        """Test that model has trainable parameters."""
        from models import GNN

        model = GNN(feature_sizes=feature_sizes, **minimal_model_config)

        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_count > 0


class TestModelForwardPass:
    """Tests for GNN model forward pass."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        from datasets import PyGSMILESDataset, MolecularBatch
        from datasets.features import precompute_all_and_filter

        smiles_list = ["C", "CC", "CCC"]
        targets = [1.0, 2.0, 3.0]

        valid_smiles, valid_targets, precomputed = precompute_all_and_filter(
            smiles_list, targets, max_hops=2, num_workers=1
        )

        dataset = PyGSMILESDataset(
            smiles_list=valid_smiles,
            targets=valid_targets,
            precomputed_data=precomputed
        )

        data_list = [dataset[i] for i in range(len(dataset))]
        return MolecularBatch.from_data_list(data_list)

    def test_forward_pass_cpu(self, minimal_model_config, feature_sizes, sample_batch):
        """Test model forward pass on CPU."""
        from models import GNN

        # Adjust num_shells to match batch
        minimal_model_config['num_shells'] = 2

        model = GNN(feature_sizes=feature_sizes, **minimal_model_config)
        model.eval()

        device = torch.device('cpu')

        # Prepare batch tensors
        atom_features = {k: v.to(device) for k, v in sample_batch.atom_features_map.items()}
        multi_hop_edges = sample_batch.multi_hop_edge_indices.to(device)
        batch_indices = sample_batch.batch_indices.to(device)
        total_charges = sample_batch.total_charges.to(device)
        tetrahedral = sample_batch.final_tetrahedral_chiral_tensor.to(device)
        cis = sample_batch.final_cis_tensor.to(device)
        trans = sample_batch.final_trans_tensor.to(device)

        with torch.no_grad():
            output, attention_weights, partial_charges = model(
                atom_features,
                multi_hop_edges,
                batch_indices,
                total_charges,
                tetrahedral,
                cis,
                trans
            )

        assert output is not None
        assert output.shape[0] == 3  # 3 molecules
        assert output.shape[1] == 1  # 1 output dim

    def test_forward_pass_produces_finite_values(self, minimal_model_config, feature_sizes, sample_batch):
        """Test that forward pass produces finite values."""
        from models import GNN

        minimal_model_config['num_shells'] = 2

        model = GNN(feature_sizes=feature_sizes, **minimal_model_config)
        model.eval()

        device = torch.device('cpu')

        atom_features = {k: v.to(device) for k, v in sample_batch.atom_features_map.items()}
        multi_hop_edges = sample_batch.multi_hop_edge_indices.to(device)
        batch_indices = sample_batch.batch_indices.to(device)
        total_charges = sample_batch.total_charges.to(device)
        tetrahedral = sample_batch.final_tetrahedral_chiral_tensor.to(device)
        cis = sample_batch.final_cis_tensor.to(device)
        trans = sample_batch.final_trans_tensor.to(device)

        with torch.no_grad():
            output, _, _ = model(
                atom_features,
                multi_hop_edges,
                batch_indices,
                total_charges,
                tetrahedral,
                cis,
                trans
            )

        assert torch.isfinite(output).all()

    def test_forward_pass_gradient_computation(self, minimal_model_config, feature_sizes, sample_batch):
        """Test that gradients can be computed."""
        from models import GNN

        minimal_model_config['num_shells'] = 2

        model = GNN(feature_sizes=feature_sizes, **minimal_model_config)
        model.train()

        device = torch.device('cpu')

        atom_features = {k: v.to(device) for k, v in sample_batch.atom_features_map.items()}
        multi_hop_edges = sample_batch.multi_hop_edge_indices.to(device)
        batch_indices = sample_batch.batch_indices.to(device)
        total_charges = sample_batch.total_charges.to(device)
        tetrahedral = sample_batch.final_tetrahedral_chiral_tensor.to(device)
        cis = sample_batch.final_cis_tensor.to(device)
        trans = sample_batch.final_trans_tensor.to(device)

        output, _, _ = model(
            atom_features,
            multi_hop_edges,
            batch_indices,
            total_charges,
            tetrahedral,
            cis,
            trans
        )

        # Compute loss and backprop
        loss = output.sum()
        loss.backward()

        # Check that gradients were computed
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients were computed"


class TestDifferentPoolingTypes:
    """Tests for different pooling strategies."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        from datasets import PyGSMILESDataset, MolecularBatch
        from datasets.features import precompute_all_and_filter

        smiles_list = ["C", "CC", "CCC"]
        targets = [1.0, 2.0, 3.0]

        valid_smiles, valid_targets, precomputed = precompute_all_and_filter(
            smiles_list, targets, max_hops=2, num_workers=1
        )

        dataset = PyGSMILESDataset(
            smiles_list=valid_smiles,
            targets=valid_targets,
            precomputed_data=precomputed
        )

        data_list = [dataset[i] for i in range(len(dataset))]
        return MolecularBatch.from_data_list(data_list)

    @pytest.mark.parametrize("pooling_type", ["mean", "max", "sum", "attention"])
    def test_different_pooling_types(self, pooling_type, feature_sizes, sample_batch):
        """Test model with different pooling types."""
        from models import GNN

        config = {
            'hidden_dim': 32,
            'output_dim': 1,
            'num_shells': 2,
            'num_message_passing_layers': 1,
            'dropout': 0.0,
            'ffn_hidden_dim': 32,
            'ffn_num_layers': 1,
            'pooling_type': pooling_type,
            'task_type': 'regression',
            'embedding_dim': 16,
            'use_partial_charges': False,
            'use_stereochemistry': False,
            'ffn_dropout': 0.0,
            'activation_type': 'silu',
            'shell_conv_num_mlp_layers': 1,
            'shell_conv_dropout': 0.0,
            'attention_num_heads': 2,
            'attention_temperature': 1.0,
            'loss_function': 'l1',
        }

        model = GNN(feature_sizes=feature_sizes, **config)
        model.eval()

        device = torch.device('cpu')

        atom_features = {k: v.to(device) for k, v in sample_batch.atom_features_map.items()}
        multi_hop_edges = sample_batch.multi_hop_edge_indices.to(device)
        batch_indices = sample_batch.batch_indices.to(device)
        total_charges = sample_batch.total_charges.to(device)
        tetrahedral = sample_batch.final_tetrahedral_chiral_tensor.to(device)
        cis = sample_batch.final_cis_tensor.to(device)
        trans = sample_batch.final_trans_tensor.to(device)

        with torch.no_grad():
            output, attention_weights, _ = model(
                atom_features,
                multi_hop_edges,
                batch_indices,
                total_charges,
                tetrahedral,
                cis,
                trans
            )

        assert output is not None
        assert output.shape == (3, 1)

        # Attention pooling should return attention weights
        if pooling_type == 'attention':
            assert attention_weights is not None
        else:
            assert attention_weights is None


class TestLossFunctions:
    """Tests for different loss functions."""

    @pytest.mark.parametrize("loss_function", ["l1", "mse", "evidential"])
    def test_different_loss_functions(self, loss_function, feature_sizes):
        """Test model creation with different loss functions."""
        from models import GNN

        config = {
            'hidden_dim': 32,
            'output_dim': 4 if loss_function == 'evidential' else 1,  # Evidential needs 4x output
            'num_shells': 2,
            'num_message_passing_layers': 1,
            'dropout': 0.0,
            'ffn_hidden_dim': 32,
            'ffn_num_layers': 1,
            'pooling_type': 'mean',
            'task_type': 'regression',
            'embedding_dim': 16,
            'use_partial_charges': False,
            'use_stereochemistry': False,
            'ffn_dropout': 0.0,
            'activation_type': 'silu',
            'shell_conv_num_mlp_layers': 1,
            'shell_conv_dropout': 0.0,
            'attention_num_heads': 2,
            'attention_temperature': 1.0,
            'loss_function': loss_function,
        }

        model = GNN(feature_sizes=feature_sizes, **config)

        assert model is not None
        assert model.loss_function == loss_function
