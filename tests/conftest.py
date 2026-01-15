"""
Pytest configuration and fixtures for AIMNet-X2D tests.
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import torch

# Add src directory to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


# =============================================================================
# SMILES Fixtures
# =============================================================================

@pytest.fixture
def valid_smiles_simple():
    """Simple valid SMILES strings."""
    return ["C", "CC", "CCC", "CCO", "CCN"]


@pytest.fixture
def valid_smiles_complex():
    """Complex valid SMILES strings with rings and branches."""
    return [
        "c1ccccc1",           # benzene
        "c1ccc(O)cc1",        # phenol
        "CC(=O)O",            # acetic acid
        "c1ccc2ccccc2c1",     # naphthalene
        "CC(C)CC",            # isopentane
    ]


@pytest.fixture
def valid_smiles_chiral():
    """SMILES with tetrahedral chirality."""
    return [
        "C[C@H](O)F",         # chiral center
        "C[C@@H](Cl)Br",      # opposite chirality
        "F[C@H](Cl)Br",       # halogenated chiral
    ]


@pytest.fixture
def valid_smiles_stereobonds():
    """SMILES with E/Z stereochemistry."""
    return [
        "C/C=C/C",            # trans-2-butene
        r"C/C=C\C",           # cis-2-butene
        "C/C=C/C=C/C",        # all-trans
    ]


@pytest.fixture
def invalid_smiles():
    """Invalid SMILES strings that should fail parsing."""
    return [
        "invalid",
        "xxx",
        "C(C(C",              # unclosed parentheses
        "",                   # empty string
        "C1CC",               # unclosed ring
    ]


@pytest.fixture
def all_valid_smiles(valid_smiles_simple, valid_smiles_complex, valid_smiles_chiral):
    """All valid SMILES combined."""
    return valid_smiles_simple + valid_smiles_complex + valid_smiles_chiral


# =============================================================================
# Target Value Fixtures
# =============================================================================

@pytest.fixture
def single_targets():
    """Single float targets for regression."""
    return [1.5, -0.3, 2.7, 0.0, -1.2]


@pytest.fixture
def multitask_targets():
    """Multi-task targets (list of lists)."""
    return [
        [1.5, 2.3, -0.8],
        [-0.3, 1.1, 0.5],
        [2.7, -1.2, 1.9],
        [0.0, 0.5, 0.5],
        [-1.2, 2.0, -0.3],
    ]


@pytest.fixture
def outlier_targets():
    """Targets with outliers for percentile cutoff testing."""
    return [1.0, 1.1, 1.2, 100.0, 1.0, -100.0, 0.9, 1.1]


# =============================================================================
# Model Configuration Fixtures
# =============================================================================

@pytest.fixture
def minimal_feature_sizes():
    """Minimal feature sizes for fast testing."""
    return {
        'atom_type': 119,
        'hydrogen_count': 9,
        'degree': 7,
        'hybridization': 7,
    }


@pytest.fixture
def minimal_model_config(minimal_feature_sizes):
    """Minimal GNN configuration for unit tests."""
    return {
        'feature_sizes': minimal_feature_sizes,
        'hidden_dim': 32,
        'output_dim': 1,
        'num_shells': 2,
        'num_message_passing_layers': 1,
        'ffn_hidden_dim': 32,
        'ffn_num_layers': 1,
        'pooling_type': 'mean',
        'task_type': 'regression',
        'embedding_dim': 16,
        'use_partial_charges': False,
        'use_stereochemistry': False,
        'ffn_dropout': 0.0,
        'activation_type': 'relu',
        'shell_conv_num_mlp_layers': 1,
        'shell_conv_dropout': 0.0,
        'attention_num_heads': 2,
        'attention_temperature': 1.0,
        'loss_function': 'l1',
    }


@pytest.fixture
def multitask_model_config(minimal_feature_sizes):
    """Multi-task GNN configuration."""
    return {
        'feature_sizes': minimal_feature_sizes,
        'hidden_dim': 32,
        'output_dim': 3,
        'num_shells': 2,
        'num_message_passing_layers': 1,
        'ffn_hidden_dim': 32,
        'ffn_num_layers': 1,
        'pooling_type': 'mean',
        'task_type': 'multitask',
        'embedding_dim': 16,
        'use_partial_charges': False,
        'use_stereochemistry': False,
        'ffn_dropout': 0.0,
        'activation_type': 'relu',
        'shell_conv_num_mlp_layers': 1,
        'shell_conv_dropout': 0.0,
        'attention_num_heads': 2,
        'attention_temperature': 1.0,
        'loss_function': 'l1',
    }


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def cpu_device():
    """Force CPU device for deterministic tests."""
    return torch.device('cpu')


# =============================================================================
# Seed Fixtures
# =============================================================================

@pytest.fixture(autouse=False)
def set_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def tmp_model_path(tmp_path):
    """Temporary path for saving models."""
    return tmp_path / "model.pth"


@pytest.fixture
def tmp_hdf5_path(tmp_path):
    """Temporary path for HDF5 files."""
    return tmp_path / "data.h5"


@pytest.fixture
def tmp_csv_path(tmp_path):
    """Temporary path for CSV files."""
    return tmp_path / "data.csv"
