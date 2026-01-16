"""
Core module for GPU-native molecular GNN.

This module provides the refactored architecture with:
- MolecularGraphBatch: Batched molecular data
- BatchFeaturizer: SMILES to batch conversion
- SimplifiedGNN: Main GNN model
- Engine: Unified training and inference
- Preprocessing: SAE and scaling transforms
"""

from .batch import MolecularGraphBatch
from .batch_adapter import BatchAdapter
from .featurizer import BatchFeaturizer
from .model import SimplifiedGNN
from .model_config import ModelConfig
from .engine import Engine
from .engine_config import EngineConfig
from .preprocessing import (
    SAETransform,
    StandardScaler,
    PreprocessingPipeline,
)
from .layers import (
    scatter_add,
    ShellConvBlock,
    AttentionPooling,
    FeedForwardNetwork,
    StereochemistryEncoder,
)

__all__ = [
    # Data
    "MolecularGraphBatch",
    "BatchFeaturizer",
    "BatchAdapter",
    # Model
    "SimplifiedGNN",
    "ModelConfig",
    # Engine
    "Engine",
    "EngineConfig",
    # Preprocessing
    "SAETransform",
    "StandardScaler",
    "PreprocessingPipeline",
    # Layers
    "scatter_add",
    "ShellConvBlock",
    "AttentionPooling",
    "FeedForwardNetwork",
    "StereochemistryEncoder",
]
