"""
Core module for GPU-native molecular processing.

This module contains the fundamental data structures and operations
optimized for batch processing on GPU.
"""

from .batch import MolecularGraphBatch
from .batch_adapter import BatchAdapter
from .engine_config import EngineConfig
from .model_config import ModelConfig
from .preprocessing import SAETransform, StandardScaler, PreprocessingPipeline
from .featurizer import BatchFeaturizer
from .layers import (
    scatter_add,
    ShellConvBlock,
    AttentionPooling,
    FeedForwardNetwork,
    StereochemistryEncoder,
)
from .model import SimplifiedGNN

__all__ = [
    # Data
    "MolecularGraphBatch",
    "BatchFeaturizer",
    "BatchAdapter",
    # Model
    "SimplifiedGNN",
    "ModelConfig",
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
