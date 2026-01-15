"""
Core module for GPU-native molecular processing.

This module contains the fundamental data structures and operations
optimized for batch processing on GPU.
"""

from .batch import MolecularGraphBatch
from .model_config import ModelConfig
from .preprocessing import SAETransform, StandardScaler, PreprocessingPipeline
from .featurizer import BatchFeaturizer
from .layers import ShellConvBlock, AttentionPooling, FeedForwardNetwork

__all__ = [
    "MolecularGraphBatch",
    "ModelConfig",
    "SAETransform",
    "StandardScaler",
    "PreprocessingPipeline",
    "BatchFeaturizer",
    "ShellConvBlock",
    "AttentionPooling",
    "FeedForwardNetwork",
]
