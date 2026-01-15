"""
Core module for GPU-native molecular processing.

This module contains the fundamental data structures and operations
optimized for batch processing on GPU.
"""

from .batch import MolecularGraphBatch
from .preprocessing import SAETransform

__all__ = ["MolecularGraphBatch", "SAETransform"]
