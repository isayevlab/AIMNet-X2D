"""
Inference package for AIMNet-X2D.

This package contains all inference-related functionality organized by purpose.
Provides streaming inference, uncertainty estimation, and embedding extraction.
"""

# Main inference interface
from .engine import InferenceEngine
from .pipeline import InferencePipeline
from .config import InferenceConfig
from .model_loader import ModelLoader
from .results_writer import ResultsWriter
from .uncertainty import MCDropoutPredictor, UncertaintyEstimator
from .embeddings import EmbeddingExtractor, StreamingEmbeddingWriter
from .preprocessing import PreprocessingReconstructor

# Legacy function for backward compatibility
from .engine import inference_main

__all__ = [
    # Main interfaces
    "InferenceEngine",
    "InferencePipeline",
    "InferenceConfig",
    "ModelLoader",
    "ResultsWriter",

    # Uncertainty estimation
    "MCDropoutPredictor",
    "UncertaintyEstimator",

    # Embeddings
    "EmbeddingExtractor",
    "StreamingEmbeddingWriter",

    # Preprocessing
    "PreprocessingReconstructor",

    # Legacy
    "inference_main",
]