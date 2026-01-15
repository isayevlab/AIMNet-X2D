"""
Configuration classes for inference pipeline.
"""

from dataclasses import dataclass
from pathlib import Path

from datasets.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MC_DROPOUT_RATE,
)


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""

    # Input/Output
    input_path: str
    output_path: str
    smiles_column: str = "smiles"

    # Model settings
    model_path: str = None
    batch_size: int = 64
    max_hops: int = None  # Will be set from model

    # Processing settings
    chunk_size: int = DEFAULT_CHUNK_SIZE
    num_workers: int = 4

    # Uncertainty estimation
    mc_samples: int = 0
    mc_dropout_rate: float = DEFAULT_MC_DROPOUT_RATE
    
    # Embeddings
    save_embeddings: bool = False
    embeddings_path: str = "embeddings.h5"
    include_atom_embeddings: bool = False
    
    # Performance
    mixed_precision: bool = False
    
    # DDP settings
    ddp_enabled: bool = False
    rank: int = 0
    world_size: int = 1
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_paths()
        self._validate_parameters()
    
    def _validate_paths(self):
        """Validate input/output paths."""
        if not self.input_path:
            raise ValueError("input_path must be specified")
        
        # Check if input file exists
        from pathlib import Path
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        if not self.output_path:
            raise ValueError("output_path must be specified")
        
        # Create output directory if it doesn't exist
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create embeddings directory if needed
        if self.save_embeddings:
            Path(self.embeddings_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _validate_parameters(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.mc_samples < 0:
            raise ValueError("mc_samples must be non-negative")
        
        if not 0 <= self.mc_dropout_rate <= 1:
            raise ValueError("mc_dropout_rate must be between 0 and 1")
    
    @classmethod
    def from_args(cls, args) -> 'InferenceConfig':
        """Create config from command line arguments."""
        return cls(
            input_path=args.inference_csv or args.inference_hdf5,
            output_path=args.inference_output,
            smiles_column=args.smiles_column,
            model_path=args.model_save_path,
            batch_size=args.stream_batch_size or args.batch_size,
            max_hops=None,  # Will be loaded from model
            chunk_size=args.stream_chunk_size,
            num_workers=args.num_workers,
            mc_samples=args.mc_samples,
            save_embeddings=args.save_embeddings,
            embeddings_path=args.embeddings_output_path,
            include_atom_embeddings=getattr(args, 'include_atom_embeddings', False),
            mixed_precision=args.mixed_precision,
            ddp_enabled=False,  # Will be set by engine
            rank=0,  # Will be set by engine
            world_size=1,  # Will be set by engine
        )