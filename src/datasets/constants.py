# datasets/constants.py
"""
Constants used throughout the molecular datasets and training pipeline.
"""

from rdkit.Chem.rdchem import HybridizationType

# =============================================================================
# Atom Feature Constants
# =============================================================================
ATOM_TYPES = list(range(1, 119))  # Atomic numbers from 1 to 118
DEGREES = list(range(6))          # Degrees from 0 to 5
HYBRIDIZATIONS = [
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2
]

# =============================================================================
# Data Processing Constants
# =============================================================================
DEFAULT_SHUFFLE_BUFFER_SIZE = 1000
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_MOLECULE_ESTIMATE = 10000  # Default estimate when count unknown
DEFAULT_SAMPLE_SIZE = 1000  # For validation sampling

# =============================================================================
# Training Constants
# =============================================================================
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.1
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_LR_STEP_GAMMA = 0.1
DEFAULT_RANDOM_SEED = 42

# =============================================================================
# Inference Constants
# =============================================================================
DEFAULT_MC_DROPOUT_RATE = 0.1
DEFAULT_BATCH_FLUSH_THRESHOLD = 100  # Flush embeddings every N batches
DEFAULT_PROGRESS_LOG_INTERVAL = 1000  # Log progress every N molecules

# =============================================================================
# Distributed Training Constants
# =============================================================================
DDP_SYNC_DELAY = 0.1  # seconds to wait for synchronization