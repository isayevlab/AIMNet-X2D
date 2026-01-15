# Phase 3: Architecture Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve architecture by splitting the InferencePipeline god class (1128 lines, 33 methods) into focused, testable components.

**Architecture:** Conservative incremental approach - extract two well-isolated components (ModelLoader, ResultsWriter) while keeping the orchestration in InferencePipeline. This reduces risk while improving modularity.

**Tech Stack:** Python 3.12, pytest for verification

---

## Risk Assessment

| Approach | Risk | Benefit |
|----------|------|---------|
| Full 6-class split | High - many integration points | Maximum modularity |
| Extract 2 components | Low - isolated functionality | Good modularity, minimal risk |
| No change | None | Technical debt remains |

**Decision:** Extract 2 components (ModelLoader, ResultsWriter) - best risk/benefit ratio.

---

## Summary

| Task | Description | Lines Moved | Risk |
|------|-------------|-------------|------|
| 3.1 | Extract ModelLoader | ~250 lines | Low |
| 3.2 | Extract ResultsWriter | ~200 lines | Low |
| 3.3 | Update module exports | ~10 lines | Minimal |

---

## Task 3.1: Extract ModelLoader Class

**Create:** `src/inference/model_loader.py`
**Modify:** `src/inference/pipeline.py`

### Step 1: Create model_loader.py

Create `/home/olexandr/AIMNet-X2D/src/inference/model_loader.py`:

```python
"""
Model loading and validation for inference.

Handles loading trained models, reconstructing preprocessing pipelines,
and validating HDF5/model compatibility.
"""

import torch
from typing import Any
from pathlib import Path

from models import GNN
from .config import InferenceConfig
from .preprocessing import PreprocessingReconstructor
from utils.logging import get_logger

import h5py

logger = get_logger(__name__)


# Import compatibility check helpers from pipeline (they're already module-level)
from .pipeline import (
    _check_hdf5_max_hops_compatibility,
    _check_hdf5_preprocessing_compatibility,
    _check_hdf5_task_type_compatibility,
    _check_hdf5_inference_data_compatibility,
    _format_compatibility_error,
)


class ModelLoader:
    """Handles model loading, validation, and preprocessing reconstruction."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model: GNN | None = None
        self.preprocessing_pipeline: Any | None = None
        self.hyperparams: dict[str, Any] | None = None

    def load(self, device: torch.device) -> tuple[GNN, Any | None]:
        """
        Load model and preprocessing pipeline.

        Returns:
            Tuple of (model, preprocessing_pipeline)
        """
        self._load_model_and_preprocessing(device)

        if self.config.input_hdf5:
            self._verify_hdf5_model_compatibility()

        return self.model, self.preprocessing_pipeline

    def _load_model_and_preprocessing(self, device: torch.device) -> None:
        """Load model and preprocessing pipeline from checkpoint."""
        checkpoint = torch.load(self.config.model_path, map_location=device, weights_only=False)

        self.hyperparams = checkpoint.get('hyperparameters', {})
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Reconstruct preprocessing pipeline
        self.preprocessing_pipeline = PreprocessingReconstructor.reconstruct(
            checkpoint, self.config
        )

        # Build and load model
        self.model = self._build_model_from_hyperparams(self.hyperparams, state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()

        self._validate_loaded_model(self.hyperparams)

        logger.info(f"Model loaded from {self.config.model_path}")

    def _build_model_from_hyperparams(
        self,
        hyperparams: dict[str, Any],
        state_dict: dict[str, Any]
    ) -> GNN:
        """Build GNN model from hyperparameters."""
        output_dim = self._get_output_dim_from_state_dict(state_dict, hyperparams)

        return GNN(
            hidden_dim=hyperparams.get('hidden_dim', 128),
            num_shells=hyperparams.get('num_shells', 4),
            output_dim=output_dim,
            num_atom_types=hyperparams.get('num_atom_types', 100),
            num_degrees=hyperparams.get('num_degrees', 11),
            num_hydrogen_counts=hyperparams.get('num_hydrogen_counts', 5),
            num_hybridizations=hyperparams.get('num_hybridizations', 8),
            use_partial_charges=hyperparams.get('use_partial_charges', False),
            use_stereochemistry=hyperparams.get('use_stereochemistry', False),
            pooling=hyperparams.get('pooling', 'attention'),
            ffn_hidden_dim=hyperparams.get('ffn_hidden_dim', None),
            ffn_num_layers=hyperparams.get('ffn_num_layers', 2),
            ffn_dropout=hyperparams.get('ffn_dropout', 0.0),
            activation=hyperparams.get('activation', 'silu'),
            num_attention_heads=hyperparams.get('num_attention_heads', 4),
            attention_temperature=hyperparams.get('attention_temperature', 1.0),
            output_type=hyperparams.get('output_type', 'single'),
        )

    def _get_output_dim_from_state_dict(
        self,
        state_dict: dict[str, Any],
        hyperparams: dict[str, Any]
    ) -> int:
        """Determine output dimension from state dict or hyperparameters."""
        for key in ['output_layer.weight', 'output_layer.0.weight']:
            if key in state_dict:
                return state_dict[key].shape[0]
        return hyperparams.get('output_dim', 1)

    def _validate_loaded_model(self, hyperparams: dict[str, Any]) -> None:
        """Validate the loaded model configuration."""
        task_type = hyperparams.get('task_type', 'regression')
        if task_type == 'multitask':
            num_tasks = hyperparams.get('num_output_tasks', 1)
            logger.info(f"Multitask model with {num_tasks} tasks")

    def _verify_hdf5_model_compatibility(self) -> None:
        """Verify HDF5 file is compatible with the loaded model."""
        if not self.config.input_hdf5:
            return

        with h5py.File(self.config.input_hdf5, 'r') as f:
            if 'compatibility' not in f.attrs:
                logger.warning("HDF5 missing compatibility metadata - skipping validation")
                return

            compat = eval(f.attrs['compatibility'])
            errors = self._check_compatibility(compat)

            if errors:
                error_msg = _format_compatibility_error(
                    errors, self.config.model_path, self.config.input_hdf5
                )
                raise ValueError(error_msg)

            logger.info("HDF5 compatibility verified")

    def _check_compatibility(self, compat: dict[str, Any]) -> list[str]:
        """Run all compatibility checks and return list of errors."""
        errors = []

        # Get model parameters
        model_num_shells = self.hyperparams.get('num_shells', 4)
        model_task_type = self.hyperparams.get('task_type', 'regression')

        # Check max hops
        if err := _check_hdf5_max_hops_compatibility(
            compat.get('max_hops', -1), model_num_shells
        ):
            errors.append(err)

        # Check preprocessing
        if err := _check_hdf5_preprocessing_compatibility(
            compat.get('preprocessing_applied', False),
            self.preprocessing_pipeline is not None
        ):
            errors.append(err)

        # Check task type
        if err := _check_hdf5_task_type_compatibility(
            compat.get('task_type', 'unknown'), model_task_type
        ):
            errors.append(err)

        # Check inference data
        if err := _check_hdf5_inference_data_compatibility(
            compat.get('preprocessing_applied', False)
        ):
            errors.append(err)

        return errors
```

### Step 2: Update pipeline.py to use ModelLoader

In `/home/olexandr/AIMNet-X2D/src/inference/pipeline.py`:

1. Add import: `from .model_loader import ModelLoader`
2. Update `setup()` method to use ModelLoader
3. Remove the extracted methods from InferencePipeline

### Step 3: Run tests and commit

```bash
python -m pytest tests/ -q --tb=short
git add src/inference/model_loader.py src/inference/pipeline.py
git commit -m "Extract ModelLoader class from InferencePipeline"
```

---

## Task 3.2: Extract ResultsWriter Class

**Create:** `src/inference/results_writer.py`
**Modify:** `src/inference/pipeline.py`

### Step 1: Create results_writer.py

Create `/home/olexandr/AIMNet-X2D/src/inference/results_writer.py`:

```python
"""
Results writing and DDP coordination for inference.

Handles output file setup, writing chunk results, and combining
results from multiple DDP ranks.
"""

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .config import InferenceConfig
from datasets.constants import DDP_SYNC_DELAY
from utils.distributed import safe_get_rank, is_main_process
from utils.logging import get_logger

logger = get_logger(__name__)


class ResultsWriter:
    """Handles writing inference results and DDP coordination."""

    def __init__(self, config: InferenceConfig, hyperparams: dict[str, Any]):
        self.config = config
        self.hyperparams = hyperparams
        self._output_file_handle = None

    def setup_output_file(self) -> str:
        """Setup output file and return the path."""
        if self.config.ddp_enabled:
            base, ext = os.path.splitext(self.config.output_path)
            output_path = f"{base}_rank{self.config.rank}{ext}"
        else:
            output_path = self.config.output_path

        # Write header
        header = self.generate_output_header()
        with open(output_path, 'w') as f:
            f.write(','.join(header) + '\n')

        return output_path

    def generate_output_header(self) -> list[str]:
        """Generate CSV header based on model configuration."""
        header = ['smiles']

        task_type = self.hyperparams.get('task_type', 'regression')
        if task_type == 'multitask':
            num_tasks = self.hyperparams.get('num_output_tasks', 1)
            for i in range(num_tasks):
                header.append(f'prediction_{i}')
                if self.config.mc_samples > 0:
                    header.append(f'uncertainty_{i}')
        else:
            header.append('prediction')
            if self.config.mc_samples > 0:
                header.append('uncertainty')

        return header

    def write_chunk_results(
        self,
        smiles_list: list[str],
        predictions: np.ndarray,
        uncertainties: np.ndarray | None,
        output_file: str
    ) -> None:
        """Write a chunk of results to the output file."""
        results = {'smiles': smiles_list}

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        for i in range(predictions.shape[1]):
            results[f'prediction_{i}' if predictions.shape[1] > 1 else 'prediction'] = predictions[:, i]
            if uncertainties is not None:
                unc_col = f'uncertainty_{i}' if predictions.shape[1] > 1 else 'uncertainty'
                results[unc_col] = uncertainties[:, i] if uncertainties.ndim > 1 else uncertainties

        df = pd.DataFrame(results)
        df.to_csv(output_file, mode='a', header=False, index=False)

    def combine_ddp_results(self, rank_output_file: str) -> None:
        """Combine results from all DDP ranks into final output."""
        if not self.config.ddp_enabled:
            return

        # Synchronize ranks
        dist.barrier()

        if not is_main_process():
            return

        time.sleep(DDP_SYNC_DELAY)

        # Collect rank files
        base, ext = os.path.splitext(self.config.output_path)
        rank_files = []
        for rank in range(self.config.world_size):
            rank_file = f"{base}_rank{rank}{ext}"
            if os.path.exists(rank_file):
                rank_files.append(rank_file)

        if not rank_files:
            logger.error("No rank files found!")
            return

        # Combine files
        combined_df = pd.concat([pd.read_csv(f) for f in rank_files], ignore_index=True)
        combined_df.to_csv(self.config.output_path, index=False)

        # Cleanup rank files
        for f in rank_files:
            try:
                os.remove(f)
            except Exception as e:
                logger.warning(f"Could not remove {f}: {e}")

        logger.info(f"Combined {len(rank_files)} rank files -> {self.config.output_path}")

    def cleanup(self) -> None:
        """Cleanup file handles and temporary files."""
        if self._output_file_handle:
            self._output_file_handle.close()
            self._output_file_handle = None
```

### Step 2: Update pipeline.py to use ResultsWriter

1. Add import: `from .results_writer import ResultsWriter`
2. Create ResultsWriter in `setup()` method
3. Use ResultsWriter methods instead of inline implementations
4. Remove the extracted methods from InferencePipeline

### Step 3: Run tests and commit

```bash
python -m pytest tests/ -q --tb=short
git add src/inference/results_writer.py src/inference/pipeline.py
git commit -m "Extract ResultsWriter class from InferencePipeline"
```

---

## Task 3.3: Update Module Exports

**Modify:** `src/inference/__init__.py`

### Step 1: Update exports

```python
from .config import InferenceConfig
from .pipeline import InferencePipeline
from .model_loader import ModelLoader
from .results_writer import ResultsWriter
from .preprocessing import PreprocessingReconstructor
from .uncertainty import MCDropoutPredictor, DeterministicPredictor
from .embeddings import EmbeddingManager

__all__ = [
    'InferenceConfig',
    'InferencePipeline',
    'ModelLoader',
    'ResultsWriter',
    'PreprocessingReconstructor',
    'MCDropoutPredictor',
    'DeterministicPredictor',
    'EmbeddingManager',
]
```

### Step 2: Run tests and commit

```bash
python -m pytest tests/ -q --tb=short
git add src/inference/__init__.py
git commit -m "Export ModelLoader and ResultsWriter from inference module"
```

---

## Final Verification

```bash
# Run full test suite
python -m pytest tests/ -v --tb=short

# Check InferencePipeline size reduction
wc -l src/inference/pipeline.py

# Verify new files
wc -l src/inference/model_loader.py src/inference/results_writer.py
```

**Expected results:**
- pipeline.py: ~700 lines (down from 1128)
- model_loader.py: ~180 lines
- results_writer.py: ~120 lines
- All tests passing

---

## Estimated Scope

| Task | Files | Complexity | Est. Time |
|------|-------|------------|-----------|
| 3.1 ModelLoader | 2 files | Medium | 30 min |
| 3.2 ResultsWriter | 2 files | Medium | 25 min |
| 3.3 Exports | 1 file | Low | 5 min |
| **Total** | **5 files** | | **~1 hour** |

---

## Benefits

1. **Testability**: ModelLoader and ResultsWriter can be unit tested independently
2. **Reusability**: Components can be used in other inference scenarios
3. **Maintainability**: Smaller, focused classes are easier to understand
4. **Separation of Concerns**: Loading, processing, and writing are now separate
