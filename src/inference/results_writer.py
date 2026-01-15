"""
Results writing functionality for inference pipeline.

This module handles all output file operations including:
- Setting up output files with proper headers
- Writing chunk results
- Combining DDP results from multiple ranks
- File cleanup operations
"""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from .config import InferenceConfig
from datasets.constants import DDP_SYNC_DELAY
from utils.logging import get_logger

logger = get_logger(__name__)


class ResultsWriter:
    """Handles writing inference results to output files."""

    def __init__(
        self,
        config: InferenceConfig,
        hyperparams: dict[str, Any],
        model: torch.nn.Module = None
    ) -> None:
        """
        Initialize ResultsWriter.

        Args:
            config: Inference configuration
            hyperparams: Model hyperparameters (from checkpoint)
            model: The model (needed for output dimension detection)
        """
        self.config = config
        self.hyperparams = hyperparams
        self.model = model

        # DDP state
        self.is_ddp = config.ddp_enabled
        self.rank = config.rank
        self.world_size = config.world_size

    def set_model(self, model: torch.nn.Module) -> None:
        """Set the model reference (needed for header generation)."""
        self.model = model

    def setup_output_file(self) -> str:
        """
        Setup output file path for this rank.

        Returns:
            Path to the output file for this rank
        """
        if self.is_ddp and self.world_size > 1:
            base, ext = os.path.splitext(self.config.output_path)
            output_file = f"{base}_rank{self.rank}{ext}"
        else:
            output_file = self.config.output_path

        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Write header
        header = self._generate_output_header()
        with open(output_file, 'w') as f:
            f.write(','.join(header) + '\n')

        return output_file

    def _generate_output_header(self) -> list[str]:
        """
        Generate output CSV header with proper column naming.

        Returns:
            List of column names for the output CSV
        """
        header = []

        # Always include SMILES
        header.append(self.config.smiles_column)

        # Add ID column - check HDF5 metadata for ID info
        try:
            import h5py
            with h5py.File(self.config.input_path, 'r') as f:
                if 'metadata' in f:
                    metadata = f['metadata']
                    # Check if HDF5 has ID information stored
                    if 'has_id_column' in metadata.attrs and metadata.attrs['has_id_column']:
                        id_col_name = metadata.attrs.get('id_column_name', 'id')
                        header.append(id_col_name)
                    else:
                        header.append('row_id')  # Auto-generated
                else:
                    header.append('row_id')  # Fallback
        except (OSError, KeyError, TypeError):
            header.append('row_id')  # Fallback on error

        # Get target column names from saved model
        target_column_name = 'target'
        multi_target_names = None

        # Load from hyperparams
        if self.hyperparams:
            target_column_name = self.hyperparams.get('target_column_name', 'target')
            multi_target_names = self.hyperparams.get('multi_target_column_names', None)

        # Determine number of output dimensions and column names
        if self.model is None:
            # Fallback: try to load from model path
            model_path = self.config.model_path
            if model_path and os.path.exists(model_path):
                try:
                    model_artifact = torch.load(model_path, map_location='cpu')
                    if "hyperparams" in model_artifact:
                        hyperparams = model_artifact['hyperparams']
                        target_column_name = hyperparams.get('target_column_name', 'target')
                        multi_target_names = hyperparams.get('multi_target_column_names', None)
                except (OSError, RuntimeError, KeyError):
                    pass  # Use defaults

        # Get loss function and output dimension from model
        loss_function = 'l1'
        output_dim = 1

        if self.model is not None:
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                loss_function = getattr(self.model.module, 'loss_function', 'l1')
                output_layer_size = self.model.module.output_layer.weight.shape[0]
            else:
                loss_function = getattr(self.model, 'loss_function', 'l1')
                output_layer_size = self.model.output_layer.weight.shape[0]

            # For evidential loss, calculate actual number of tasks
            if loss_function == 'evidential' and output_layer_size % 4 == 0:
                output_dim = output_layer_size // 4
            else:
                output_dim = output_layer_size

        # Add prediction columns with proper names
        if output_dim > 1:  # Multi-task
            if multi_target_names and len(multi_target_names) == output_dim:
                # Use original target column names
                for col_name in multi_target_names:
                    header.append(f"{col_name}_prediction")
                    if self.config.mc_samples > 0 or loss_function == 'evidential':
                        header.append(f"{col_name}_uncertainty")
            else:
                # Fallback to generic names
                for i in range(output_dim):
                    header.append(f"{target_column_name}_{i}_prediction")
                    if self.config.mc_samples > 0 or loss_function == 'evidential':
                        header.append(f"{target_column_name}_{i}_uncertainty")
        else:  # Single task
            header.append(f"{target_column_name}_prediction")
            if self.config.mc_samples > 0 or loss_function == 'evidential':
                header.append(f"{target_column_name}_uncertainty")

        return header

    def write_chunk_results(
        self,
        chunk_df: pd.DataFrame,
        valid_data: dict,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        output_file: str,
        chunk_start: int = 0
    ) -> None:
        """
        Write chunk results with ID preservation and proper column naming.

        Args:
            chunk_df: Original chunk DataFrame (may be None for HDF5)
            valid_data: Dictionary with 'smiles', 'indices', 'precomputed' keys
            predictions: Numpy array of predictions
            uncertainties: Numpy array of uncertainties
            output_file: Path to output file
            chunk_start: Starting index for this chunk (for row_id calculation)
        """
        if len(predictions) == 0:
            return

        # Check if evidential
        loss_function = 'l1'
        if self.model is not None:
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                loss_function = getattr(self.model.module, 'loss_function', 'l1')
            else:
                loss_function = getattr(self.model, 'loss_function', 'l1')

        has_uncertainties = (self.config.mc_samples > 0) or (loss_function == 'evidential')

        with open(output_file, 'a') as f:
            # Process each valid molecule
            for i, (smi, pred_idx) in enumerate(zip(valid_data['smiles'], range(len(predictions)))):
                line = [smi]

                # Add ID - use original index from HDF5
                original_idx = valid_data['indices'][i] + chunk_start
                line.append(str(original_idx))

                # Add predictions
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # Multi-task
                    for j in range(predictions.shape[1]):
                        line.append(str(predictions[pred_idx, j]))
                        if has_uncertainties:
                            line.append(str(uncertainties[pred_idx, j]))
                else:
                    # Single task
                    pred_val = predictions[pred_idx].item() if len(predictions.shape) > 1 else predictions[pred_idx]
                    line.append(str(pred_val))
                    if has_uncertainties:
                        unc_val = uncertainties[pred_idx].item() if len(uncertainties.shape) > 1 else uncertainties[pred_idx]
                        line.append(str(unc_val))

                f.write(','.join(line) + '\n')

            f.flush()

    def combine_ddp_results(self, rank_output_file: str) -> None:
        """
        Combine results from all DDP ranks with SMILES preservation.

        Proper file synchronization without sleep-based race conditions.

        Args:
            rank_output_file: Path to this rank's output file
        """
        if not self.is_ddp:
            return

        # Barrier after all ranks have written and synced files
        if dist.is_available() and dist.is_initialized():
            logger.info(f"Rank {self.rank}: Finished writing, synchronizing...")
            dist.barrier()

        # Only rank 0 combines files
        if self.rank != 0:
            logger.info(f"Rank {self.rank}: Exiting after barrier...")
            return  # Early return for non-zero ranks

        logger.info("Rank 0: Beginning file combination...")

        # Verify all rank files exist
        existing_files = []
        for rank in range(self.world_size):
            base, ext = os.path.splitext(self.config.output_path)
            rank_file = f"{base}_rank{rank}{ext}"

            if os.path.exists(rank_file):
                file_size = os.path.getsize(rank_file)
                logger.info(f"  Rank {rank}: {rank_file} ({file_size:,} bytes) - exists")
            else:
                logger.error(f"  Rank {rank}: MISSING {rank_file}")
            existing_files.append(rank_file) if os.path.exists(rank_file) else None

        # Re-collect existing files properly
        existing_files = []
        for rank in range(self.world_size):
            base, ext = os.path.splitext(self.config.output_path)
            rank_file = f"{base}_rank{rank}{ext}"
            if os.path.exists(rank_file):
                existing_files.append(rank_file)

        if len(existing_files) != self.world_size:
            raise RuntimeError(
                f"Missing rank files after barrier! Expected {self.world_size}, found {len(existing_files)}\n"
                f"This indicates a file system synchronization issue or rank failure."
            )

        # Combine files preserving SMILES and IDs
        try:
            # Read all rank files
            dfs = []
            for rank_file in existing_files:
                df_rank = pd.read_csv(rank_file)
                dfs.append(df_rank)
                logger.info(f"  Read {len(df_rank)} rows from {os.path.basename(rank_file)}")

            # Concatenate
            combined_df = pd.concat(dfs, ignore_index=True)

            # Sort by row_id if present to restore original order
            if 'row_id' in combined_df.columns:
                combined_df = combined_df.sort_values('row_id').reset_index(drop=True)

            # Save combined file
            combined_df.to_csv(self.config.output_path, index=False)

            output_size = os.path.getsize(self.config.output_path)

            logger.info("")
            logger.info("=" * 60)
            logger.info(f"SUCCESS: Combined predictions from {self.world_size} GPUs")
            logger.info("=" * 60)
            logger.info(f"  Total rows: {len(combined_df):,}")
            logger.info(f"  Output file: {self.config.output_path}")
            logger.info(f"  File size: {output_size:,} bytes ({output_size/1024/1024:.2f} MB)")
            logger.info(f"  Columns: {', '.join(combined_df.columns)}")
            logger.info("=" * 60)
            logger.info("")

            # Clean up rank files
            for rank_file in existing_files:
                try:
                    os.remove(rank_file)
                    logger.info(f"  Removed: {rank_file}")
                except Exception as e:
                    logger.warning(f"  Could not remove {rank_file}: {e}")

        except Exception as e:
            logger.error("")
            logger.error("=" * 60)
            logger.error("ERROR COMBINING FILES")
            logger.error("=" * 60)
            logger.error(f"{e}")
            logger.error("")
            logger.error("Rank files preserved for debugging:")
            for f in existing_files:
                logger.error(f"  - {f}")
            logger.error("=" * 60)
            logger.error("")
            raise

        logger.info("Rank 0: File combination complete, continuing...")

    def cleanup(self, embedding_manager: Any = None) -> None:
        """
        Properly clean up resources.

        Args:
            embedding_manager: Optional embedding manager to finalize
        """
        try:
            rank = self.rank
            logger.info(f"Rank {rank}: Starting cleanup...")

            # 1. Finalize any ongoing operations
            if embedding_manager:
                embedding_manager.finalize()

            # 2. Close file handles
            self._close_file_handles()

            # 3. Synchronize all ranks before GPU cleanup
            if dist.is_available() and dist.is_initialized():
                logger.info(f"Rank {rank}: Synchronizing before GPU cleanup...")
                try:
                    dist.barrier()
                except Exception as e:
                    logger.error(f"Rank {rank}: Barrier error: {e}")

            # 4. GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()

            # 5. CPU memory cleanup
            import gc
            gc.collect()

            # 6. Final barrier before process group cleanup
            if dist.is_available() and dist.is_initialized():
                logger.info(f"Rank {rank}: Final synchronization...")
                try:
                    dist.barrier()
                except Exception as e:
                    logger.error(f"Rank {rank}: Final barrier error: {e}")

                if rank == 0:
                    logger.info("All ranks synchronized, cleanup complete")

            logger.info(f"Rank {rank}: Cleanup successful")

        except Exception as e:
            logger.error(f"Rank {self.rank}: Cleanup error: {e}")

        finally:
            if dist.is_available() and dist.is_initialized():
                try:
                    time.sleep(DDP_SYNC_DELAY)
                except (InterruptedError, KeyboardInterrupt):
                    pass

    def _close_file_handles(self) -> None:
        """Close any open file handles."""
        try:
            # Close any HDF5 files
            import h5py
            # This will close any unclosed HDF5 files
            h5py.get_config().default_file_mode = 'r'

            # Close any CSV writers or other file handles
            # Add specific cleanup for your file handles here

        except Exception as e:
            logger.warning(f"Error closing file handles: {e}")

    def sync_file_to_disk(self, output_file: str) -> None:
        """
        Force OS to flush file buffers to disk.

        Args:
            output_file: Path to file to sync
        """
        if self.is_ddp:
            try:
                fd = os.open(output_file, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except OSError as e:
                logger.warning(f"Rank {self.rank}: fsync failed: {e}")
