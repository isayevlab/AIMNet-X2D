"""
Main inference pipeline orchestration.
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Any
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

from .config import InferenceConfig
from .model_loader import ModelLoader
from .results_writer import ResultsWriter
from .uncertainty import MCDropoutPredictor, DeterministicPredictor, UncertaintyEstimator
from .embeddings import EmbeddingManager
from datasets import _worker_process_smiles, MolecularBatch
from datasets.constants import DEFAULT_MOLECULE_ESTIMATE
from torch_geometric.data import Data
from utils.distributed import is_main_process
from utils.logging import get_logger

import h5py

logger = get_logger(__name__)


def _check_hdf5_max_hops_compatibility(
    hdf5_max_hops: int,
    model_num_shells: int
) -> str | None:
    """Check max_hops compatibility. Returns error message or None if compatible."""
    if hdf5_max_hops != model_num_shells:
        return (
            f"CRITICAL: Max hops mismatch!\n"
            f"   HDF5 file: {hdf5_max_hops} hops\n"
            f"   Model expects: {model_num_shells} hops\n"
            f"   \n"
            f"   This is a FATAL incompatibility - the molecular features\n"
            f"   in the HDF5 file do not match what the model was trained on.\n"
            f"   \n"
            f"   The HDF5 file contains BFS features computed to depth {hdf5_max_hops},\n"
            f"   but the model expects features computed to depth {model_num_shells}.\n"
            f"   These are fundamentally different graph representations.\n"
            f"   \n"
            f"   You MUST recreate the HDF5 file with --num_shells={model_num_shells}"
        )
    return None


def _check_hdf5_preprocessing_compatibility(
    hdf5_has_preprocessing: bool,
    model_has_preprocessing: bool
) -> str | None:
    """Check preprocessing compatibility. Returns error message or None if compatible."""
    if hdf5_has_preprocessing and not model_has_preprocessing:
        return (
            "HDF5 file has preprocessing applied but model has no preprocessing pipeline. "
            "This will produce incorrect results. Regenerate HDF5 without preprocessing."
        )
    return None


def _check_hdf5_task_type_compatibility(
    hdf5_task_type: str,
    model_task_type: str
) -> str | None:
    """Check task type compatibility. Returns error message or None if compatible."""
    if hdf5_task_type != model_task_type:
        return (
            f"Task type mismatch:\n"
            f"   HDF5: {hdf5_task_type}\n"
            f"   Model: {model_task_type}"
        )
    return None


def _check_hdf5_inference_data_compatibility(
    preprocessing_applied: bool
) -> str | None:
    """Check if HDF5 contains raw data suitable for inference. Returns error message or None if compatible."""
    if preprocessing_applied:
        return (
            "HDF5 contains PREPROCESSED data\n"
            "   Inference requires RAW data\n"
            "   Use create_inference_hdf5.py to create proper HDF5"
        )
    return None


def _format_compatibility_error(
    errors: list[str],
    model_path: str,
    input_path: str
) -> str:
    """Format compatibility errors into a detailed error message."""
    error_msg = "\n" + "=" * 60 + "\n"
    error_msg += "HDF5 FILE IS INCOMPATIBLE WITH MODEL\n"
    error_msg += "=" * 60 + "\n\n"
    for i, e in enumerate(errors, 1):
        error_msg += f"{i}. {e}\n\n"
    error_msg += "=" * 60 + "\n"
    error_msg += "SOLUTION:\n"
    error_msg += "=" * 60 + "\n"
    error_msg += f"Recreate the HDF5 file with matching parameters:\n\n"
    error_msg += f"  python create_inference_hdf5.py \\\n"
    error_msg += f"    --model_path {model_path} \\\n"
    error_msg += f"    --input_csv YOUR_DATA.csv \\\n"
    error_msg += f"    --output_hdf5 {input_path} \\\n"
    error_msg += f"    --smiles_column smiles\n"
    error_msg += "=" * 60 + "\n"
    return error_msg


class InferencePipeline:
    """Main inference pipeline that orchestrates the entire process."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.model = None
        self.preprocessing_pipeline = None
        self.hyperparams = None
        self.device = None
        self.embedding_manager = None
        self.uncertainty_estimator = None
        self.deterministic_predictor = None
        self.results_writer = None

        # DDP state
        self.is_ddp = config.ddp_enabled
        self.rank = config.rank
        self.world_size = config.world_size

        # Statistics
        self.total_processed = 0
        self.valid_count = 0
        self.invalid_count = 0

        # Track processed SMILES for correspondence
        self.processed_smiles = []
        self.processed_indices = []

    def setup(self, device: torch.device) -> None:
        """Setup the inference pipeline components."""
        self.device = device

        # Load model and preprocessing using ModelLoader
        loader = ModelLoader(self.config)
        self.model, self.preprocessing_pipeline, self.hyperparams = loader.load(device)

        # Verify HDF5 compatibility if applicable
        loader.verify_hdf5_model_compatibility(self.model)

        # Setup results writer
        self.results_writer = ResultsWriter(self.config, self.hyperparams, self.model)

        # Setup prediction methods
        if self.config.mc_samples > 0:
            from .uncertainty import MCDropoutPredictor
            self.uncertainty_estimator = MCDropoutPredictor(
                model=self.model,
                device=device,
                num_samples=self.config.mc_samples
            )
        else:
            from .uncertainty import DeterministicPredictor
            self.deterministic_predictor = DeterministicPredictor(
                model=self.model,
                device=device
            )

        # Setup embedding extraction if requested
        if self.config.save_embeddings:
            total_molecules = self._estimate_total_molecules()
            # Adjust for DDP work distribution
            if self.is_ddp:
                _, _, rank_molecules = self._calculate_ddp_work_distribution(total_molecules)
                total_molecules = rank_molecules

            from .embeddings import EmbeddingManager
            self.embedding_manager = EmbeddingManager(
                model=self.model,
                device=device,
                output_path=self.config.embeddings_path,
                expected_total_molecules=total_molecules,
                include_atom_embeddings=self.config.include_atom_embeddings,
                rank=self.rank,
                world_size=self.world_size
            )

    def _estimate_total_molecules(self) -> int:
        """Estimate total number of molecules for embedding pre-allocation."""
        try:
            if self.config.input_path.endswith('.csv'):
                # Quick line count for CSV
                with open(self.config.input_path, 'r') as f:
                    return sum(1 for _ in f) - 1  # Subtract header
            else:
                # Default estimate for other formats
                return DEFAULT_MOLECULE_ESTIMATE
        except (OSError, IOError):
            return DEFAULT_MOLECULE_ESTIMATE

    def run_streaming_inference(self) -> None:
        """Run streaming inference on CSV input."""
        if not self.config.input_path.endswith('.csv'):
            raise ValueError("Streaming inference requires CSV input")

        try:
            # Count total molecules
            total_molecules = self._estimate_total_molecules()

            # Calculate work distribution for DDP
            if self.is_ddp:
                start_line, end_line, rank_molecules = self._calculate_ddp_work_distribution(total_molecules)
            else:
                start_line, end_line = 1, None  # Skip header
                rank_molecules = total_molecules

            if is_main_process():
                logger.info(f"Starting streaming inference for {rank_molecules} molecules")

            # Setup output file using results writer
            output_file = self.results_writer.setup_output_file()

            # Process in chunks
            self._process_csv_chunks(start_line, end_line, output_file)

            # Combine DDP results if needed
            if self.is_ddp:
                self.results_writer.combine_ddp_results(output_file)

            # Finalize embeddings if needed
            if self.embedding_manager:
                self.embedding_manager.finalize()

            if is_main_process():
                logger.info("Streaming inference completed")
                logger.info(f"Processed: {self.total_processed}, Valid: {self.valid_count}, Invalid: {self.invalid_count}")

        except Exception as e:
            logger.error(f"Error combining files: {e}")


    def _calculate_ddp_work_distribution(self, total_molecules: int) -> tuple[int, int, int]:
        """Calculate work distribution for DDP."""
        if not self.is_ddp or self.world_size <= 1:
            return 1, None, total_molecules  # Skip header, process all

        # Distribute molecules across ranks
        molecules_per_rank = total_molecules // self.world_size
        remainder = total_molecules % self.world_size

        # Calculate start and end for this rank
        if self.rank < remainder:
            # First 'remainder' ranks get one extra molecule
            rank_molecules = molecules_per_rank + 1
            start_molecule = self.rank * rank_molecules
        else:
            # Remaining ranks get standard amount
            rank_molecules = molecules_per_rank
            start_molecule = remainder * (molecules_per_rank + 1) + (self.rank - remainder) * molecules_per_rank

        end_molecule = start_molecule + rank_molecules

        # Convert to line numbers (add 1 for header)
        start_line = start_molecule + 1  # +1 for header
        end_line = end_molecule + 1      # +1 for header

        logger.info(f"Rank {self.rank}: molecules {start_molecule}-{end_molecule-1} "
                    f"(lines {start_line}-{end_line-1}): {rank_molecules} molecules")

        return start_line, end_line, rank_molecules

    def _process_csv_chunks(self, start_line: int, end_line: int | None, output_file: str) -> None:
        """Process CSV file in chunks with position tracking."""
        # Setup chunk reading
        skiprows = list(range(1, start_line)) if start_line > 1 else None
        nrows = None if end_line is None else (end_line - start_line)

        chunk_iterator = pd.read_csv(
            self.config.input_path,
            skiprows=skiprows,
            nrows=nrows,
            chunksize=self.config.chunk_size
        )

        current_position = start_line - 1 if start_line > 1 else 0

        for chunk_idx, chunk_df in enumerate(chunk_iterator):
            self._current_chunk_start = current_position
            self._process_single_chunk(chunk_df, chunk_idx, output_file)
            current_position += len(chunk_df)

        # Sync file to disk before DDP barrier
        self.results_writer.sync_file_to_disk(output_file)


    def _process_single_chunk(self, chunk_df: pd.DataFrame, chunk_idx: int, output_file: str) -> None:
        """Process a single chunk of data."""
        smiles_list = chunk_df[self.config.smiles_column].tolist()

        if not smiles_list:
            return

        start_time = time.time()

        # Parallel feature computation
        valid_data = self._compute_features_parallel(smiles_list, chunk_idx)

        if not valid_data['smiles']:
            if is_main_process():
                logger.info(f"No valid SMILES in chunk {chunk_idx+1}")
            return

        # Create data loader
        data_loader = self._create_chunk_dataloader(valid_data)

        # Define embedding callback for proper correspondence
        def embedding_callback(batch):
            if self.embedding_manager:
                self.embedding_manager.process_batch(batch)

        # Make predictions with embedding extraction
        if self.config.mc_samples > 0:
            predictions, uncertainties = self.uncertainty_estimator.predict_with_uncertainty(
                data_loader,
                preprocessing_pipeline=self.preprocessing_pipeline,
                show_progress=False,
                embedding_callback=embedding_callback
            )
        else:
            # Check if this is an evidential model
            loss_function = getattr(self.model, 'loss_function', 'l1')
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                loss_function = getattr(self.model.module, 'loss_function', 'l1')

            if loss_function == 'evidential':
                # Use evidential uncertainty estimation
                predictions, uncertainties = self._predict_evidential_with_uncertainty(
                    data_loader, embedding_callback
                )
            else:
                predictions = self.deterministic_predictor.predict(
                    data_loader,
                    preprocessing_pipeline=self.preprocessing_pipeline,
                    embedding_callback=embedding_callback
                )
                uncertainties = np.zeros_like(predictions)

        # Write results using results writer
        self.results_writer.write_chunk_results(
            chunk_df, valid_data, predictions, uncertainties, output_file,
            chunk_start=getattr(self, '_current_chunk_start', 0)
        )

        # Update statistics
        self.total_processed += len(valid_data['smiles'])
        processing_time = time.time() - start_time

        if self.rank == 0:  # Only main rank prints progress
            logger.info(f"Chunk {chunk_idx+1}: {len(valid_data['smiles'])} molecules in {processing_time:.2f}s")

    def _predict_evidential_with_uncertainty(self, data_loader, embedding_callback) -> tuple[np.ndarray, np.ndarray]:
        """Make predictions with evidential uncertainty estimation."""
        self.model.eval()
        all_preds = []
        all_uncertainties = []
        all_smiles = []  # FIXED: Collect SMILES for SAE inverse transform

        with torch.no_grad():
            for batch in data_loader:
                if batch is None:
                    continue

                # FIXED: Collect SMILES from each batch
                all_smiles.extend(batch.smiles_list)

                # Extract embeddings if callback provided
                if embedding_callback is not None:
                    embedding_callback(batch)

                # Prepare batch
                batch_multi_hop_edges = batch.multi_hop_edge_indices.to(self.device)
                batch_indices = batch.batch_indices.to(self.device)
                batch_atom_features = {k: v.to(self.device) for k, v in batch.atom_features_map.items()}
                total_charges = batch.total_charges.to(self.device)
                tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(self.device)
                cis_indices = batch.final_cis_tensor.to(self.device)
                trans_indices = batch.final_trans_tensor.to(self.device)

                # Forward pass
                outputs, _, _ = self.model(
                    batch_atom_features,
                    batch_multi_hop_edges,
                    batch_indices,
                    total_charges,
                    tetrahedral_indices,
                    cis_indices,
                    trans_indices
                )

                # Process evidential outputs
                predictions, uncertainties = self._process_evidential_outputs_with_uncertainty(outputs)

                all_preds.append(predictions.cpu().numpy())
                all_uncertainties.append(uncertainties.cpu().numpy())

        if all_preds:
            combined_preds = np.concatenate(all_preds, axis=0)
            combined_uncertainties = np.concatenate(all_uncertainties, axis=0)

            # FIXED: Apply inverse preprocessing with SMILES
            if self.preprocessing_pipeline:
                combined_preds = self.preprocessing_pipeline.inverse_transform(
                    smiles_list=all_smiles,  # FIXED: Pass the collected SMILES
                    transformed_targets=combined_preds
                )

            return combined_preds, combined_uncertainties

        return np.array([]), np.array([])

    def _process_evidential_outputs_with_uncertainty(self, outputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Process evidential outputs to get predictions and uncertainties."""
        batch_size = outputs.shape[0]
        if outputs.shape[1] % 4 == 0:
            num_tasks = outputs.shape[1] // 4
            evidential_params = outputs.view(batch_size, num_tasks, 4)

            # Extract evidential parameters
            gamma = evidential_params[:, :, 0]  # predicted mean
            nu = F.softplus(evidential_params[:, :, 1]) + 1.0  # degrees of freedom
            alpha = F.softplus(evidential_params[:, :, 2]) + 1.0  # concentration
            beta = F.softplus(evidential_params[:, :, 3])  # rate parameter

            # Calculate uncertainty (epistemic + aleatoric)
            aleatoric = beta / torch.clamp(alpha - 1, min=1e-6)
            epistemic = beta / (nu * torch.clamp(alpha - 1, min=1e-6))
            total_uncertainty = aleatoric + epistemic

            return gamma, total_uncertainty
        else:
            # Fallback for non-evidential outputs
            return outputs, torch.zeros_like(outputs)

    def _compute_features_parallel(self, smiles_list: list, chunk_idx: int) -> dict:
        """Compute molecular features in parallel."""
        max_hops = self.config.max_hops or 3  # Default to 3 if not set
        process_inputs = [(idx, smi, max_hops) for idx, smi in enumerate(smiles_list)]

        with Pool(processes=self.config.num_workers) as pool:
            results = list(pool.imap(
                _worker_process_smiles,
                process_inputs,
                chunksize=max(1, len(process_inputs) // (self.config.num_workers * 4))
            ))

        # Collect valid results
        valid_data = {
            'smiles': [],
            'indices': [],
            'precomputed': []
        }

        for idx, precomp in results:
            if precomp is not None:
                valid_data['smiles'].append(smiles_list[idx])
                valid_data['indices'].append(idx)
                valid_data['precomputed'].append(precomp)
                self.valid_count += 1
            else:
                self.invalid_count += 1

        return valid_data

    def _create_chunk_dataloader(self, valid_data: dict) -> torch.utils.data.DataLoader:
        """Create DataLoader for a chunk of valid data."""
        data_objects = []

        for smi, precomp in zip(valid_data['smiles'], valid_data['precomputed']):
            data_obj = self._create_data_object(smi, precomp)
            if data_obj is not None:
                data_objects.append(data_obj)

        return torch.utils.data.DataLoader(
            data_objects,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=MolecularBatch.from_data_list,
            num_workers=0  # No multiprocessing in DataLoader for inference
        )

    def _create_data_object(self, smiles: str, precomp: dict) -> Data | None:
        """Create a PyG Data object from precomputed features."""
        try:
            num_atoms = precomp['atom_features']['atom_type'].shape[0]
            x_dummy = torch.ones((num_atoms, 1), dtype=torch.float)

            data_obj = Data()
            data_obj.x = x_dummy
            data_obj.smiles = smiles
            data_obj.target = torch.tensor([0.0], dtype=torch.float)  # Dummy target

            # Multi-hop edges
            data_obj.multi_hop_edges = [torch.from_numpy(e).long() for e in precomp["multi_hop_edges"]]

            # Atom features
            atom_feats_map = {}
            for k, arr in precomp["atom_features"].items():
                atom_feats_map[k] = torch.from_numpy(arr).long()
            data_obj.atom_features_map = atom_feats_map

            # Stereochemistry
            data_obj.chiral_tensors = [torch.from_numpy(x).long() for x in precomp["chiral_tensors"]]
            data_obj.cis_bonds_tensors = [torch.from_numpy(x).long() for x in precomp["cis_bonds_tensors"]]
            data_obj.trans_bonds_tensors = [torch.from_numpy(x).long() for x in precomp["trans_bonds_tensors"]]

            # Additional features
            data_obj.total_charge = torch.tensor([precomp["total_charge"]], dtype=torch.float)
            data_obj.atomic_numbers = torch.from_numpy(precomp["atomic_numbers"]).long()

            return data_obj
        except Exception as e:
            logger.error(f"Error creating data object for {smiles[:30]}...: {str(e)}")
            return None

    def cleanup_and_exit(self) -> None:
        """
        Properly clean up DDP resources without hanging.

        Delegates to ResultsWriter for cleanup operations, with additional
        model-specific cleanup for GPU memory.
        """
        # Move model to CPU before cleanup to free GPU memory
        if torch.cuda.is_available() and hasattr(self, 'model') and self.model is not None:
            self.model.cpu()

        # Delegate to results writer for main cleanup
        if self.results_writer:
            self.results_writer.cleanup(embedding_manager=self.embedding_manager)
        else:
            # Fallback cleanup if results_writer not initialized
            if self.embedding_manager:
                self.embedding_manager.finalize()
