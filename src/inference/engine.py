"""
High-level inference engine and legacy compatibility.
"""

import torch
import torch.distributed as dist
from typing import Optional

from .config import InferenceConfig
from .pipeline import InferencePipeline
from datasets import create_iterable_pyg_dataloader
from datasets.constants import DEFAULT_SHUFFLE_BUFFER_SIZE
from training import predict_gnn
from utils.distributed import safe_get_rank, is_main_process

import tqdm

import numpy as np

from typing import List




class InferenceEngine:
    """High-level inference engine that handles different input types."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.pipeline = InferencePipeline(config)
    
    def run(self, device: torch.device):
        """Run inference based on input type."""
        # Setup pipeline
        self.pipeline.setup(device)
        
        # Determine input type and run appropriate inference
        if self.config.input_path.endswith('.csv'):
            self._run_csv_inference()
        elif self.config.input_path.endswith('.h5') or self.config.input_path.endswith('.hdf5'):
            self._run_hdf5_inference(device)
        else:
            raise ValueError(f"Unsupported input format: {self.config.input_path}")
    
    def _run_csv_inference(self):
        """Run streaming inference on CSV input."""
        if is_main_process():
            print(f"[Engine] Running streaming CSV inference")
        
        self.pipeline.run_streaming_inference()

    def _run_hdf5_inference(self, device: torch.device):
        """Run inference on HDF5 input with SMILES and ID tracking."""
        if is_main_process():
            print(f"[Engine] Running HDF5 inference")
        
        # Get total molecule count
        total_molecules = None
        try:
            import h5py
            with h5py.File(self.config.input_path, 'r') as f:
                if 'metadata' in f:
                    total_molecules = f['metadata'].attrs.get('num_samples', None)
                if total_molecules is None and 'data' in f:
                    total_molecules = f['data'].shape[0]
        except (OSError, KeyError):
            pass  # File may not exist or have expected structure
        
        if is_main_process() and total_molecules:
            print(f"[Engine] Processing {total_molecules:,} molecules")
        
        # Create data loader
        inference_loader = create_iterable_pyg_dataloader(
            hdf5_path=self.config.input_path,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER_SIZE,
            ddp_enabled=self.config.ddp_enabled,
            rank=self.config.rank,
            world_size=self.config.world_size
        )
        
        # Calculate batches for progress bar
        progress_batches = None
        if total_molecules and self.config.batch_size:
            progress_batches = (total_molecules + self.config.batch_size - 1) // self.config.batch_size
        
        # FIXED: Collect predictions and SMILES from this rank
        all_smiles = []
        all_predictions = []
        
        self.pipeline.model.eval()
        
        with torch.no_grad():
            for batch in tqdm.tqdm(inference_loader, total=progress_batches, desc="HDF5 Inference"):
                if batch is None:
                    continue
                
                # Collect SMILES from each batch
                all_smiles.extend(batch.smiles_list)
                
                # Prepare batch data
                batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
                batch_indices = batch.batch_indices.to(device)
                batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
                total_charges = batch.total_charges.to(device)
                tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
                cis_indices = batch.final_cis_tensor.to(device)
                trans_indices = batch.final_trans_tensor.to(device)
                
                # Forward pass
                outputs, _, _ = self.pipeline.model(
                    batch_atom_features,
                    batch_multi_hop_edges,
                    batch_indices,
                    total_charges,
                    tetrahedral_indices,
                    cis_indices,
                    trans_indices
                )
                
                # Process evidential outputs if needed
                loss_function = getattr(self.pipeline.model, 'loss_function', 'l1')
                if isinstance(self.pipeline.model, torch.nn.parallel.DistributedDataParallel):
                    loss_function = getattr(self.pipeline.model.module, 'loss_function', 'l1')
                
                if loss_function == 'evidential':
                    # Extract mean predictions from evidential outputs
                    batch_size = outputs.shape[0]
                    if outputs.shape[1] % 4 == 0:
                        num_tasks = outputs.shape[1] // 4
                        evidential_params = outputs.view(batch_size, num_tasks, 4)
                        predictions = evidential_params[:, :, 0]  # gamma (mean)
                    else:
                        predictions = outputs
                else:
                    predictions = outputs
                
                all_predictions.append(predictions.detach().cpu().numpy())
        
        # Combine predictions from this rank
        if all_predictions:
            predictions = np.concatenate(all_predictions, axis=0)
            
            # Ensure proper shape
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            
            # Apply inverse preprocessing with SMILES from this rank
            if self.pipeline.preprocessing_pipeline:
                predictions = self.pipeline.preprocessing_pipeline.inverse_transform(
                    smiles_list=all_smiles,
                    transformed_targets=predictions
                )
        else:
            predictions = np.array([])
            all_smiles = []
        
        # FIXED: Gather predictions AND SMILES across all ranks before saving
        if self.config.ddp_enabled and dist.is_initialized():
            # Gather predictions to rank 0
            gathered_predictions = self._gather_predictions_ddp(predictions, device)
            
            # FIXED: Gather SMILES to rank 0
            gathered_smiles = self._gather_smiles_ddp(all_smiles, device)
            
            # Only rank 0 saves the combined results
            if self.config.rank == 0:
                self._save_hdf5_predictions_with_smiles(gathered_predictions, gathered_smiles)
        else:
            # Single GPU: save directly
            if is_main_process():
                self._save_hdf5_predictions_with_smiles(predictions, all_smiles)
        
        # Extract embeddings if requested
        if self.config.save_embeddings and is_main_process():
            self._extract_hdf5_embeddings(inference_loader, device)

    def _gather_smiles_ddp(self, local_smiles: List[str], device: torch.device) -> List[str]:
        """
        Gather SMILES from all ranks to rank 0.
        
        CRITICAL: This ensures SMILES correspond to predictions after gathering.
        
        Args:
            local_smiles: SMILES list from this rank
            device: Device for tensor operations
            
        Returns:
            On rank 0: Combined SMILES list from all ranks
            On other ranks: Empty list
        """
        import pickle
        
        if not dist.is_initialized():
            return local_smiles
        
        rank = self.config.rank
        world_size = self.config.world_size
        
        if is_main_process():
            print(f"[Engine] Gathering SMILES from {world_size} ranks...")
        
        # Serialize local SMILES to bytes
        local_bytes = pickle.dumps(local_smiles)
        local_size = torch.tensor([len(local_bytes)], dtype=torch.long, device=device)
        
        # Gather sizes from all ranks
        sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes_list, local_size)
        
        # Pad to max size for gathering
        max_size = max(s.item() for s in sizes_list)
        if len(local_bytes) < max_size:
            local_bytes += b'\x00' * (max_size - len(local_bytes))
        
        # Convert to tensor
        local_tensor = torch.ByteTensor(list(local_bytes)).to(device)
        
        # Gather tensors from all ranks
        gathered_tensors = [
            torch.zeros(max_size, dtype=torch.uint8, device=device) 
            for _ in range(world_size)
        ]
        dist.all_gather(gathered_tensors, local_tensor)
        
        # Rank 0 deserializes and combines all SMILES
        if rank == 0:
            all_smiles = []
            for i in range(world_size):
                valid_size = sizes_list[i].item()
                chunk_bytes = gathered_tensors[i][:valid_size].cpu().numpy().tobytes()
                chunk_smiles = pickle.loads(chunk_bytes)
                all_smiles.extend(chunk_smiles)
                print(f"[Engine] Rank 0: Gathered {len(chunk_smiles)} SMILES from rank {i}")
            
            print(f"[Engine] Rank 0: Total gathered SMILES: {len(all_smiles)}")
            return all_smiles
        else:
            # Other ranks don't need the combined result
            return []

    def _gather_predictions_ddp(self, local_predictions: np.ndarray, device: torch.device) -> np.ndarray:
        """
        Gather predictions from all ranks to rank 0.
        
        Args:
            local_predictions: Predictions from this rank
            device: Device for tensor operations
            
        Returns:
            On rank 0: Combined predictions from all ranks
            On other ranks: Empty array
        """
        if not dist.is_initialized():
            return local_predictions
        
        if is_main_process():
            print(f"[Engine] Gathering predictions from {self.config.world_size} ranks...")
        
        # Use the existing utility from distributed module
        from utils.distributed import gather_ndarray_to_rank0
        
        gathered = gather_ndarray_to_rank0(local_predictions, device)
        
        if self.config.rank == 0:
            print(f"[Engine] Rank 0: Gathered {len(gathered)} predictions from all ranks")
        
        return gathered

    def _save_hdf5_predictions_with_smiles(self, predictions, smiles_list):
        """
        Save HDF5 predictions with SMILES and IDs to CSV.
        
        CRITICAL: This should ONLY be called by rank 0 in DDP mode.
        
        Args:
            predictions: Numpy array of predictions
            smiles_list: List of SMILES strings (must match predictions length)
        """
        import pandas as pd
        
        # CRITICAL FIX: Validate this is only called by rank 0 in DDP
        if self.config.ddp_enabled and self.config.rank != 0:
            print(f"[Engine] ERROR: Rank {self.config.rank} called _save_hdf5_predictions_with_smiles!")
            print(f"[Engine] This should ONLY be called by rank 0!")
            return
        
        if len(predictions) == 0:
            print("[Engine] WARNING: No predictions to save!")
            return
        
        # CRITICAL: Validate counts match
        if len(smiles_list) != len(predictions):
            raise ValueError(
                f"CRITICAL MISMATCH DETECTED!\n\n"
                f"SMILES count: {len(smiles_list)}\n"
                f"Predictions count: {len(predictions)}\n"
                f"Rank: {self.config.rank}\n"
                f"DDP enabled: {self.config.ddp_enabled}\n"
                f"World size: {self.config.world_size}\n\n"
                f"These counts MUST match or predictions will be assigned to wrong molecules!\n\n"
                f"This usually means:\n"
                f"1. SMILES gathering failed in DDP mode\n"
                f"2. Some batches were skipped inconsistently across ranks\n"
                f"3. Data loader is returning different data on different ranks\n\n"
                f"SOLUTION: Check that SMILES are gathered correctly before saving."
            )
        
        print(f"[Engine] Rank 0: Saving {len(smiles_list)} predictions to CSV...")
        
        # Create DataFrame with SMILES first
        df_dict = {'smiles': smiles_list}
        
        # Add row IDs
        df_dict['row_id'] = list(range(len(smiles_list)))
        
        # Add predictions
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-task
            for i in range(predictions.shape[1]):
                df_dict[f"prediction_{i}"] = predictions[:, i]
        else:
            # Single-task
            df_dict["prediction"] = predictions.flatten() if len(predictions.shape) > 1 else predictions
        
        # Create DataFrame and save
        pred_df = pd.DataFrame(df_dict)
        pred_df.to_csv(self.config.output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"✅ HDF5 INFERENCE COMPLETE")
        print(f"{'='*60}")
        print(f"  Molecules: {len(smiles_list):,}")
        print(f"  Output: {self.config.output_path}")
        print(f"  Columns: {', '.join(pred_df.columns)}")
        if self.config.ddp_enabled:
            print(f"  Combined from: {self.config.world_size} GPU ranks")
        print(f"{'='*60}\n")

    def _extract_hdf5_embeddings(self, inference_loader, device):
        """Extract embeddings from HDF5 inference."""
        from training.extractors import extract_embeddings_from_inference
        
        # Create a fresh loader for embedding extraction
        embedding_loader = create_iterable_pyg_dataloader(
            hdf5_path=self.config.input_path,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # Single threaded for embeddings
            shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER_SIZE,
            ddp_enabled=False,
            rank=0,
            world_size=1
        )
        
        # Mock args object for compatibility
        class MockArgs:
            def __init__(self, embeddings_path):
                self.embeddings_output_path = embeddings_path
        
        mock_args = MockArgs(self.config.embeddings_path)
        extract_embeddings_from_inference(mock_args, self.pipeline.model, embedding_loader, device)


def inference_main(args, device, is_ddp, local_rank, world_size, total_molecules=None):
    """
    Legacy compatibility function for the original inference_main.
    Now with progress tracking!
    """
    # Create config from args
    config = InferenceConfig.from_args(args)
    config.ddp_enabled = is_ddp
    config.rank = local_rank
    config.world_size = world_size
    
    try:
        # Create and run engine
        engine = InferenceEngine(config)
        engine.total_molecules = total_molecules  # Pass through for progress
        engine.run(device)
        
        # Success message
        if is_main_process():
            print("[Engine] Inference completed successfully")
            
    except Exception as e:
        print(f"[Engine] Rank {local_rank}: Error during inference: {e}")
        return