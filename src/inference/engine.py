"""
High-level inference engine and legacy compatibility.
"""

import torch
import torch.distributed as dist
from typing import Optional

from .config import InferenceConfig
from .pipeline import InferencePipeline
from datasets import create_iterable_pyg_dataloader
from training import predict_gnn
from utils.distributed import safe_get_rank, is_main_process


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
        """Run inference on HDF5 input."""
        if is_main_process():
            print(f"[Engine] Running HDF5 inference")
        
        # Create data loader
        inference_loader = create_iterable_pyg_dataloader(
            hdf5_path=self.config.input_path,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            shuffle_buffer_size=1000,
            ddp_enabled=self.config.ddp_enabled,
            rank=self.config.rank,
            world_size=self.config.world_size
        )
        
        # Run inference
        predictions = predict_gnn(
            model=self.pipeline.model,
            data_loader=inference_loader,
            device=device,
            task_type=self.pipeline.model.task_type,
            preprocessing_pipeline=self.pipeline.preprocessing_pipeline,  # NEW API
            is_ddp=self.config.ddp_enabled
        )
        
        # Save predictions
        if is_main_process():
            self._save_hdf5_predictions(predictions)
        
        # Extract embeddings if requested
        if self.config.save_embeddings and is_main_process():
            self._extract_hdf5_embeddings(inference_loader, device)
    
    def _save_hdf5_predictions(self, predictions):
        """Save HDF5 predictions to CSV."""
        import pandas as pd
        
        # Determine column names
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            columns = [f"prediction_{i}" for i in range(predictions.shape[1])]
        else:
            columns = ["prediction"]
        
        # Save to CSV
        pred_df = pd.DataFrame(predictions, columns=columns)
        pred_df.to_csv(self.config.output_path, index=False)
        
        print(f"[Engine] HDF5 predictions saved to: {self.config.output_path}")
    
    def _extract_hdf5_embeddings(self, inference_loader, device):
        """Extract embeddings from HDF5 inference."""
        from training.extractors import extract_embeddings_from_inference
        
        # Create a fresh loader for embedding extraction
        embedding_loader = create_iterable_pyg_dataloader(
            hdf5_path=self.config.input_path,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # Single threaded for embeddings
            shuffle_buffer_size=1000,
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


def inference_main(args, device, is_ddp, local_rank, world_size):
    """Legacy compatibility function for the original inference_main."""
    # Create config from args
    config = InferenceConfig.from_args(args)
    config.ddp_enabled = is_ddp
    config.rank = local_rank
    config.world_size = world_size
    
    try:
        # Create and run engine
        engine = InferenceEngine(config)
        engine.run(device)
        
        # Success message
        if is_main_process():
            print("[Engine] Inference completed successfully")
            
    except Exception as e:
        print(f"[Engine] Rank {local_rank}: Error during inference: {e}")
        # Don't re-raise the error, just exit gracefully
        return