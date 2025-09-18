"""
Uncertainty estimation for inference.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Callable, Union
from tqdm import tqdm

from utils.distributed import safe_get_rank


class UncertaintyEstimator:
    """Base class for uncertainty estimation methods."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def predict_with_uncertainty(self, data_loader, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        raise NotImplementedError

class MCDropoutPredictor(UncertaintyEstimator):
    """Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(self, model: nn.Module, device: torch.device, num_samples: int = 30):
        super().__init__(model, device)
        self.num_samples = num_samples
    
    def predict_with_uncertainty(
        self, 
        data_loader, 
        preprocessing_pipeline=None,
        show_progress: bool = True,
        embedding_callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with Monte Carlo Dropout uncertainty estimation.
        
        Args:
            data_loader: DataLoader with input data
            preprocessing_pipeline: Optional preprocessing pipeline
            show_progress: Whether to show progress bars
            embedding_callback: Optional callback for embedding extraction
            
        Returns:
            Tuple of (mean_predictions, uncertainties)
        """
        self.model.eval()
        
        # Enable dropout during inference
        self._enable_dropout()
        
        all_sample_predictions = []
        smiles_collected = False
        all_smiles = []
        
        # Progress bar for MC samples
        sample_iterator = range(self.num_samples)
        if show_progress and (safe_get_rank() == 0):
            sample_iterator = tqdm(sample_iterator, desc="MC Dropout Samples")
        
        for sample_idx in sample_iterator:
            sample_predictions = self._single_forward_pass(
                data_loader, 
                embedding_callback if sample_idx == 0 else None,  # Only extract embeddings on first pass
                collect_smiles=(not smiles_collected)
            )
            
            # Collect SMILES from first sample
            if not smiles_collected and len(sample_predictions) > 1:
                all_smiles = sample_predictions[1]  # SMILES list
                sample_predictions = sample_predictions[0]  # Predictions only
                smiles_collected = True
            elif isinstance(sample_predictions, tuple):
                sample_predictions = sample_predictions[0]  # Just predictions
            
            if len(sample_predictions) > 0:
                sample_array = np.concatenate(sample_predictions, axis=0)
                
                # Ensure proper shape
                if len(sample_array.shape) == 1:
                    sample_array = sample_array.reshape(-1, 1)
                
                # Apply inverse preprocessing if needed
                if preprocessing_pipeline is not None:
                    sample_array = preprocessing_pipeline.inverse_transform(
                        smiles_list=all_smiles,
                        transformed_targets=sample_array
                    )
                
                all_sample_predictions.append(sample_array)
        
        if len(all_sample_predictions) > 0:
            # Stack predictions: [num_samples, num_molecules, num_tasks]
            stacked = np.stack(all_sample_predictions, axis=0)
            mean_preds = np.mean(stacked, axis=0)
            uncertainties = np.std(stacked, axis=0)
            return mean_preds, uncertainties
        
        return np.array([]), np.array([])
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        def enable_dropout_fn(module):
            if isinstance(module, nn.Dropout):
                module.train()
        
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.apply(enable_dropout_fn)
        else:
            self.model.apply(enable_dropout_fn)
    
    def _single_forward_pass(self, data_loader, embedding_callback: Optional[Callable] = None, collect_smiles: bool = False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[str]]]:
        """Perform a single forward pass through the data."""
        predictions = []
        smiles_list = [] if collect_smiles else None
        
        with torch.no_grad():
            for batch in data_loader:
                if batch is None:
                    continue
                
                # Collect SMILES if requested
                if collect_smiles:
                    smiles_list.extend(batch.smiles_list)
                
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
                
                predictions.append(outputs.detach().cpu().numpy())
        
        if collect_smiles:
            return predictions, smiles_list
        else:
            return predictions

class DeterministicPredictor:
    """Standard deterministic prediction without uncertainty."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def predict(
        self, 
        data_loader, 
        preprocessing_pipeline=None,
        embedding_callback: Optional[Callable] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Make deterministic predictions.
        
        Args:
            data_loader: DataLoader with input data
            preprocessing_pipeline: Optional preprocessing pipeline
            embedding_callback: Optional callback for embedding extraction
            show_progress: Whether to show progress bars
            
        Returns:
            Array of predictions
        """
        self.model.eval()
        predictions = []
        all_smiles = []  # Collect SMILES for SAE inverse transform
        
        # Add progress bar for batch processing
        batch_iterator = data_loader
        if show_progress:
            batch_iterator = tqdm(
                data_loader, 
                desc="Inference batches", 
                unit="batch",
                leave=False
            )
        
        with torch.no_grad():
            for batch in batch_iterator:
                if batch is None:
                    continue
                
                # Collect SMILES for SAE inverse transform
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
                
                predictions.append(outputs.detach().cpu().numpy())
        
        if predictions:
            all_preds = np.concatenate(predictions, axis=0)
            
            # Ensure proper shape for inverse transform
            if len(all_preds.shape) == 1:
                all_preds = all_preds.reshape(-1, 1)
            
            # Apply complete inverse preprocessing (including SAE)
            if preprocessing_pipeline is not None:
                all_preds = preprocessing_pipeline.inverse_transform(
                    smiles_list=all_smiles,
                    transformed_targets=all_preds
                )
            
            return all_preds
        
        return np.array([])