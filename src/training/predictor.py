"""
Prediction functionality for trained GNN models.

This module contains functions for making predictions and uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import tqdm

from utils.distributed import safe_get_rank, gather_ndarray_to_rank0


@torch.no_grad()
def predict_gnn(
    model,
    data_loader,
    device,
    task_type='regression',
    preprocessing_pipeline=None,
    is_ddp=False,
    show_progress=True,  # NEW parameter
    progress_total=None  # NEW parameter
):
    """
    Returns predictions in CPU numpy form with optional progress bar.
    
    Args:
        show_progress: Whether to show progress bar
        progress_total: Total number of items (for progress bar)
    """
    model.eval()
    all_preds = []
    all_smiles = []
    
    # Setup progress bar
    is_main = not is_ddp or safe_get_rank() == 0
    
    if show_progress and is_main:
        # Try to get total from data_loader or use provided total
        try:
            total = len(data_loader) if hasattr(data_loader, '__len__') else progress_total
            if total:
                pbar = tqdm.tqdm(total=total, desc="Inference", unit="batch")
            else:
                pbar = tqdm.tqdm(desc="Inference", unit="batch")
        except:
            pbar = tqdm.tqdm(desc="Inference", unit="batch")
    else:
        pbar = None

    for batch_idx, batch in enumerate(data_loader):
        if batch is None:
            continue
            
        # Prepare batch data
        batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
        batch_indices = batch.batch_indices.to(device)
        batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
        total_charges = batch.total_charges.to(device)
        tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
        cis_indices = batch.final_cis_tensor.to(device)
        trans_indices = batch.final_trans_tensor.to(device)

        # Forward pass
        outputs, _, _ = model(
            batch_atom_features,
            batch_multi_hop_edges,
            batch_indices,
            total_charges,
            tetrahedral_indices,
            cis_indices,
            trans_indices
        )

        # Process evidential outputs to get predictions
        predictions = _process_evidential_outputs(outputs, model)
        
        all_preds.append(predictions.detach().cpu().numpy())
        all_smiles.extend(batch.smiles_list)
        
        # Update progress bar
        if pbar is not None:
            pbar.update(1)
            # Show molecule count in postfix
            pbar.set_postfix({'molecules': len(all_smiles)})

    # Close progress bar
    if pbar is not None:
        pbar.close()

    if len(all_preds) == 0:
        local_preds = np.array([])
    else:
        local_preds = np.concatenate(all_preds, axis=0)
        local_smiles = all_smiles
        
        # Ensure proper shape for inverse transform
        if task_type in ['regression', 'multitask'] and len(local_preds.shape) == 1:
            local_preds = local_preds.reshape(-1, 1)
            
        # Apply complete inverse preprocessing
        if preprocessing_pipeline is not None and task_type in ['regression', 'multitask']:
            if is_main:
                print("Applying inverse preprocessing...")
            local_preds = preprocessing_pipeline.inverse_transform(
                smiles_list=local_smiles,
                transformed_targets=local_preds
            )

    # If DDP is enabled, combine predictions across all ranks
    if is_ddp and dist.is_initialized():
        return _combine_ddp_predictions(local_preds, device)
    else:
        return local_preds

@torch.no_grad()
def predict_with_mc_dropout(
    model,
    data_loader,
    device,
    num_samples=30,
    task_type='regression',
    preprocessing_pipeline=None,
    is_ddp=False
):
    """
    Get predictions with uncertainty estimates using Monte Carlo Dropout.
    
    FIXED: Improved progress bar UI - shows total operations instead of nested bars.
    """
    # Set model to eval mode
    model.eval()
    
    # Enable dropout layers during inference
    def enable_dropout(m):
        if type(m) == nn.Dropout:
            m.train()
    
    # Apply to model (handle DDP wrapper if present)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.apply(enable_dropout)
    else:
        model.apply(enable_dropout)
    
    all_predictions = []
    smiles_collected = False
    all_smiles = []
    
    # Progress tracking
    is_main = not is_ddp or safe_get_rank() == 0
    
    # FIXED: Calculate total operations for clearer progress tracking
    try:
        num_batches = len(data_loader)
    except:
        num_batches = None
    
    if num_batches is not None and is_main:
        total_ops = num_samples * num_batches
        print(f"[MC Dropout] Running {num_samples} samples × {num_batches} batches = {total_ops} total operations")
        pbar = tqdm.tqdm(total=total_ops, desc="MC Dropout Inference")
    else:
        pbar = None
    
    # Run MC dropout samples
    for sample_idx in range(num_samples):
        sample_preds = []
        sample_smiles = []
        
        for batch in data_loader:
            if batch is None:
                continue
                
            # Collect SMILES only once
            if not smiles_collected:
                sample_smiles.extend(batch.smiles_list)
            
            # Prepare batch data
            batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
            batch_indices = batch.batch_indices.to(device)
            batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
            total_charges = batch.total_charges.to(device)
            tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
            cis_indices = batch.final_cis_tensor.to(device)
            trans_indices = batch.final_trans_tensor.to(device)

            # Forward pass
            outputs, _, _ = model(
                batch_atom_features,
                batch_multi_hop_edges,
                batch_indices,
                total_charges,
                tetrahedral_indices,
                cis_indices,
                trans_indices
            )
            
            # Process evidential outputs
            outputs = _process_evidential_outputs(outputs, model)
            
            sample_preds.append(outputs.detach().cpu().numpy())
            
            # FIXED: Update progress bar per batch
            if pbar is not None:
                pbar.update(1)
        
        # Store SMILES from first sample
        if not smiles_collected:
            all_smiles = sample_smiles
            smiles_collected = True
        
        if len(sample_preds) > 0:
            # Concatenate all batch predictions for this MC sample
            sample_pred_array = np.concatenate(sample_preds, axis=0)
            
            # Ensure proper shape for inverse transform
            if task_type in ['regression', 'multitask'] and len(sample_pred_array.shape) == 1:
                sample_pred_array = sample_pred_array.reshape(-1, 1)
            
            # Apply complete inverse preprocessing
            if preprocessing_pipeline is not None and task_type in ['regression', 'multitask']:
                sample_pred_array = preprocessing_pipeline.inverse_transform(
                    smiles_list=all_smiles,
                    transformed_targets=sample_pred_array
                )
                
            all_predictions.append(sample_pred_array)
    
    # Close progress bar
    if pbar is not None:
        pbar.close()
    
    # If we have predictions, calculate statistics
    if len(all_predictions) > 0:
        # Stack to shape [num_samples, num_molecules, num_tasks]
        stacked_predictions = np.stack(all_predictions, axis=0)
        
        # Calculate mean and standard deviation across samples
        mean_predictions = np.mean(stacked_predictions, axis=0)
        uncertainties = np.std(stacked_predictions, axis=0)
        
        # Handle DDP if needed
        if is_ddp and dist.is_initialized():
            mean_predictions = _combine_ddp_predictions(mean_predictions, device)
            uncertainties = _combine_ddp_predictions(uncertainties, device)
        
        if is_main:
            print(f"[MC Dropout] Completed: {num_samples} samples processed")
        
        return mean_predictions, uncertainties
    else:
        return np.array([]), np.array([])

@torch.no_grad()
def predict_gnn_with_smiles(model, data_loader, device, task_type, preprocessing_pipeline=None, is_ddp=False):
    """
    Get predictions with corresponding SMILES strings.
    
    Args:
        model: Model to use for prediction
        data_loader: DataLoader with input data
        device: Device to run inference on
        task_type: Type of task ('regression' or 'multitask')
        preprocessing_pipeline: Preprocessing pipeline for inverse transforms
        is_ddp: Whether DDP is enabled
        
    Returns:
        Tuple of (predictions, smiles_list)
    """
    model.eval()
    all_preds = []
    all_smiles = []

    for batch in data_loader:
        if batch is None:
            continue

        # Prepare inputs
        batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
        batch_indices = batch.batch_indices.to(device)
        batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
        total_charges = batch.total_charges.to(device)
        tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
        cis_indices = batch.final_cis_tensor.to(device)
        trans_indices = batch.final_trans_tensor.to(device)

        # Forward pass
        outputs, _, _ = model(
            batch_atom_features,
            batch_multi_hop_edges,
            batch_indices,
            total_charges,
            tetrahedral_indices,
            cis_indices,
            trans_indices
        )

        # Process evidential outputs
        outputs = _process_evidential_outputs(outputs, model)

        preds_np = outputs.detach().cpu().numpy()
        all_preds.append(preds_np)
        all_smiles.extend(batch.smiles_list)

    if len(all_preds) == 0:
        return np.array([], dtype=np.float32), []

    preds_array = np.concatenate(all_preds, axis=0)
    
    # Ensure proper shape for inverse transform
    if task_type in ['regression', 'multitask'] and len(preds_array.shape) == 1:
        preds_array = preds_array.reshape(-1, 1)
    
    # Apply complete inverse preprocessing
    if preprocessing_pipeline is not None and task_type in ["regression", "multitask"]:
        preds_array = preprocessing_pipeline.inverse_transform(
            smiles_list=all_smiles,
            transformed_targets=preds_array
        )

    # Note: For DDP, SMILES collection would need special handling
    # This function is typically used for single-process inference
    if is_ddp and dist.is_initialized():
        preds_array = _combine_ddp_predictions(preds_array, device)
        # For SMILES, we'd need to gather string lists across ranks
        from utils.distributed import gather_strings_to_rank0
        all_smiles = gather_strings_to_rank0(all_smiles, device)

    return preds_array, all_smiles

@torch.no_grad()
def predict_evidential_with_uncertainty(
    model,
    data_loader,
    device,
    task_type='regression',
    preprocessing_pipeline=None,  # CHANGED: was std_scaler
    is_ddp=False
):
    """
    Get predictions with uncertainties from evidential learning.
    
    Args:
        model: Model to use for prediction (must be trained with evidential loss)
        data_loader: DataLoader with input data
        device: Device to run inference on
        task_type: Type of task ('regression' or 'multitask')
        preprocessing_pipeline: Preprocessing pipeline for inverse transforms
        is_ddp: Whether DDP is enabled
        
    Returns:
        Tuple of (predictions, uncertainties)
    """
    model.eval()
    all_preds = []
    all_uncertainties = []
    all_smiles = []

    for batch in data_loader:
        if batch is None:
            continue
            
        # Collect SMILES for SAE inverse transform
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
        outputs, _, _ = model(
            batch_atom_features,
            batch_multi_hop_edges,
            batch_indices,
            total_charges,
            tetrahedral_indices,
            cis_indices,
            trans_indices
        )

        # Process evidential outputs to get predictions and uncertainties
        predictions, uncertainties = _process_evidential_outputs_with_uncertainty(outputs, model)
        
        all_preds.append(predictions.detach().cpu().numpy())
        all_uncertainties.append(uncertainties.detach().cpu().numpy())

    if len(all_preds) == 0:
        local_preds = np.array([])
        local_uncertainties = np.array([])
    else:
        # Combine predictions and uncertainties
        local_preds = np.concatenate(all_preds, axis=0)
        local_uncertainties = np.concatenate(all_uncertainties, axis=0)
        
        # Ensure proper shape for inverse transform
        if task_type in ['regression', 'multitask'] and len(local_preds.shape) == 1:
            local_preds = local_preds.reshape(-1, 1)
            local_uncertainties = local_uncertainties.reshape(-1, 1)
        
        # Apply complete inverse preprocessing (including SAE)
        if preprocessing_pipeline is not None and task_type in ['regression', 'multitask']:
            local_preds = preprocessing_pipeline.inverse_transform(
                smiles_list=all_smiles,
                transformed_targets=local_preds
            )

    # Handle DDP if needed
    if is_ddp and dist.is_initialized():
        local_preds = _combine_ddp_predictions(local_preds, device)
        local_uncertainties = _combine_ddp_predictions(local_uncertainties, device)

    return local_preds, local_uncertainties


def _process_evidential_outputs(outputs: torch.Tensor, model) -> torch.Tensor:
    """
    Process evidential outputs to extract predictions.
    
    For evidential learning, the model outputs 4 parameters per task.
    We extract the gamma (mean) parameter as the prediction.
    
    Args:
        outputs: Raw model outputs
        model: The model (to check loss function)
        
    Returns:
        Processed predictions
    """
    # Check if this is an evidential model
    loss_function = getattr(model, 'loss_function', 'l1')
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        loss_function = getattr(model.module, 'loss_function', 'l1')
    
    if loss_function == 'evidential':
        # For evidential outputs: [batch_size, num_tasks * 4]
        # Reshape to [batch_size, num_tasks, 4] and extract gamma (index 0)
        batch_size = outputs.shape[0]
        if outputs.shape[1] % 4 == 0:
            num_tasks = outputs.shape[1] // 4
            evidential_params = outputs.view(batch_size, num_tasks, 4)
            predictions = evidential_params[:, :, 0]  # gamma (mean)
            return predictions
    
    # For non-evidential models, return outputs as-is
    return outputs


def _process_evidential_outputs_with_uncertainty(outputs: torch.Tensor, model) -> tuple:
    """
    Process evidential outputs to extract both predictions and uncertainties.
    
    Args:
        outputs: Raw model outputs from evidential model
        model: The model
        
    Returns:
        Tuple of (predictions, uncertainties)
    """
    # Check if this is an evidential model
    loss_function = getattr(model, 'loss_function', 'l1')
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        loss_function = getattr(model.module, 'loss_function', 'l1')
    
    if loss_function == 'evidential':
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
            # Aleatoric uncertainty: beta / (alpha - 1) for alpha > 1
            aleatoric = beta / torch.clamp(alpha - 1, min=1e-6)
            
            # Epistemic uncertainty: beta / (nu * (alpha - 1)) for alpha > 1
            epistemic = beta / (nu * torch.clamp(alpha - 1, min=1e-6))
            
            # Total uncertainty
            total_uncertainty = aleatoric + epistemic
            
            return gamma, total_uncertainty
    
    # For non-evidential models, return zeros for uncertainty
    uncertainties = torch.zeros_like(outputs)
    return outputs, uncertainties


def _combine_ddp_predictions(local_preds, device):
    """
    Combine predictions across DDP ranks.
    
    Args:
        local_preds: Local predictions as numpy array
        device: Device for tensor operations
        
    Returns:
        Combined predictions (on rank 0) or local predictions (on other ranks)
    """
    # Convert local predictions to proper format for gathering
    if len(local_preds) == 0:
        # Handle empty predictions
        local_preds_formatted = np.array([]).reshape(0, 1)
    elif len(local_preds.shape) == 1:
        local_preds_formatted = local_preds.reshape(-1, 1)
    else:
        local_preds_formatted = local_preds
    
    # Use the gather function from distributed utils
    combined_preds = gather_ndarray_to_rank0(local_preds_formatted, device)
    
    # Return appropriate result based on rank
    rank = safe_get_rank()
    if rank == 0:
        return combined_preds
    else:
        return local_preds


@torch.no_grad()
def predict_with_uncertainty_estimation(
    model,
    data_loader, 
    device,
    uncertainty_method='mc_dropout',
    num_samples=30,
    task_type='regression',
    preprocessing_pipeline=None,  # CHANGED: was std_scaler
    is_ddp=False
):
    """
    Unified function for prediction with uncertainty estimation.
    
    Args:
        model: Model to use for prediction
        data_loader: DataLoader with input data
        device: Device to run inference on
        uncertainty_method: Method for uncertainty estimation ('mc_dropout', 'evidential')
        num_samples: Number of MC dropout samples (ignored for evidential)
        task_type: Type of task ('regression' or 'multitask')
        preprocessing_pipeline: Preprocessing pipeline for inverse transforms
        is_ddp: Whether DDP is enabled
        
    Returns:
        Tuple of (predictions, uncertainties)
    """
    # Check the model's loss function to determine appropriate uncertainty method
    loss_function = getattr(model, 'loss_function', 'l1')
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        loss_function = getattr(model.module, 'loss_function', 'l1')
    
    if uncertainty_method == 'evidential' or loss_function == 'evidential':
        return predict_evidential_with_uncertainty(
            model, data_loader, device, task_type, preprocessing_pipeline, is_ddp
        )
    elif uncertainty_method == 'mc_dropout':
        return predict_with_mc_dropout(
            model, data_loader, device, num_samples, task_type, preprocessing_pipeline, is_ddp
        )
    else:
        raise ValueError(f"Unsupported uncertainty method: {uncertainty_method}")

@torch.no_grad()
def predict_batch_with_processing(
    model,
    batch,
    device,
    task_type='regression',
    preprocessing_pipeline=None,
    return_uncertainty=False,
    uncertainty_method='evidential'
):
    """
    Process a single batch and return predictions (and optionally uncertainties).
    
    This is a utility function for processing individual batches,
    useful for streaming inference or custom prediction loops.
    
    Args:
        model: Model to use for prediction
        batch: Single batch of data
        device: Device to run inference on
        task_type: Type of task ('regression' or 'multitask')
        preprocessing_pipeline: Preprocessing pipeline for inverse transforms
        return_uncertainty: Whether to return uncertainty estimates
        uncertainty_method: Method for uncertainty estimation
        
    Returns:
        Predictions (and uncertainties if requested)
    """
    if batch is None:
        if return_uncertainty:
            return np.array([]), np.array([])
        else:
            return np.array([])
    
    model.eval()
    
    # Prepare batch data
    batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
    batch_indices = batch.batch_indices.to(device)
    batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
    total_charges = batch.total_charges.to(device)
    tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
    cis_indices = batch.final_cis_tensor.to(device)
    trans_indices = batch.final_trans_tensor.to(device)

    # Forward pass
    outputs, _, _ = model(
        batch_atom_features,
        batch_multi_hop_edges,
        batch_indices,
        total_charges,
        tetrahedral_indices,
        cis_indices,
        trans_indices
    )

    # Process outputs based on whether uncertainty is requested
    if return_uncertainty:
        predictions, uncertainties = _process_evidential_outputs_with_uncertainty(outputs, model)
        predictions_np = predictions.detach().cpu().numpy()
        uncertainties_np = uncertainties.detach().cpu().numpy()
    else:
        predictions = _process_evidential_outputs(outputs, model)
        predictions_np = predictions.detach().cpu().numpy()
        uncertainties_np = None
    
    # Apply complete inverse preprocessing (standard scaling + SAE)
    if preprocessing_pipeline is not None:
        # Ensure proper shape
        if len(predictions_np.shape) == 1:
            predictions_np = predictions_np.reshape(-1, 1)
        
        # Apply inverse preprocessing using SMILES from batch
        predictions_np = preprocessing_pipeline.inverse_transform(
            smiles_list=batch.smiles_list,
            transformed_targets=predictions_np
        )
    
    if return_uncertainty:
        return predictions_np, uncertainties_np
    else:
        return predictions_np