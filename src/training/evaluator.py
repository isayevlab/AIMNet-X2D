"""
Evaluation functionality for GNN models.

This module contains functions for evaluating trained models and computing metrics.
"""

import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.distributed import safe_get_rank, gather_ndarray_to_rank0


@torch.no_grad()
def evaluate(
    model, 
    data_loader, 
    criterion, 
    device, 
    task_type='regression',
    mixed_precision=False,
    num_tasks=1,
    preprocessing_pipeline=None,
    is_ddp=False
):
    """
    Evaluate model on a given data loader.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader with evaluation data
        criterion: Loss function
        device: Device to evaluate on
        task_type: Type of task ('regression' or 'multitask')
        mixed_precision: Whether to use mixed precision
        num_tasks: Number of tasks
        preprocessing_pipeline: Preprocessing pipeline for inverse transforms
        is_ddp: Whether distributed data parallel is enabled
    """
    model.eval()
    total_size = 0

    all_preds_list = []
    all_targets_list = []
    total_loss = 0.0

    for batch_idx, batch in enumerate(data_loader):
        if batch is None:
            continue
            
        # Prepare batch data
        batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
        batch_indices = batch.batch_indices.to(device)
        batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}

        if isinstance(batch.targets, list):
            continue
        else:
            targets = batch.targets.to(device)

        total_charges = batch.total_charges.to(device)
        tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
        cis_indices = batch.final_cis_tensor.to(device)
        trans_indices = batch.final_trans_tensor.to(device)

        batch_size = targets.size(0)
        total_size += batch_size

        # Forward pass
        if mixed_precision and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs, _, _ = model(
                    batch_atom_features,
                    batch_multi_hop_edges,
                    batch_indices,
                    total_charges,
                    tetrahedral_indices,
                    cis_indices,
                    trans_indices
                )
                loss = criterion(outputs, targets)
        else:
            outputs, _, _ = model(
                batch_atom_features,
                batch_multi_hop_edges,
                batch_indices,
                total_charges,
                tetrahedral_indices,
                cis_indices,
                trans_indices
            )
            if torch.isnan(outputs).any():
                print("NaN found in outputs!")
            loss = criterion(outputs, targets)

        total_loss += loss.item() * batch_size

        # Process evidential outputs for metrics calculation
        processed_outputs = _process_evidential_outputs_for_metrics(outputs, model)

        # Collect local predictions/targets
        all_preds_list.append(processed_outputs.cpu().numpy())
        all_targets_list.append(targets.cpu().numpy())

    # Compute local average loss
    avg_loss = total_loss / (total_size if total_size > 0 else 1)

    # Calculate metrics based on task type
    if task_type == 'multitask':
        metrics = _compute_multitask_metrics(all_preds_list, all_targets_list, avg_loss, preprocessing_pipeline)
    else:
        metrics = _compute_single_task_metrics(all_preds_list, all_targets_list, avg_loss, preprocessing_pipeline)

    # Combine metrics across ranks for DDP
    if is_ddp and dist.is_available() and dist.is_initialized():
        metrics = _combine_ddp_metrics(
            metrics, total_loss, total_size, all_preds_list, all_targets_list,
            device, task_type, num_tasks, preprocessing_pipeline  # CHANGED: was std_scaler
        )

    return metrics

def _process_evidential_outputs_for_metrics(outputs: torch.Tensor, model) -> torch.Tensor:
    """
    Process evidential outputs for metrics calculation.
    
    For evidential models, extract the mean prediction (gamma parameter).
    For other models, return outputs as-is.
    """
    # Check if this is an evidential model
    loss_function = getattr(model, 'loss_function', 'l1')
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        loss_function = getattr(model.module, 'loss_function', 'l1')
    
    if loss_function == 'evidential':
        # For evidential outputs: [batch_size, num_tasks * 4]
        batch_size = outputs.shape[0]
        if outputs.shape[1] % 4 == 0:
            num_tasks = outputs.shape[1] // 4
            evidential_params = outputs.view(batch_size, num_tasks, 4)
            predictions = evidential_params[:, :, 0]  # gamma (mean)
            return predictions
    
    return outputs

def _combine_ddp_metrics(metrics, total_loss, total_size, all_preds_list, all_targets_list,
                        device, task_type, num_tasks, preprocessing_pipeline):
    """Combine evaluation metrics across DDP ranks."""
    # All-reduce the total_loss and total_size
    local_tensor = torch.tensor([total_loss, total_size], dtype=torch.float, device=device)
    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
    global_loss_sum = local_tensor[0].item()
    global_count = local_tensor[1].item()

    if global_count > 0:
        global_avg_loss = global_loss_sum / global_count
    else:
        global_avg_loss = 0.0

    # Gather predictions/targets on rank 0, then compute final metrics
    rank = safe_get_rank()
    
    if task_type == 'multitask':
        final_metrics = _combine_multitask_ddp_metrics(
            all_preds_list, all_targets_list, global_avg_loss, num_tasks, preprocessing_pipeline, rank, device
        )
    else:
        final_metrics = _combine_single_task_ddp_metrics(
            all_preds_list, all_targets_list, global_avg_loss, preprocessing_pipeline, rank, device
        )

    # Broadcast final_metrics to all ranks
    final_metrics = _broadcast_metrics(final_metrics, rank, device)
    
    return final_metrics

def _compute_multitask_metrics(all_preds_list, all_targets_list, avg_loss, preprocessing_pipeline):
    """Compute metrics for multitask evaluation."""
    if len(all_preds_list) == 0:
        return {'loss': avg_loss}
    
    Y_pred = np.concatenate(all_preds_list, axis=0)
    Y_true = np.concatenate(all_targets_list, axis=0)

    # Apply inverse preprocessing if pipeline is provided
    # Note: We don't have SMILES here, so we can only do standard scaling inverse
    if preprocessing_pipeline is not None and preprocessing_pipeline.standard_scaler is not None:
        Y_pred = preprocessing_pipeline.standard_scaler.inverse_transform(Y_pred)
        Y_true = preprocessing_pipeline.standard_scaler.inverse_transform(Y_true)

    mae_vals = []
    rmse_vals = []
    r2_vals = []
    M = Y_true.shape[1]
    
    for m in range(M):
        mae_m = mean_absolute_error(Y_true[:, m], Y_pred[:, m])
        rmse_m = math.sqrt(mean_squared_error(Y_true[:, m], Y_pred[:, m]))
        r2_m = r2_score(Y_true[:, m], Y_pred[:, m])
        mae_vals.append(mae_m)
        rmse_vals.append(rmse_m)
        r2_vals.append(r2_m)

    mae_avg = float(np.mean(mae_vals))
    rmse_avg = float(np.mean(rmse_vals))
    r2_avg = float(np.mean(r2_vals))

    return {
        'loss': avg_loss,
        'mae': mae_avg,
        'rmse': rmse_avg,
        'r2': r2_avg,
        'mae_per_target': mae_vals,
        'rmse_per_target': rmse_vals,
        'r2_per_target': r2_vals
    }


def _compute_single_task_metrics(all_preds_list, all_targets_list, avg_loss, preprocessing_pipeline):
    """Compute metrics for single-task evaluation."""
    if len(all_preds_list) == 0:
        return {'loss': avg_loss}
    
    preds_np = np.concatenate(all_preds_list, axis=0)
    targets_np = np.concatenate(all_targets_list, axis=0)
    
    # Apply inverse preprocessing if pipeline is provided
    # Note: We don't have SMILES here, so we can only do standard scaling inverse
    if preprocessing_pipeline is not None and preprocessing_pipeline.standard_scaler is not None:
        preds_np = preprocessing_pipeline.standard_scaler.inverse_transform(preds_np)
        targets_np = preprocessing_pipeline.standard_scaler.inverse_transform(targets_np)
        
    rmse_value = math.sqrt(mean_squared_error(targets_np, preds_np))
    
    return {
        'loss': avg_loss,
        'mae': mean_absolute_error(targets_np, preds_np),
        'rmse': rmse_value,
        'r2': r2_score(targets_np, preds_np)
    }


def _combine_multitask_ddp_metrics(all_preds_list, all_targets_list, global_avg_loss, 
                                  num_tasks, preprocessing_pipeline, rank, device):
    """Combine multitask metrics across DDP ranks."""
    # Flatten local preds
    if len(all_preds_list) == 0:
        local_preds_np = np.zeros((0, num_tasks), dtype=np.float32)
        local_targs_np = np.zeros((0, num_tasks), dtype=np.float32)
    else:
        local_preds_np = np.concatenate(all_preds_list, axis=0).astype(np.float32)
        local_targs_np = np.concatenate(all_targets_list, axis=0).astype(np.float32)

    global_preds = gather_ndarray_to_rank0(local_preds_np, device)
    global_targs = gather_ndarray_to_rank0(local_targs_np, device)

    if rank == 0 and global_preds.shape[0] > 0:
        # Apply inverse preprocessing if pipeline is provided
        if preprocessing_pipeline is not None and preprocessing_pipeline.standard_scaler is not None:
            global_preds = preprocessing_pipeline.standard_scaler.inverse_transform(global_preds)
            global_targs = preprocessing_pipeline.standard_scaler.inverse_transform(global_targs)

        M = global_targs.shape[1]
        mae_vals = []
        rmse_vals = []
        r2_vals = []
        for m in range(M):
            mae_m = mean_absolute_error(global_targs[:, m], global_preds[:, m])
            rmse_m = math.sqrt(mean_squared_error(global_targs[:, m], global_preds[:, m]))
            r2_m = r2_score(global_targs[:, m], global_preds[:, m])
            mae_vals.append(mae_m)
            rmse_vals.append(rmse_m)
            r2_vals.append(r2_m)

        final_metrics = {
            'loss': global_avg_loss,
            'mae': float(np.mean(mae_vals)),
            'rmse': float(np.mean(rmse_vals)),
            'r2': float(np.mean(r2_vals)),
            'mae_per_target': mae_vals,
            'rmse_per_target': rmse_vals,
            'r2_per_target': r2_vals
        }
    elif rank == 0:
        # No data
        final_metrics = {'loss': global_avg_loss}
    else:
        final_metrics = {}

    return final_metrics


def _combine_single_task_ddp_metrics(all_preds_list, all_targets_list, global_avg_loss, 
                                   preprocessing_pipeline, rank, device):
    """Combine single-task metrics across DDP ranks."""
    # single-task regression
    if len(all_preds_list) == 0:
        local_preds_np = np.zeros((0,1), dtype=np.float32)
        local_targs_np = np.zeros((0,1), dtype=np.float32)
    else:
        local_preds_np = np.concatenate(all_preds_list, axis=0).astype(np.float32)
        local_targs_np = np.concatenate(all_targets_list, axis=0).astype(np.float32)

    global_preds = gather_ndarray_to_rank0(local_preds_np, device)
    global_targs = gather_ndarray_to_rank0(local_targs_np, device)

    if rank == 0 and global_preds.shape[0] > 0:
        # Apply inverse preprocessing if pipeline is provided
        if preprocessing_pipeline is not None and preprocessing_pipeline.standard_scaler is not None:
            global_preds = preprocessing_pipeline.standard_scaler.inverse_transform(global_preds)
            global_targs = preprocessing_pipeline.standard_scaler.inverse_transform(global_targs)

        mae_val = mean_absolute_error(global_targs, global_preds)
        rmse_val = math.sqrt(mean_squared_error(global_targs, global_preds))
        r2_val = r2_score(global_targs, global_preds)
        final_metrics = {
            'loss': global_avg_loss,
            'mae': mae_val,
            'rmse': rmse_val,
            'r2': r2_val
        }
    elif rank == 0:
        final_metrics = {'loss': global_avg_loss}
    else:
        final_metrics = {}

    return final_metrics

def _broadcast_metrics(final_metrics, rank, device):
    """Broadcast final metrics from rank 0 to all other ranks."""
    final_metrics_pickled = None
    if rank == 0:
        import pickle
        final_metrics_pickled = pickle.dumps(final_metrics)

    # Rank 0 sends length to others
    if rank == 0:
        length_t = torch.tensor([len(final_metrics_pickled)], dtype=torch.long, device=device)
    else:
        length_t = torch.tensor([0], dtype=torch.long, device=device)
    dist.broadcast(length_t, src=0)

    # Broadcast the actual dictionary
    if rank != 0:
        final_metrics_pickled = bytearray(length_t.item())
    final_metrics_byte = torch.ByteTensor(list(final_metrics_pickled)).to(device)
    dist.broadcast(final_metrics_byte, src=0)
    
    # Unpickle on non-zero ranks
    if rank != 0:
        import pickle
        final_metrics = pickle.loads(final_metrics_byte.cpu().numpy().tobytes())

    return final_metrics