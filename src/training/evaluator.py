"""
Evaluation functionality for GNN models.

This module contains functions for evaluating trained models and computing metrics.
"""

import math
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from utils.distributed import safe_get_rank, gather_ndarray_to_rank0, gather_strings_to_rank0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task_type: str = 'regression',
    mixed_precision: bool = False,
    num_tasks: int = 1,
    preprocessing_pipeline: Optional[Any] = None,
    is_ddp: bool = False
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, List[str]]:
    """
    Evaluate model on a given data loader.
    
    FIXED: Tracks skipped batches to prevent SMILES/prediction misalignment.
    """
    model.eval()
    total_size = 0

    all_preds_list = []
    all_targets_list = []
    all_smiles_list = []
    total_loss = 0.0
    
    # FIXED: Track skipped batches
    skipped_batches = []

    for batch_idx, batch in enumerate(data_loader):
        if batch is None:
            skipped_batches.append(batch_idx)
            print(f"WARNING: Batch {batch_idx} was None, skipping")
            continue
        
        # Collect SMILES from each batch
        all_smiles_list.extend(batch.smiles_list)
            
        # Prepare batch data
        batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
        batch_indices = batch.batch_indices.to(device)
        batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}

        if isinstance(batch.targets, list):
            print(f"WARNING: Batch {batch_idx} has list targets, skipping")
            skipped_batches.append(batch_idx)
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
                print(f"WARNING: NaN found in outputs for batch {batch_idx}!")
            loss = criterion(outputs, targets)

        total_loss += loss.item() * batch_size

        # Process evidential outputs for metrics calculation
        processed_outputs = _process_evidential_outputs_for_metrics(outputs, model)

        # Collect local predictions/targets
        all_preds_list.append(processed_outputs.cpu().numpy())
        all_targets_list.append(targets.cpu().numpy())
    
    # FIXED: Report skipped batches
    if skipped_batches:
        print(f"WARNING: Skipped {len(skipped_batches)} batches during evaluation: {skipped_batches[:10]}{'...' if len(skipped_batches) > 10 else ''}")

    # Compute local average loss
    avg_loss = total_loss / (total_size if total_size > 0 else 1)

    # Calculate metrics based on task type
    if task_type == 'multitask':
        metrics = _compute_multitask_metrics(
            all_preds_list, all_targets_list, all_smiles_list, 
            avg_loss, preprocessing_pipeline
        )
    else:
        metrics = _compute_single_task_metrics(
            all_preds_list, all_targets_list, all_smiles_list,
            avg_loss, preprocessing_pipeline
        )

    # Combine metrics across ranks for DDP
    if is_ddp and dist.is_available() and dist.is_initialized():
        metrics = _combine_ddp_metrics(
            metrics, total_loss, total_size, all_preds_list, all_targets_list,
            all_smiles_list,
            device, task_type, num_tasks, preprocessing_pipeline
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
                        all_smiles_list, device, task_type, num_tasks, preprocessing_pipeline):
    """
    Combine evaluation metrics across DDP ranks.
    
    FIXED: Now handles SMILES for proper SAE inverse transform.
    """
    # All-reduce the total_loss and total_size
    local_tensor = torch.tensor([total_loss, total_size], dtype=torch.float, device=device)
    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
    global_loss_sum = local_tensor[0].item()
    global_count = local_tensor[1].item()

    if global_count > 0:
        global_avg_loss = global_loss_sum / global_count
    else:
        global_avg_loss = 0.0

    # Gather predictions/targets/smiles on rank 0, then compute final metrics
    rank = safe_get_rank()
    
    if task_type == 'multitask':
        final_metrics = _combine_multitask_ddp_metrics(
            all_preds_list, all_targets_list, all_smiles_list,
            global_avg_loss, num_tasks, preprocessing_pipeline, rank, device
        )
    else:
        final_metrics = _combine_single_task_ddp_metrics(
            all_preds_list, all_targets_list, all_smiles_list,
            global_avg_loss, preprocessing_pipeline, rank, device
        )

    # Broadcast final_metrics to all ranks
    final_metrics = _broadcast_metrics(final_metrics, rank, device)
    
    return final_metrics

def _compute_multitask_metrics(all_preds_list, all_targets_list, all_smiles_list, 
                               avg_loss, preprocessing_pipeline):
    """
    Compute metrics for multitask evaluation.
    
    FIXED: Now accepts and uses SMILES for proper inverse transform.
    """
    if len(all_preds_list) == 0:
        return {'loss': avg_loss}
    
    Y_pred = np.concatenate(all_preds_list, axis=0)
    Y_true = np.concatenate(all_targets_list, axis=0)

    # FIXED: Apply COMPLETE inverse preprocessing including SAE
    if preprocessing_pipeline is not None:
        # Ensure proper shape
        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape(-1, 1)
        if len(Y_true.shape) == 1:
            Y_true = Y_true.reshape(-1, 1)
        
        # Apply inverse transform with SMILES for SAE
        Y_pred = preprocessing_pipeline.inverse_transform(
            smiles_list=all_smiles_list,
            transformed_targets=Y_pred
        )
        Y_true = preprocessing_pipeline.inverse_transform(
            smiles_list=all_smiles_list,
            transformed_targets=Y_true
        )

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


def _compute_single_task_metrics(all_preds_list, all_targets_list, all_smiles_list,
                                 avg_loss, preprocessing_pipeline):
    """
    Compute metrics for single-task evaluation.
    
    FIXED: Validates that SMILES count matches predictions before inverse transform.
    """
    if len(all_preds_list) == 0:
        return {'loss': avg_loss}
    
    preds_np = np.concatenate(all_preds_list, axis=0)
    targets_np = np.concatenate(all_targets_list, axis=0)
    
    # CRITICAL: Validate counts match
    if len(all_smiles_list) != len(preds_np):
        raise ValueError(
            f"CRITICAL DATA CORRUPTION DETECTED!\n\n"
            f"SMILES count: {len(all_smiles_list)}\n"
            f"Predictions count: {len(preds_np)}\n"
            f"These MUST match or predictions will be assigned to wrong molecules!\n\n"
            f"This usually means:\n"
            f"1. Some batches were skipped due to errors\n"
            f"2. Data loader is dropping invalid molecules inconsistently\n"
            f"3. HDF5 file is corrupted\n\n"
            f"SOLUTION: Check logs for 'WARNING: Invalid SMILES' messages.\n"
            f"Recreate your HDF5 files or check your CSV for corrupted rows."
        )
    
    # Apply COMPLETE inverse preprocessing including SAE
    if preprocessing_pipeline is not None:
        # Ensure proper shape
        if len(preds_np.shape) == 1:
            preds_np = preds_np.reshape(-1, 1)
        if len(targets_np.shape) == 1:
            targets_np = targets_np.reshape(-1, 1)
        
        try:
            # Apply inverse transform with SMILES for SAE
            preds_np = preprocessing_pipeline.inverse_transform(
                smiles_list=all_smiles_list,
                transformed_targets=preds_np
            )
            targets_np = preprocessing_pipeline.inverse_transform(
                smiles_list=all_smiles_list,
                transformed_targets=targets_np
            )
        except Exception as e:
            raise ValueError(
                f"Preprocessing inverse transform failed!\n"
                f"Error: {e}\n\n"
                f"This usually means:\n"
                f"1. Model preprocessing doesn't match data preprocessing\n"
                f"2. SAE statistics are missing or corrupted\n"
                f"3. SMILES in data don't match training SMILES\n\n"
                f"SOLUTION: Ensure you're using the same model and data format."
            )
        
    rmse_value = math.sqrt(mean_squared_error(targets_np, preds_np))
    
    return {
        'loss': avg_loss,
        'mae': mean_absolute_error(targets_np, preds_np),
        'rmse': rmse_value,
        'r2': r2_score(targets_np, preds_np)
    }

def _combine_multitask_ddp_metrics(all_preds_list, all_targets_list, all_smiles_list,
                                   global_avg_loss, num_tasks, preprocessing_pipeline, 
                                   rank, device):
    """
    Combine multitask metrics across DDP ranks.
    
    FIXED: Now handles SMILES gathering for SAE inverse transform.
    """
    # Flatten local preds
    if len(all_preds_list) == 0:
        local_preds_np = np.zeros((0, num_tasks), dtype=np.float32)
        local_targs_np = np.zeros((0, num_tasks), dtype=np.float32)
    else:
        local_preds_np = np.concatenate(all_preds_list, axis=0).astype(np.float32)
        local_targs_np = np.concatenate(all_targets_list, axis=0).astype(np.float32)

    global_preds = gather_ndarray_to_rank0(local_preds_np, device)
    global_targs = gather_ndarray_to_rank0(local_targs_np, device)
    
    # FIXED: Gather SMILES to rank 0
    global_smiles = gather_strings_to_rank0(all_smiles_list, device)

    if rank == 0 and global_preds.shape[0] > 0:
        # FIXED: Apply COMPLETE inverse preprocessing including SAE
        if preprocessing_pipeline is not None:
            global_preds = preprocessing_pipeline.inverse_transform(
                smiles_list=global_smiles,
                transformed_targets=global_preds
            )
            global_targs = preprocessing_pipeline.inverse_transform(
                smiles_list=global_smiles,
                transformed_targets=global_targs
            )

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

def _combine_single_task_ddp_metrics(all_preds_list, all_targets_list, all_smiles_list,
                                     global_avg_loss, preprocessing_pipeline, rank, device):
    """
    Combine single-task metrics across DDP ranks.
    
    FIXED: Now handles SMILES gathering for SAE inverse transform.
    """
    # single-task regression
    if len(all_preds_list) == 0:
        local_preds_np = np.zeros((0,1), dtype=np.float32)
        local_targs_np = np.zeros((0,1), dtype=np.float32)
    else:
        local_preds_np = np.concatenate(all_preds_list, axis=0).astype(np.float32)
        local_targs_np = np.concatenate(all_targets_list, axis=0).astype(np.float32)

    global_preds = gather_ndarray_to_rank0(local_preds_np, device)
    global_targs = gather_ndarray_to_rank0(local_targs_np, device)
    
    # FIXED: Gather SMILES to rank 0
    global_smiles = gather_strings_to_rank0(all_smiles_list, device)

    if rank == 0 and global_preds.shape[0] > 0:
        # FIXED: Apply COMPLETE inverse preprocessing including SAE
        if preprocessing_pipeline is not None:
            global_preds = preprocessing_pipeline.inverse_transform(
                smiles_list=global_smiles,
                transformed_targets=global_preds
            )
            global_targs = preprocessing_pipeline.inverse_transform(
                smiles_list=global_smiles,
                transformed_targets=global_targs
            )

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