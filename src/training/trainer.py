"""
Core training functionality for GNN models.

This module contains the main training loop and related utilities.
"""

import math
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
import tqdm
import wandb

from utils import get_layer_wise_learning_rates, is_main_process, safe_get_rank
from utils.logging import get_logger
from models import WeightedL1Loss, WeightedMSELoss, EvidentialLoss, WeightedEvidentialLoss
from .evaluator import evaluate

logger = get_logger(__name__)




def _setup_loss_function(current_args, task_type, multitask_weights):
    """Setup the appropriate loss function based on configuration."""
    if current_args.loss_function == 'l1':
        if task_type == 'multitask':
            w_tensor = torch.tensor(multitask_weights, dtype=torch.float)
            criterion = WeightedL1Loss(w_tensor)
            if safe_get_rank() == 0:
                logger.info(f"Using WeightedL1Loss for multitask with weights = {multitask_weights}")
        elif task_type == 'regression':
            criterion = nn.L1Loss()
    elif current_args.loss_function == 'mse':
        if task_type == 'multitask':
            w_tensor = torch.tensor(multitask_weights, dtype=torch.float)
            criterion = WeightedMSELoss(w_tensor)
            if safe_get_rank() == 0:
                logger.info(f"Using WeightedMSELoss for multitask with weights = {multitask_weights}")
        elif task_type == 'regression':
            criterion = nn.MSELoss()
    elif current_args.loss_function == 'evidential':
        lambda_reg = getattr(current_args, 'evidential_lambda', 1.0)
        if task_type == 'multitask':
            w_tensor = torch.tensor(multitask_weights, dtype=torch.float)
            criterion = WeightedEvidentialLoss(w_tensor, lambda_reg=lambda_reg)
            if safe_get_rank() == 0:
                logger.info(f"Using WeightedEvidentialLoss for multitask with weights = {multitask_weights}, lambda = {lambda_reg}")
        elif task_type == 'regression':
            criterion = EvidentialLoss(lambda_reg=lambda_reg)
            if safe_get_rank() == 0:
                logger.info(f"Using EvidentialLoss for regression with lambda = {lambda_reg}")
    else:
        raise ValueError(f"Invalid loss function: {current_args.loss_function}")
    
    return criterion


def _setup_scheduler(optimizer, current_args):
    """Setup the learning rate scheduler based on configuration."""
    if current_args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=current_args.lr_reduce_factor,
            patience=int(current_args.lr_patience),
            verbose=True
        )
    elif current_args.lr_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=current_args.lr_cosine_t_max,
            eta_min=0,
            verbose=True
        )
    elif current_args.lr_scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=current_args.lr_step_size,
            gamma=current_args.lr_step_gamma,
            verbose=True
        )
    elif current_args.lr_scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=current_args.lr_exp_gamma,
            verbose=True
        )
    else:
        scheduler = None
    
    return scheduler


def _maybe_set_epoch(loader, epoch):
    """Set epoch for DistributedSampler if present."""
    if hasattr(loader, "sampler") and isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
        loader.sampler.set_epoch(epoch)


def _training_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch, num_epochs):
    """Execute one training epoch."""
    epoch_local_loss_sum = 0.0
    epoch_local_count = 0

    pbar = tqdm.tqdm(enumerate(train_loader),
                   total=len(train_loader),
                   desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, batch in pbar:
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

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision if enabled
        if scaler is not None:
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
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
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
                logger.warning("NaN found in outputs!")
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Accumulate local sums for train loss
        batch_size = targets.size(0)
        epoch_local_loss_sum += loss.item() * batch_size
        epoch_local_count += batch_size

    # Calculate epoch training loss (with DDP reduction if needed)
    if dist.is_available() and dist.is_initialized():
        local_tensor = torch.tensor([epoch_local_loss_sum, epoch_local_count],
                                  dtype=torch.float, device=device)
        dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
        global_sum = local_tensor[0].item()
        global_count = local_tensor[1].item()
        epoch_train_loss = global_sum / global_count if global_count > 0 else 0.0
    else:
        epoch_train_loss = epoch_local_loss_sum / epoch_local_count if epoch_local_count > 0 else 0.0

    return epoch_train_loss

def train_gnn(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    learning_rate,
    device,
    early_stopping=False,
    task_type='regression',
    mixed_precision=False,
    num_tasks=1,
    multitask_weights=None,
    preprocessing_pipeline=None,  # CHANGED: was std_scaler
    is_ddp=False,
    current_args=None
):
    """
    Train a GNN model with properly implemented early stopping.
    """
    # Initialize model weights
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.init_weights()
    else:
        model.init_weights()
        
    model.train()

    # Setup optimizer
    if current_args.layer_wise_lr_decay:
        model_to_use = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        parameter_groups = get_layer_wise_learning_rates(
            model_to_use,
            learning_rate, 
            decay_factor=current_args.lr_decay_factor
        )
        optimizer = torch.optim.Adam(parameter_groups)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup loss function and scheduler
    criterion = _setup_loss_function(current_args, task_type, multitask_weights)
    scheduler = _setup_scheduler(optimizer, current_args)

    # Setup early stopping - FIXED INITIALIZATION
    patience = current_args.patience
    best_val_loss = float('inf')
    best_metrics = None
    patience_counter = 0
    best_model_state = None  # Renamed for clarity
    best_epoch = 0

    # Setup mixed precision scaler if enabled
    scaler = torch.cuda.amp.GradScaler() if (mixed_precision and device.type == 'cuda') else None

    # Track epoch times
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        _maybe_set_epoch(train_loader, epoch)
        model.train()

        # Training loop
        epoch_train_loss = _training_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch, num_epochs
        )

        # Evaluate on validation set
        with torch.no_grad():
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                task_type=task_type,
                mixed_precision=mixed_precision,
                num_tasks=num_tasks,
                preprocessing_pipeline=preprocessing_pipeline,  # CHANGED: was std_scaler
                is_ddp=is_ddp
            )

        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # FIXED: Handle early stopping with proper state management
        (stop_training, best_val_loss, patience_counter, best_model_state, 
         best_metrics, best_epoch) = _handle_epoch_end_fixed(
            model=model,
            val_metrics=val_metrics,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            best_model_state=best_model_state,
            best_metrics=best_metrics,
            epoch=epoch,
            early_stopping=early_stopping,
            patience=patience,
            current_args=current_args,
            optimizer=optimizer,
            device=device,
            epoch_train_loss=epoch_train_loss,
            epoch_duration=epoch_duration,
            task_type=task_type,
            is_ddp=is_ddp
        )

        if stop_training:
            if is_main_process():
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
            break

    # FIXED: Load best model if early stopping was used
    if early_stopping and best_model_state is not None:
        if is_main_process():
            logger.info(f"Loading best model from epoch {best_epoch}")
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)

    # Broadcast best model to all processes (for DDP)
    if is_ddp and dist.is_initialized():
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    # Log final metrics
    if len(epoch_times) > 0:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        if is_main_process():
            logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")

    if best_metrics is not None and is_main_process():
        if current_args.enable_wandb:
            wandb.run.summary.update(best_metrics)
            wandb.run.summary.update({"avg_epoch_time": avg_epoch_time})

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    return model

def _handle_epoch_end_fixed(
    model, val_metrics, best_val_loss, patience_counter, best_model_state,
    best_metrics, epoch, early_stopping, patience, current_args, optimizer,
    device, epoch_train_loss, epoch_duration, task_type, is_ddp
):
    """
    FIXED: Handle end-of-epoch processing with proper return values.
    
    Returns:
        Tuple of (stop_training, best_val_loss, patience_counter, 
                 best_model_state, best_metrics, best_epoch)
    """
    stop_training = False
    current_val_loss = val_metrics['loss']
    best_epoch = epoch + 1  # Default to current epoch

    # Only main process handles early stopping logic
    if (not dist.is_initialized()) or (safe_get_rank() == 0):
        
        # Check if this is the best model so far
        if current_val_loss < best_val_loss:
            # Update best metrics
            best_val_loss = current_val_loss
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience
            
            # Save best model state
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                best_model_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
            else:
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Update best metrics for logging
            best_metrics = {
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                **{f"best_val_{k}": v for k, v in val_metrics.items()},
            }
            
            if is_main_process():
                logger.info(f"New best model at epoch {best_epoch}: val_loss = {best_val_loss:.6f}")

        else:
            # No improvement
            patience_counter += 1
            if is_main_process():
                logger.debug(f"No improvement for {patience_counter}/{patience} epochs")

        # Log current epoch metrics
        _print_epoch_progress(epoch, val_metrics, epoch_train_loss, optimizer, task_type)

        # Log to wandb if enabled
        if current_args.enable_wandb:
            wandb_dict = {
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "val_loss": current_val_loss,
                "epoch_time": epoch_duration,
                "patience_counter": patience_counter,
                "best_val_loss": best_val_loss,
            }

            if task_type == 'multitask':
                wandb_dict.update({
                    "val_mae_avg": val_metrics.get('mae'),
                    "val_rmse_avg": val_metrics.get('rmse'),
                    "val_r2_avg": val_metrics.get('r2'),
                })
                if 'mae_per_target' in val_metrics:
                    for i, mae_i in enumerate(val_metrics['mae_per_target']):
                        wandb_dict[f"val_mae_target_{i}"] = mae_i
                    for i, rmse_i in enumerate(val_metrics['rmse_per_target']):
                        wandb_dict[f"val_rmse_target_{i}"] = rmse_i
                    for i, r2_i in enumerate(val_metrics['r2_per_target']):
                        wandb_dict[f"val_r2_target_{i}"] = r2_i
            else:
                wandb_dict.update({
                    "val_mae": val_metrics.get('mae'),
                    "val_rmse": val_metrics.get('rmse'),
                    "val_r2": val_metrics.get('r2'),
                })
            
            if is_main_process():
                wandb.log(wandb_dict)

        # Check early stopping condition
        if early_stopping and patience_counter >= patience:
            if is_main_process():
                logger.info(f"Early stopping triggered! No improvement for {patience} epochs.")
                logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
            stop_training = True

    # Broadcast early stopping decision to all processes (for DDP)
    if is_ddp and dist.is_initialized():
        # Create tensors for broadcasting
        stop_tensor = torch.tensor([1 if stop_training else 0], dtype=torch.uint8, device=device)
        best_loss_tensor = torch.tensor([best_val_loss], dtype=torch.float, device=device)
        patience_tensor = torch.tensor([patience_counter], dtype=torch.int, device=device)
        best_epoch_tensor = torch.tensor([best_epoch], dtype=torch.int, device=device)
        
        # Broadcast from rank 0
        dist.broadcast(stop_tensor, src=0)
        dist.broadcast(best_loss_tensor, src=0)
        dist.broadcast(patience_tensor, src=0)
        dist.broadcast(best_epoch_tensor, src=0)
        
        # Update values on non-main processes
        stop_training = stop_tensor.item() == 1
        best_val_loss = best_loss_tensor.item()
        patience_counter = patience_tensor.item()
        best_epoch = best_epoch_tensor.item()

    return (stop_training, best_val_loss, patience_counter, 
            best_model_state, best_metrics, best_epoch)


def _print_epoch_progress(epoch, val_metrics, epoch_train_loss, optimizer, task_type):
    """Print training progress for the current epoch."""
    if task_type == 'multitask':
        logger.info(f"Epoch {epoch+1} | LR: {optimizer.param_groups[0]['lr']:.8f}")
        logger.info(f"[Train Loss: {epoch_train_loss:.5f}] "
              f"Val Loss: {val_metrics['loss']:.5f}, MAE: {val_metrics['mae']:.5f}, "
              f"RMSE: {val_metrics['rmse']:.5f}, R2: {val_metrics['r2']:.5f}")
        if 'mae_per_target' in val_metrics:
            for i, (mae_i, rmse_i, r2_i) in enumerate(zip(
                val_metrics['mae_per_target'],
                val_metrics['rmse_per_target'],
                val_metrics['r2_per_target']
            )):
                logger.info(f"  [Val Target {i}] MAE={mae_i:.5f}, RMSE={rmse_i:.5f}, R2={r2_i:.5f}")
    else:
        logger.info(f"Epoch {epoch + 1} | LR: {optimizer.param_groups[0]['lr']:.8f}")
        logger.info(f"[Train Loss: {epoch_train_loss:.5f}] "
              f"Val => Loss: {val_metrics['loss']:.5f}, MAE: {val_metrics['mae']:.5f}, "
              f"RMSE: {val_metrics['rmse']:.5f}, R2: {val_metrics['r2']:.5f}")