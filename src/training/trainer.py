"""
Core training functionality for GNN models.

This module contains the main training loop and related utilities.
"""

import gc
import time
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config.constants import GRADIENT_CLIP_MAX_NORM, DEFAULT_EVIDENTIAL_LAMBDA
from utils import get_layer_wise_learning_rates, is_main_process, safe_get_rank
from utils.logging import get_logger
from models import WeightedL1Loss, WeightedMSELoss, EvidentialLoss, WeightedEvidentialLoss
from .evaluator import evaluate

logger = get_logger(__name__)


def _update_learning_rate(
    scheduler: Any | None,
    val_loss: float,
    scheduler_name: str
) -> None:
    """Step the learning rate scheduler based on validation loss."""
    if scheduler is None:
        return
    if scheduler_name == 'reduce_on_plateau':
        scheduler.step(val_loss)
    else:
        scheduler.step()


def _check_early_stopping(
    val_loss: float,
    best_val_loss: float,
    patience_counter: int,
    patience: int,
    min_delta: float = 0.0
) -> tuple[float, int, bool, bool]:
    """
    Check early stopping condition.

    Returns: (new_best_loss, new_patience_counter, is_best, should_stop)
    """
    is_best = val_loss < best_val_loss - min_delta
    if is_best:
        return val_loss, 0, True, False
    else:
        new_counter = patience_counter + 1
        should_stop = new_counter >= patience
        return best_val_loss, new_counter, False, should_stop


def _broadcast_early_stopping_decision(
    should_stop: bool,
    is_distributed: bool,
    device: torch.device
) -> bool:
    """Broadcast early stopping decision across DDP ranks."""
    if not is_distributed:
        return should_stop

    stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
    dist.broadcast(stop_tensor, src=0)
    return stop_tensor.item() == 1


def _setup_loss_function(
    current_args: Any,
    task_type: str,
    multitask_weights: list[float] | None
) -> nn.Module:
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
        lambda_reg = getattr(current_args, 'evidential_lambda', DEFAULT_EVIDENTIAL_LAMBDA)
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


def _setup_scheduler(
    optimizer: torch.optim.Optimizer,
    current_args: Any
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Setup the learning rate scheduler based on configuration."""
    if current_args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=current_args.lr_reduce_factor,
            patience=int(current_args.lr_patience),
        )
    elif current_args.lr_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=current_args.lr_cosine_t_max,
            eta_min=0,
        )
    elif current_args.lr_scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=current_args.lr_step_size,
            gamma=current_args.lr_step_gamma,
        )
    elif current_args.lr_scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=current_args.lr_exp_gamma,
        )
    else:
        scheduler = None
    
    return scheduler


def _maybe_set_epoch(loader: DataLoader, epoch: int) -> None:
    """Set epoch for DistributedSampler if present."""
    if hasattr(loader, "sampler") and isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
        loader.sampler.set_epoch(epoch)


def _training_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    num_epochs: int
) -> float:
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
        chiral_signs = batch.chiral_signs.to(device) if batch.chiral_signs.numel() > 0 else None
        chiral_is_virtual_lp = batch.chiral_is_virtual_lp.to(device) if batch.chiral_is_virtual_lp.numel() > 0 else None
        allene_centers = batch.allene_centers.to(device) if batch.allene_centers.numel() > 0 else None
        allene_subs = batch.allene_subs.to(device) if batch.allene_subs.numel() > 0 else None

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
                    trans_indices,
                    chiral_signs,
                    chiral_is_virtual_lp,
                    allene_centers,
                    allene_subs
                )

                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
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
                trans_indices,
                chiral_signs,
                chiral_is_virtual_lp,
                allene_centers,
                allene_subs
            )
            if torch.isnan(outputs).any():
                logger.warning("NaN found in outputs!")
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
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
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    early_stopping: bool = False,
    task_type: str = 'regression',
    mixed_precision: bool = False,
    num_tasks: int = 1,
    multitask_weights: list[float] | None = None,
    preprocessing_pipeline: Any | None = None,
    is_ddp: bool = False,
    current_args: Any | None = None
) -> nn.Module:
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

def _save_best_model_state(
    model: nn.Module,
    val_metrics: dict[str, Any],
    epoch: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Save the best model state and metrics.

    Returns: (best_model_state, best_metrics)
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        best_model_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
    else:
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    best_metrics = {
        'best_epoch': epoch + 1,
        'best_val_loss': val_metrics['loss'],
        **{f"best_val_{k}": v for k, v in val_metrics.items()},
    }

    return best_model_state, best_metrics


def _log_wandb_metrics(
    epoch: int,
    epoch_train_loss: float,
    epoch_duration: float,
    val_metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    patience_counter: int,
    best_val_loss: float,
    task_type: str
) -> None:
    """Log metrics to wandb."""
    wandb_dict = {
        "epoch": epoch + 1,
        "train_loss": epoch_train_loss,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "val_loss": val_metrics['loss'],
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


def _broadcast_training_state(
    stop_training: bool,
    best_val_loss: float,
    patience_counter: int,
    best_epoch: int,
    is_ddp: bool,
    device: torch.device
) -> tuple[bool, float, int, int]:
    """
    Broadcast training state across DDP ranks.

    Returns: (stop_training, best_val_loss, patience_counter, best_epoch)
    """
    if not (is_ddp and dist.is_initialized()):
        return stop_training, best_val_loss, patience_counter, best_epoch

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

    # Return updated values
    return (
        stop_tensor.item() == 1,
        best_loss_tensor.item(),
        patience_tensor.item(),
        best_epoch_tensor.item()
    )


def _handle_epoch_end_fixed(
    model: nn.Module,
    val_metrics: dict[str, Any],
    best_val_loss: float,
    patience_counter: int,
    best_model_state: dict[str, torch.Tensor] | None,
    best_metrics: dict[str, Any] | None,
    epoch: int,
    early_stopping: bool,
    patience: int,
    current_args: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_train_loss: float,
    epoch_duration: float,
    task_type: str,
    is_ddp: bool
) -> tuple[bool, float, int, dict[str, torch.Tensor] | None, dict[str, Any] | None, int]:
    """
    Handle end-of-epoch processing with proper return values.

    Returns:
        Tuple of (stop_training, best_val_loss, patience_counter,
                 best_model_state, best_metrics, best_epoch)
    """
    stop_training = False
    current_val_loss = val_metrics['loss']
    best_epoch = epoch + 1  # Default to current epoch

    # Only main process handles early stopping logic
    if (not dist.is_initialized()) or (safe_get_rank() == 0):
        # Check early stopping condition using helper
        new_best_loss, new_patience_counter, is_best, _ = _check_early_stopping(
            val_loss=current_val_loss,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            patience=patience
        )

        if is_best:
            best_val_loss = new_best_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state, best_metrics = _save_best_model_state(model, val_metrics, epoch)

            if is_main_process():
                logger.info(f"New best model at epoch {best_epoch}: val_loss = {best_val_loss:.6f}")
        else:
            patience_counter = new_patience_counter
            if is_main_process():
                logger.info(f"No improvement for {patience_counter}/{patience} epochs")

        # Log current epoch metrics
        _print_epoch_progress(epoch, val_metrics, epoch_train_loss, optimizer, task_type)

        # Log to wandb if enabled
        if current_args.enable_wandb:
            _log_wandb_metrics(
                epoch=epoch,
                epoch_train_loss=epoch_train_loss,
                epoch_duration=epoch_duration,
                val_metrics=val_metrics,
                optimizer=optimizer,
                patience_counter=patience_counter,
                best_val_loss=best_val_loss,
                task_type=task_type
            )

        # Check if we should stop training
        if early_stopping and patience_counter >= patience:
            if is_main_process():
                logger.info(f"Early stopping triggered! No improvement for {patience} epochs.")
                logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
            stop_training = True

    # Broadcast training state to all processes (for DDP)
    stop_training, best_val_loss, patience_counter, best_epoch = _broadcast_training_state(
        stop_training=stop_training,
        best_val_loss=best_val_loss,
        patience_counter=patience_counter,
        best_epoch=best_epoch,
        is_ddp=is_ddp,
        device=device
    )

    return (stop_training, best_val_loss, patience_counter,
            best_model_state, best_metrics, best_epoch)


def _print_epoch_progress(
    epoch: int,
    val_metrics: dict[str, Any],
    epoch_train_loss: float,
    optimizer: torch.optim.Optimizer,
    task_type: str
) -> None:
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