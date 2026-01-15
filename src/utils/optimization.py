"""
Optimization utilities for training.
"""

from typing import Any

import torch.nn as nn

from utils.logging import get_logger

logger = get_logger(__name__)


def get_layer_wise_learning_rates(model: nn.Module, base_lr: float, decay_factor: float = 0.8) -> list[dict[str, Any]]:
    """
    Apply lower learning rates to earlier layers in the model.
    
    This is useful for transfer learning where you want to update
    earlier layers more slowly than later layers.
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate
        decay_factor: Decay factor for layer-wise learning rate
                     (lower means more aggressive decay)
    
    Returns:
        List of parameter groups with different learning rates
    """
    parameter_groups = []
    
    # Get all named parameters
    named_params = list(model.named_parameters())
    
    # Group parameters by layer depth (based on number of dots in name)
    layers = {}
    for name, param in named_params:
        if param.requires_grad:
            depth = name.count('.')
            if depth not in layers:
                layers[depth] = []
            layers[depth].append((name, param))
    
    # Sort layers by depth (deeper layers come last)
    sorted_depths = sorted(layers.keys())
    
    # Calculate learning rate for each depth
    for i, depth in enumerate(sorted_depths):
        # Calculate decay based on relative position
        position_factor = i / max(1, len(sorted_depths) - 1)  # 0 for first, 1 for last
        # Apply decay - deeper layers get higher learning rates
        lr = base_lr * (decay_factor ** (1 - position_factor))
        
        params = [param for _, param in layers[depth]]
        parameter_groups.append({'params': params, 'lr': lr})
        
        # Log the learning rate for this depth
        layer_names = [name for name, _ in layers[depth]]
        logger.info(f"Depth {depth} (layers: {len(layer_names)}): LR = {lr:.8f}")
        if len(layer_names) <= 3:  # Show layer names if not too many
            logger.info(f"  Layers: {layer_names}")
        
    return parameter_groups


def freeze_parameters(model: nn.Module, freeze_patterns: list[str]) -> None:
    """
    Freeze model parameters matching given patterns.
    
    Args:
        model: PyTorch model
        freeze_patterns: List of string patterns to match parameter names
    """
    frozen_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        for pattern in freeze_patterns:
            if pattern in name:
                param.requires_grad = False
                frozen_count += 1
                break
    
    logger.info(f"Frozen {frozen_count}/{total_count} parameters")


def unfreeze_parameters(model: nn.Module, unfreeze_patterns: list[str]) -> None:
    """
    Unfreeze model parameters matching given patterns.
    
    Args:
        model: PyTorch model
        unfreeze_patterns: List of string patterns to match parameter names
    """
    unfrozen_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        for pattern in unfreeze_patterns:
            if pattern in name:
                param.requires_grad = True
                unfrozen_count += 1
                break
    
    logger.info(f"Unfrozen {unfrozen_count}/{total_count} parameters")


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def get_parameter_info(model: nn.Module) -> dict[str, Any]:
    """
    Get detailed information about model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter information
    """
    param_info = {
        'by_layer': {},
        'summary': count_parameters(model)
    }
    
    for name, param in model.named_parameters():
        param_info['by_layer'][name] = {
            'shape': list(param.shape),
            'numel': param.numel(),
            'requires_grad': param.requires_grad,
            'dtype': str(param.dtype)
        }
    
    return param_info