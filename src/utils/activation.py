"""
Activation function utilities.
"""

import torch.nn as nn


def get_activation_function(activation_type: str) -> nn.Module:
    """
    Returns the appropriate activation function based on the provided type.
    
    Args:
        activation_type: A string specifying the activation function type. 
                        Should be one of "relu", "leakyrelu", "elu", "gelu", or "silu".
    
    Returns:
        An activation function from torch.nn
        
    Raises:
        ValueError: If activation_type is not supported
    """
    activation_map = {
        "relu": nn.ReLU(),
        "leakyrelu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU()
    }
    
    if activation_type not in activation_map:
        supported = ", ".join(activation_map.keys())
        raise ValueError(f"Invalid activation type: {activation_type}. Supported: {supported}")
    
    return activation_map[activation_type]


def get_activation_by_name(name: str) -> nn.Module:
    """
    Alias for get_activation_function for backwards compatibility.
    
    Args:
        name: Name of the activation function
        
    Returns:
        Activation function module
    """
    return get_activation_function(name)