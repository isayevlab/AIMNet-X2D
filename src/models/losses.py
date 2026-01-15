"""
Loss functions for molecular property prediction.

This module contains loss functions used for training,
including weighted losses for multi-task learning and evidential loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss for multi-task learning.
    
    Applies different weights to different tasks in multi-task regression.
    
    Args:
        weights: Tensor of weights for each task
    """
    
    def __init__(self, weights: torch.Tensor):
        super(WeightedL1Loss, self).__init__()
        self.register_buffer('weights', weights)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted L1 loss.
        
        Args:
            y_pred: Predicted values of shape [batch_size, num_tasks]
            y_true: True values of shape [batch_size, num_tasks]
            
        Returns:
            Scalar loss value
        """
        # Compute absolute error for each task
        error = torch.abs(y_pred - y_true)
        
        # Apply task-specific weights
        weighted_error = error * self.weights.to(y_pred.device)
        
        # Sum across tasks, then average across samples
        loss_per_sample = weighted_error.sum(dim=1)
        loss = loss_per_sample.mean()
        
        return loss


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss for multi-task learning.
    
    Applies different weights to different tasks in multi-task regression.
    
    Args:
        weights: Tensor of weights for each task
    """
    
    def __init__(self, weights: torch.Tensor):
        super(WeightedMSELoss, self).__init__()
        self.register_buffer('weights', weights)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            y_pred: Predicted values of shape [batch_size, num_tasks]
            y_true: True values of shape [batch_size, num_tasks]
            
        Returns:
            Scalar loss value
        """
        # Compute squared error for each task
        error = (y_pred - y_true) ** 2
        
        # Apply task-specific weights
        weighted_error = error * self.weights.to(y_pred.device)
        
        # Sum across tasks, then average across samples
        loss_per_sample = weighted_error.sum(dim=1)
        loss = loss_per_sample.mean()
        
        return loss


class EvidentialLoss(nn.Module):
    """
    Evidential Loss for uncertainty estimation in regression.
    
    Based on "Evidential Deep Learning to Quantify Classification Uncertainty"
    but adapted for regression tasks. The model outputs evidence parameters
    that define a Normal-Inverse-Gamma distribution.
    
    Args:
        lambda_reg: Regularization strength for evidence
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, lambda_reg: float = 1.0, reduction: str = 'mean'):
        super(EvidentialLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute evidential loss for regression.
        
        For evidential regression, the model should output 4 values per target:
        - gamma (location parameter)
        - nu (degrees of freedom, >1)  
        - alpha (concentration, >1)
        - beta (rate parameter, >0)
        
        Args:
            outputs: Model outputs of shape [batch_size, num_tasks * 4] for evidential
                    or [batch_size, num_tasks] for standard regression
            targets: True values of shape [batch_size, num_tasks]
            
        Returns:
            Evidential loss value
        """
        batch_size, target_dim = targets.shape
        
        # Check if outputs are evidential (4 params per task) or standard
        if outputs.shape[1] == target_dim * 4:
            # Evidential outputs: reshape to [batch_size, num_tasks, 4]
            outputs = outputs.view(batch_size, target_dim, 4)
            
            # Extract evidential parameters
            gamma = outputs[:, :, 0]  # predicted mean
            nu = outputs[:, :, 1]     # degrees of freedom
            alpha = outputs[:, :, 2]  # concentration  
            beta = outputs[:, :, 3]   # rate parameter
            
            # Apply constraints to ensure valid parameters
            nu = F.softplus(nu) + 1.0      # nu > 1
            alpha = F.softplus(alpha) + 1.0 # alpha > 1  
            beta = F.softplus(beta)         # beta > 0
            
            # Compute evidential loss components
            # NLL loss (negative log-likelihood)
            diff = (targets - gamma)
            nll_loss = 0.5 * torch.log(torch.pi / nu) \
                     - alpha * torch.log(2 * beta) \
                     + torch.lgamma(alpha) \
                     - torch.lgamma(alpha + 0.5) \
                     + (alpha + 0.5) * torch.log(beta + nu * diff**2 / 2)
            
            # Regularization term to prevent overconfident predictions
            reg_loss = self.lambda_reg * (2 * beta + alpha)
            
            total_loss = nll_loss + reg_loss
            
        else:
            # Standard regression outputs: use MSE loss
            total_loss = (outputs - targets) ** 2
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


class WeightedEvidentialLoss(nn.Module):
    """
    Weighted Evidential Loss for multi-task learning.
    
    Args:
        weights: Tensor of weights for each task
        lambda_reg: Regularization strength for evidence
        reduction: Reduction method
    """
    
    def __init__(self, weights: torch.Tensor, lambda_reg: float = 1.0, reduction: str = 'mean'):
        super(WeightedEvidentialLoss, self).__init__()
        self.register_buffer('weights', weights)
        self.lambda_reg = lambda_reg
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted evidential loss.
        
        Args:
            outputs: Model outputs (evidential format expected)
            targets: True values of shape [batch_size, num_tasks]
            
        Returns:
            Weighted evidential loss value
        """
        batch_size, target_dim = targets.shape
        
        # Reshape outputs to [batch_size, num_tasks, 4] for evidential
        if outputs.shape[1] == target_dim * 4:
            outputs = outputs.view(batch_size, target_dim, 4)
            
            # Extract evidential parameters
            gamma = outputs[:, :, 0]
            nu = F.softplus(outputs[:, :, 1]) + 1.0
            alpha = F.softplus(outputs[:, :, 2]) + 1.0  
            beta = F.softplus(outputs[:, :, 3])
            
            # Compute evidential loss per task
            diff = (targets - gamma)
            nll_loss = 0.5 * torch.log(torch.pi / nu) \
                     - alpha * torch.log(2 * beta) \
                     + torch.lgamma(alpha) \
                     - torch.lgamma(alpha + 0.5) \
                     + (alpha + 0.5) * torch.log(beta + nu * diff**2 / 2)
            
            reg_loss = self.lambda_reg * (2 * beta + alpha)
            task_losses = nll_loss + reg_loss
            
            # Apply task-specific weights
            weighted_losses = task_losses * self.weights.to(outputs.device)
            
        else:
            # Fallback to weighted MSE
            error = (outputs - targets) ** 2
            weighted_losses = error * self.weights.to(outputs.device)
        
        # Sum across tasks, then reduce across samples
        loss_per_sample = weighted_losses.sum(dim=1)
        
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample


def create_loss_function(loss_type: str, 
                        task_type: str = 'regression',
                        multitask_weights: torch.Tensor | None = None,
                        **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('l1', 'mse', 'evidential')
        task_type: Type of task ('regression', 'multitask')
        multitask_weights: Weights for multi-task learning
        **kwargs: Additional arguments for specific loss functions
        
    Returns:
        Initialized loss function
        
    Raises:
        ValueError: If loss_type is not supported
    """
    if loss_type == 'l1':
        if task_type == 'multitask' and multitask_weights is not None:
            return WeightedL1Loss(multitask_weights)
        else:
            return nn.L1Loss()
    
    elif loss_type == 'mse':
        if task_type == 'multitask' and multitask_weights is not None:
            return WeightedMSELoss(multitask_weights)
        else:
            return nn.MSELoss()
    
    elif loss_type == 'evidential':
        lambda_reg = kwargs.get('lambda_reg', 1.0)
        if task_type == 'multitask' and multitask_weights is not None:
            return WeightedEvidentialLoss(multitask_weights, lambda_reg=lambda_reg)
        else:
            return EvidentialLoss(lambda_reg=lambda_reg)
    
    else:
        supported = ['l1', 'mse', 'evidential']
        raise ValueError(f"Unsupported loss type: {loss_type}. Supported: {supported}")