# src/data/preprocessing.py

"""
Data preprocessing module for molecular property prediction.
Handles SAE normalization and standard scaling with proper train/test isolation.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union
import tqdm
from dataclasses import dataclass

# Import from your existing modules
from datasets import partial_parse_atomic_numbers, compute_sae_dict_from_atomic_numbers_list


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    apply_sae: bool = False
    sae_subtasks: Optional[List[int]] = None
    apply_standard_scaling: bool = True
    task_type: str = "regression"  # "regression" or "multitask"
    sae_percentile_cutoff: float = 2.0


class SAENormalizer:
    """
    Size-Extensive Additive (SAE) normalization with strict train/test isolation.
    Computes atomic contributions from training data only, applies to all splits.
    """
    
    def __init__(self, task_type: str = "regression", percentile_cutoff: float = 2.0):
        self.task_type = task_type
        self.percentile_cutoff = percentile_cutoff
        self.sae_statistics = None
        self.is_fitted = False
    
    def fit(self, 
            train_smiles: List[str], 
            train_targets: Union[List[float], List[List[float]]], 
            subtasks: Optional[List[int]] = None) -> Dict:
        """
        Compute SAE statistics from TRAINING DATA ONLY.
        
        Args:
            train_smiles: Training SMILES strings
            train_targets: Training target values
            subtasks: For multitask, which subtasks to apply SAE to
            
        Returns:
            Dictionary of SAE statistics
        """
        print(f"[SAE] Computing statistics from {len(train_smiles)} training molecules only")
        
        if self.task_type == "regression":
            self.sae_statistics = self._fit_single_task(train_smiles, train_targets)
        elif self.task_type == "multitask":
            if subtasks is None:
                raise ValueError("Must specify subtasks for multitask SAE normalization")
            self.sae_statistics = self._fit_multitask(train_smiles, train_targets, subtasks)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
        
        self.is_fitted = True
        return self.sae_statistics
    
    def _fit_single_task(self, smiles_list: List[str], targets: List[float]) -> Dict:
        """Fit SAE for single-task regression."""
        print("[SAE] Single-task regression mode")
        
        # Parse atomic numbers for training data only
        train_nums = []
        good_targets = []
        
        for smi, tgt in tqdm.tqdm(zip(smiles_list, targets), 
                                  total=len(smiles_list), 
                                  desc="Parsing molecules for SAE"):
            nums = partial_parse_atomic_numbers(smi)
            if nums is not None:
                train_nums.append(nums)
                good_targets.append(tgt)
        
        if len(train_nums) == 0:
            raise ValueError("No valid molecules found for SAE computation")
        
        print(f"[SAE] Using {len(train_nums)}/{len(smiles_list)} valid molecules")
        
        # Compute SAE dictionary
        sae_dict = compute_sae_dict_from_atomic_numbers_list(
            train_nums, good_targets, percentile_cutoff=self.percentile_cutoff
        )
        
        return {"regression": sae_dict}
    
    def _fit_multitask(self, 
                      smiles_list: List[str], 
                      targets: List[List[float]], 
                      subtasks: List[int]) -> Dict:
        """Fit SAE for multitask regression."""
        print(f"[SAE] Multitask mode for subtasks: {subtasks}")
        
        # Parse atomic numbers for training set only
        train_atomic_nums = []
        for smi in tqdm.tqdm(smiles_list, desc="Parsing molecules for SAE"):
            nums = partial_parse_atomic_numbers(smi)
            train_atomic_nums.append(nums)
        
        train_array = np.array(targets, dtype=np.float64)
        sae_statistics = {}
        
        # Compute SAE for each subtask
        for subtask_idx in subtasks:
            if subtask_idx >= train_array.shape[1]:
                raise ValueError(f"Subtask index {subtask_idx} >= number of targets {train_array.shape[1]}")
            
            print(f"[SAE] Computing for subtask {subtask_idx}")
            
            # Collect data for current subtask (training only)
            subtask_targets = []
            subtask_nums = []
            
            for nums, target_vals in zip(train_atomic_nums, train_array):
                if nums is not None:
                    subtask_targets.append(target_vals[subtask_idx])
                    subtask_nums.append(nums)
            
            if len(subtask_nums) == 0:
                print(f"[SAE] WARNING: No valid molecules for subtask {subtask_idx}")
                continue
            
            # Compute SAE dictionary for this subtask
            sae_dict = compute_sae_dict_from_atomic_numbers_list(
                subtask_nums, subtask_targets, percentile_cutoff=self.percentile_cutoff
            )
            
            sae_statistics[subtask_idx] = sae_dict
            print(f"[SAE] Subtask {subtask_idx}: {len(sae_dict)} atomic contributions computed")
        
        return sae_statistics
    
    def transform(self, 
                  smiles_list: List[str], 
                  targets: Union[List[float], List[List[float]]]) -> Union[List[float], List[List[float]]]:
        """
        Apply SAE normalization using pre-computed statistics.
        
        Args:
            smiles_list: SMILES strings to transform
            targets: Target values to normalize
            
        Returns:
            SAE-normalized targets
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")
        
        if self.task_type == "regression":
            return self._transform_single_task(smiles_list, targets)
        elif self.task_type == "multitask":
            return self._transform_multitask(smiles_list, targets)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
    
    def _transform_single_task(self, smiles_list: List[str], targets: List[float]) -> List[float]:
        """Apply SAE normalization for single-task."""
        sae_dict = self.sae_statistics["regression"]
        normalized_targets = []
        
        for smi, target in zip(smiles_list, targets):
            nums = partial_parse_atomic_numbers(smi)
            shift = 0.0
            
            if nums is not None:
                shift = sum(sae_dict.get(n, 0.0) for n in nums)
            
            normalized_targets.append(target - shift)
        
        return normalized_targets
    
    def _transform_multitask(self, 
                           smiles_list: List[str], 
                           targets: List[List[float]]) -> List[List[float]]:
        """Apply SAE normalization for multitask."""
        target_array = np.array(targets, dtype=np.float64)
        
        # Parse atomic numbers for this dataset
        atomic_nums = []
        for smi in smiles_list:
            nums = partial_parse_atomic_numbers(smi)
            atomic_nums.append(nums)
        
        # Apply normalization for each subtask
        for subtask_idx, sae_dict in self.sae_statistics.items():
            if subtask_idx >= target_array.shape[1]:
                continue
                
            for i, nums in enumerate(atomic_nums):
                if nums is not None:
                    shift = sum(sae_dict.get(n, 0.0) for n in nums)
                    target_array[i, subtask_idx] -= shift
        
        return target_array.tolist()
    
    def inverse_transform(self, 
                         smiles_list: List[str], 
                         predictions: Union[np.ndarray, List[float], List[List[float]]]) -> np.ndarray:
        """
        Apply inverse SAE transformation to predictions.
        
        This adds back the atomic contributions that were subtracted during training.
        
        Args:
            smiles_list: SMILES strings corresponding to predictions
            predictions: Model predictions on SAE-normalized scale
            
        Returns:
            Predictions converted back to original scale
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before inverse_transform()")
        
        # Convert predictions to numpy array
        if isinstance(predictions, list):
            pred_array = np.array(predictions, dtype=np.float64)
        else:
            pred_array = np.array(predictions, dtype=np.float64)
        
        # Ensure proper shape
        if len(pred_array.shape) == 1:
            pred_array = pred_array.reshape(-1, 1)
        
        # Parse atomic numbers for all SMILES
        atomic_nums = []
        for smi in smiles_list:
            nums = partial_parse_atomic_numbers(smi)
            atomic_nums.append(nums)
        
        # Apply inverse transformation
        if self.task_type == "regression":
            return self._inverse_transform_single_task(atomic_nums, pred_array)
        elif self.task_type == "multitask":
            return self._inverse_transform_multitask(atomic_nums, pred_array)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
    
    def _inverse_transform_single_task(self, atomic_nums: List, pred_array: np.ndarray) -> np.ndarray:
        """Apply inverse SAE transformation for single-task."""
        sae_dict = self.sae_statistics["regression"]
        result = pred_array.copy()
        
        for i, nums in enumerate(atomic_nums):
            if nums is not None:
                shift = sum(sae_dict.get(n, 0.0) for n in nums)
                result[i] += shift
        
        return result
    
    def _inverse_transform_multitask(self, atomic_nums: List, pred_array: np.ndarray) -> np.ndarray:
        """Apply inverse SAE transformation for multitask."""
        result = pred_array.copy()
        
        # Apply inverse transformation for each subtask
        for subtask_idx, sae_dict in self.sae_statistics.items():
            if subtask_idx >= result.shape[1]:
                continue
                
            for i, nums in enumerate(atomic_nums):
                if nums is not None:
                    shift = sum(sae_dict.get(n, 0.0) for n in nums)
                    result[i, subtask_idx] += shift
        
        return result
    
    def fit_transform(self, 
                     train_smiles: List[str], 
                     train_targets: Union[List[float], List[List[float]]], 
                     subtasks: Optional[List[int]] = None) -> Union[List[float], List[List[float]]]:
        """Fit SAE on training data and transform it."""
        self.fit(train_smiles, train_targets, subtasks)
        return self.transform(train_smiles, train_targets)

class StandardScaler:
    """
    Standard scaling with strict train/test isolation.
    Computes mean/std from training data only, applies to all splits.
    """
    
    def __init__(self):
        self.means = None
        self.stds = None
        self.is_fitted = False
    
    def fit(self, train_targets: Union[List[float], List[List[float]]]) -> None:
        """
        Compute scaling statistics from TRAINING DATA ONLY.
        
        Args:
            train_targets: Training target values (post-SAE if applicable)
        """
        print(f"[Scaling] Computing statistics from {len(train_targets)} training samples")
        
        # Convert to numpy array
        target_array = np.array(train_targets, dtype=np.float32)
        if len(target_array.shape) == 1:
            target_array = target_array.reshape(-1, 1)
        
        # Compute statistics
        self.means = target_array.mean(axis=0)
        self.stds = target_array.std(axis=0, ddof=1)
        
        # Avoid division by zero
        self.stds[self.stds < 1e-12] = 1.0
        
        self.is_fitted = True
        
        print(f"[Scaling] Means: {self.means}")
        print(f"[Scaling] Stds: {self.stds}")
    
    def transform(self, targets: Union[List[float], List[List[float]]]) -> np.ndarray:
        """Apply scaling using pre-computed statistics."""
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")
        
        target_array = np.array(targets, dtype=np.float32)
        if len(target_array.shape) == 1:
            target_array = target_array.reshape(-1, 1)
        
        return (target_array - self.means) / self.stds
    
    def inverse_transform(self, scaled_targets: np.ndarray) -> np.ndarray:
        """Reverse the scaling transformation."""
        if not self.is_fitted:
            raise ValueError("Must call fit() before inverse_transform()")
        
        return scaled_targets * self.stds + self.means
    
    def fit_transform(self, train_targets: Union[List[float], List[List[float]]]) -> np.ndarray:
        """Fit scaler on training data and transform it."""
        self.fit(train_targets)
        return self.transform(train_targets)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline with proper train/test isolation.
    Ensures SAE normalization happens before standard scaling.
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.sae_normalizer = None
        self.standard_scaler = None
        self.is_fitted = False
    
    def fit(self, 
            train_smiles: List[str], 
            train_targets: Union[List[float], List[List[float]]]) -> None:
        """
        Fit preprocessing pipeline on TRAINING DATA ONLY.
        
        Args:
            train_smiles: Training SMILES strings
            train_targets: Training target values
        """
        print(f"[Pipeline] Fitting preprocessing on {len(train_smiles)} training samples")
        
        current_targets = train_targets
        
        # Step 1: SAE normalization (if enabled)
        if self.config.apply_sae:
            print("[Pipeline] Step 1: SAE normalization")
            self.sae_normalizer = SAENormalizer(
                task_type=self.config.task_type,
                percentile_cutoff=self.config.sae_percentile_cutoff
            )
            current_targets = self.sae_normalizer.fit_transform(
                train_smiles, current_targets, self.config.sae_subtasks
            )
        
        # Step 2: Standard scaling (if enabled)
        if self.config.apply_standard_scaling:
            print("[Pipeline] Step 2: Standard scaling")
            self.standard_scaler = StandardScaler()
            self.standard_scaler.fit(current_targets)
        
        self.is_fitted = True
        print("[Pipeline] Fitting complete")
    
    def transform(self, 
                  smiles_list: List[str], 
                  targets: Union[List[float], List[List[float]]], 
                  return_numpy: bool = True) -> Union[List, np.ndarray]:
        """
        Apply preprocessing transformations using pre-computed statistics.
        
        Args:
            smiles_list: SMILES strings
            targets: Target values to transform
            return_numpy: Whether to return numpy array or list
            
        Returns:
            Transformed targets
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")
        
        current_targets = targets
        
        # Step 1: SAE normalization (if fitted)
        if self.sae_normalizer is not None:
            current_targets = self.sae_normalizer.transform(smiles_list, current_targets)
        
        # Step 2: Standard scaling (if fitted)
        if self.standard_scaler is not None:
            scaled_targets = self.standard_scaler.transform(current_targets)
            if return_numpy:
                return scaled_targets
            else:
                # Convert back to list format
                if scaled_targets.shape[1] == 1:
                    return [float(x[0]) for x in scaled_targets]
                else:
                    return scaled_targets.tolist()
        
        # No scaling applied
        if return_numpy:
            target_array = np.array(current_targets, dtype=np.float32)
            if len(target_array.shape) == 1:
                target_array = target_array.reshape(-1, 1)
            return target_array
        else:
            return current_targets
    
    def inverse_transform(self, 
                     smiles_list: List[str], 
                     transformed_targets: np.ndarray) -> np.ndarray:
        """
        Apply inverse transformations to convert predictions back to original scale.
        
        Order: Standard Scaling (inverse) → SAE (inverse)
        
        Args:
            smiles_list: SMILES strings corresponding to predictions
            transformed_targets: Predictions from model (on transformed scale)
            
        Returns:
            Predictions on original scale
        """
        current_predictions = transformed_targets.copy()
        
        # Step 1: Inverse standard scaling (if applied)
        if self.standard_scaler is not None and self.standard_scaler.is_fitted:
            current_predictions = self.standard_scaler.inverse_transform(current_predictions)
        
        # Step 2: Inverse SAE transformation (if applied)
        if self.sae_normalizer is not None and self.sae_normalizer.is_fitted:
            current_predictions = self.sae_normalizer.inverse_transform(smiles_list, current_predictions)
        
        return current_predictions
    
    def fit_transform(self, 
                     train_smiles: List[str], 
                     train_targets: Union[List[float], List[List[float]]], 
                     return_numpy: bool = True) -> Union[List, np.ndarray]:
        """Fit pipeline on training data and transform it."""
        self.fit(train_smiles, train_targets)
        return self.transform(train_smiles, train_targets, return_numpy)
    
    def get_num_tasks(self, targets: Union[List[float], List[List[float]]]) -> int:
        """Get number of tasks from target structure."""
        if isinstance(targets[0], list):
            return len(targets[0])
        else:
            return 1


# Convenience function for the main script
def preprocess_molecular_data(
    train_smiles: List[str],
    train_targets: Union[List[float], List[List[float]]],
    val_smiles: List[str],
    val_targets: Union[List[float], List[List[float]]],
    test_smiles: List[str],
    test_targets: Union[List[float], List[List[float]]],
    config: PreprocessingConfig
) -> Tuple[Tuple[List, List], Tuple[List, List], Tuple[List, List], PreprocessingPipeline]:
    """
    Complete preprocessing of molecular data with proper train/test isolation.
    
    Returns:
        Tuple of ((train_targets, train_smiles), (val_targets, val_smiles), 
                 (test_targets, test_smiles), pipeline)
    """
    print("="*60)
    print("MOLECULAR DATA PREPROCESSING")
    print("="*60)
    
    # Create and fit pipeline on training data only
    pipeline = PreprocessingPipeline(config)
    pipeline.fit(train_smiles, train_targets)
    
    # Transform all datasets using training-derived statistics
    processed_train = pipeline.transform(train_smiles, train_targets, return_numpy=False)
    processed_val = pipeline.transform(val_smiles, val_targets, return_numpy=False)
    processed_test = pipeline.transform(test_smiles, test_targets, return_numpy=False)
    
    print(f"✓ Processed {len(train_smiles)} train, {len(val_smiles)} val, {len(test_smiles)} test samples")
    print("="*60)
    
    return (
        (processed_train, train_smiles),
        (processed_val, val_smiles), 
        (processed_test, test_smiles),
        pipeline
    )