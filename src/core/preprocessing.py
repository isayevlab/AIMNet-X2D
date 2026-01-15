"""
GPU-native preprocessing for molecular property prediction.

Key optimization: All transforms operate on batched tensors,
eliminating per-molecule Python loops.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np
from sklearn.linear_model import LinearRegression

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SAETransform:
    """
    GPU-native Self Atomic Energy (SAE) normalization.

    SAE removes molecular size dependence from extensive properties
    by subtracting sum of atomic contributions.

    Key optimization: Uses GPU lookup table instead of Python dict iteration.
    """

    sae_dict: dict[int, float]
    subtasks: list[int]
    max_atomic_num: int = 119

    _sae_lookup: torch.Tensor | None = field(default=None, repr=False)

    @classmethod
    def fit(
        cls,
        atomic_numbers_list: list[list[int]],
        targets: np.ndarray,
        subtasks: list[int],
    ) -> SAETransform:
        """Fit SAE coefficients using linear regression."""
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        unique_atoms = set()
        for nums in atomic_numbers_list:
            unique_atoms.update(nums)
        unique_atoms = sorted(unique_atoms)

        atom_to_idx = {a: i for i, a in enumerate(unique_atoms)}
        count_matrix = np.zeros((len(atomic_numbers_list), len(unique_atoms)))

        for mol_idx, nums in enumerate(atomic_numbers_list):
            for atom_num in nums:
                count_matrix[mol_idx, atom_to_idx[atom_num]] += 1

        sae_dict: dict[int, float] = {}

        for subtask_idx in subtasks:
            y = targets[:, subtask_idx]
            reg = LinearRegression(fit_intercept=False)
            reg.fit(count_matrix, y)

            for atom_num, coef_idx in atom_to_idx.items():
                if atom_num not in sae_dict:
                    sae_dict[atom_num] = 0.0
                sae_dict[atom_num] += reg.coef_[coef_idx] / len(subtasks)

        logger.info(f"Fitted SAE for {len(sae_dict)} elements on subtasks {subtasks}")
        return cls(sae_dict=sae_dict, subtasks=subtasks)

    def _get_sae_lookup(self, device: torch.device) -> torch.Tensor:
        """Get or create GPU lookup table."""
        if self._sae_lookup is None or self._sae_lookup.device != device:
            lookup = torch.zeros(self.max_atomic_num, dtype=torch.float32)
            for atom_num, sae_val in self.sae_dict.items():
                if 0 <= atom_num < self.max_atomic_num:
                    lookup[atom_num] = sae_val
            self._sae_lookup = lookup.to(device)
        return self._sae_lookup

    def transform_batch(
        self,
        atomic_numbers: torch.Tensor,
        atom_counts: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SAE normalization to batch (GPU-native)."""
        device = targets.device
        batch_size, max_atoms = atomic_numbers.shape

        sae_lookup = self._get_sae_lookup(device)

        atom_indices = torch.arange(max_atoms, device=device).unsqueeze(0)
        valid_mask = atom_indices < atom_counts.unsqueeze(1)

        sae_values = sae_lookup[atomic_numbers.clamp(0, self.max_atomic_num - 1)]
        sae_values = sae_values * valid_mask.float()

        sae_shifts = sae_values.sum(dim=1, keepdim=True)

        result = targets.clone()
        for subtask_idx in self.subtasks:
            if subtask_idx < result.shape[1]:
                result[:, subtask_idx] = result[:, subtask_idx] - sae_shifts.squeeze(1)

        return result

    def inverse_transform_batch(
        self,
        atomic_numbers: torch.Tensor,
        atom_counts: torch.Tensor,
        normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse SAE transformation (GPU-native)."""
        device = normalized.device
        batch_size, max_atoms = atomic_numbers.shape

        sae_lookup = self._get_sae_lookup(device)

        atom_indices = torch.arange(max_atoms, device=device).unsqueeze(0)
        valid_mask = atom_indices < atom_counts.unsqueeze(1)

        sae_values = sae_lookup[atomic_numbers.clamp(0, self.max_atomic_num - 1)]
        sae_values = sae_values * valid_mask.float()
        sae_shifts = sae_values.sum(dim=1, keepdim=True)

        result = normalized.clone()
        for subtask_idx in self.subtasks:
            if subtask_idx < result.shape[1]:
                result[:, subtask_idx] = result[:, subtask_idx] + sae_shifts.squeeze(1)

        return result

    def state_dict(self) -> dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "sae_dict": self.sae_dict,
            "subtasks": self.subtasks,
            "max_atomic_num": self.max_atomic_num,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> SAETransform:
        """Deserialize from checkpoint."""
        return cls(
            sae_dict=state["sae_dict"],
            subtasks=state["subtasks"],
            max_atomic_num=state.get("max_atomic_num", 119),
        )


@dataclass
class StandardScaler:
    """GPU-native standard scaling (zero mean, unit variance)."""

    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-8

    @classmethod
    def fit(cls, targets: torch.Tensor, eps: float = 1e-8) -> StandardScaler:
        """Fit scaler to targets."""
        mean = targets.mean(dim=0)
        std = targets.std(dim=0, unbiased=False)
        std = torch.clamp(std, min=eps)
        return cls(mean=mean, std=std, eps=eps)

    def transform_batch(self, targets: torch.Tensor) -> torch.Tensor:
        """Apply scaling."""
        mean = self.mean.to(targets.device)
        std = self.std.to(targets.device)
        return (targets - mean) / std

    def inverse_transform_batch(self, normalized: torch.Tensor) -> torch.Tensor:
        """Inverse scaling."""
        mean = self.mean.to(normalized.device)
        std = self.std.to(normalized.device)
        return normalized * std + mean

    def state_dict(self) -> dict[str, Any]:
        """Serialize."""
        return {
            "mean": self.mean.cpu().numpy().tolist(),
            "std": self.std.cpu().numpy().tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> StandardScaler:
        """Deserialize."""
        return cls(
            mean=torch.tensor(state["mean"]),
            std=torch.tensor(state["std"]),
            eps=state.get("eps", 1e-8),
        )


@dataclass
class PreprocessingPipeline:
    """Combined preprocessing pipeline: SAE -> Scaling."""

    sae_transform: SAETransform | None = None
    scaler: StandardScaler | None = None

    @classmethod
    def fit(
        cls,
        atomic_numbers_list: list[list[int]],
        targets: np.ndarray,
        apply_sae: bool = False,
        sae_subtasks: list[int] | None = None,
        apply_scaling: bool = True,
    ) -> PreprocessingPipeline:
        """Fit preprocessing pipeline."""
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        targets_tensor = torch.from_numpy(targets).float()

        sae_transform = None
        if apply_sae and sae_subtasks:
            sae_transform = SAETransform.fit(
                atomic_numbers_list, targets, sae_subtasks
            )
            max_atoms = max(len(nums) for nums in atomic_numbers_list)
            atomic_nums_padded = torch.zeros(len(atomic_numbers_list), max_atoms, dtype=torch.int64)
            atom_counts = torch.zeros(len(atomic_numbers_list), dtype=torch.int64)

            for i, nums in enumerate(atomic_numbers_list):
                atomic_nums_padded[i, :len(nums)] = torch.tensor(nums)
                atom_counts[i] = len(nums)

            targets_tensor = sae_transform.transform_batch(
                atomic_nums_padded, atom_counts, targets_tensor
            )

        scaler = None
        if apply_scaling:
            scaler = StandardScaler.fit(targets_tensor)

        return cls(sae_transform=sae_transform, scaler=scaler)

    def transform_batch(
        self,
        atomic_numbers: torch.Tensor,
        atom_counts: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply full pipeline."""
        result = targets

        if self.sae_transform is not None:
            result = self.sae_transform.transform_batch(
                atomic_numbers, atom_counts, result
            )

        if self.scaler is not None:
            result = self.scaler.transform_batch(result)

        return result

    def inverse_transform_batch(
        self,
        atomic_numbers: torch.Tensor,
        atom_counts: torch.Tensor,
        normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse full pipeline (reverse order)."""
        result = normalized

        if self.scaler is not None:
            result = self.scaler.inverse_transform_batch(result)

        if self.sae_transform is not None:
            result = self.sae_transform.inverse_transform_batch(
                atomic_numbers, atom_counts, result
            )

        return result

    def state_dict(self) -> dict[str, Any]:
        """Serialize."""
        return {
            "sae_transform": self.sae_transform.state_dict() if self.sae_transform else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> PreprocessingPipeline:
        """Deserialize."""
        return cls(
            sae_transform=SAETransform.from_state_dict(state["sae_transform"]) if state.get("sae_transform") else None,
            scaler=StandardScaler.from_state_dict(state["scaler"]) if state.get("scaler") else None,
        )
