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

from src.utils.logging import get_logger

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
