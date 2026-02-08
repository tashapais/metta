# metta/rl/loss/matched_capacity_config.py
"""Configuration for matched-capacity ablation loss."""

from typing import Any

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.rl.loss.loss import LossConfig
from metta.rl.loss.matched_capacity import MatchedCapacityLoss
from metta.rl.training import TrainingEnvironment


class MatchedCapacityConfig(LossConfig):
    """Configuration for matched-capacity ablation.

    Uses identical architecture to GC-CRL (SAEncoder + GEncoder, 1024 hidden, 64 embed)
    but with a random-target regression objective instead of contrastive learning.
    Same coefficient as GC-CRL (0.1) for fair comparison.
    """

    hidden_dim: int = Field(default=1024, gt=0, description="Hidden dim (matches GC-CRL)")
    embed_dim: int = Field(default=64, gt=0, description="Embed dim (matches GC-CRL)")
    capacity_coef: float = Field(
        default=0.1,
        ge=0,
        description="Loss coefficient (matches GC-CRL contrastive_coef for fair comparison)",
    )

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "MatchedCapacityLoss":
        return MatchedCapacityLoss(policy, trainer_cfg, env, device, instance_name, self)
