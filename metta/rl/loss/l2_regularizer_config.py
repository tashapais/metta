# metta/rl/loss/l2_regularizer_config.py
"""Configuration for L2 regularizer loss."""

from typing import Any

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.rl.loss.l2_regularizer import L2RegularizerLoss
from metta.rl.loss.loss import LossConfig
from metta.rl.training import TrainingEnvironment


class L2RegularizerConfig(LossConfig):
    """Configuration for L2 activation regularizer.

    Regularizes encoder activations with L2 penalty.
    Coefficient should be comparable to contrastive_coef (~0.00068) for fair comparison.
    """

    l2_coef: float = Field(
        default=0.00068,
        ge=0,
        description="Coefficient for L2 regularization (matched to contrastive_coef for fair comparison)",
    )

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "L2RegularizerLoss":
        return L2RegularizerLoss(policy, trainer_cfg, env, device, instance_name, self)
