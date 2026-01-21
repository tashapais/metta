# metta/rl/loss/goal_conditioned_crl_config.py
"""Configuration for Goal-Conditioned Contrastive RL loss."""

from typing import Any

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.rl.loss.goal_conditioned_crl import GoalConditionedCRLLoss
from metta.rl.loss.loss import LossConfig
from metta.rl.training import TrainingEnvironment


class GoalConditionedCRLConfig(LossConfig):
    """Configuration for goal-conditioned contrastive RL loss.

    This loss implements the approach from gc-marl where contrastive learning
    defines the Q-function directly: Q(s,a,g) = -||f_SA(s,a) - f_G(g)||_2

    Key hyperparameters following the reference implementation:
    - hidden_dim: 1024 (4-layer MLP)
    - embed_dim: 64 (embedding space dimension)
    - logsumexp_coef: 0.1 (regularization to prevent collapse)
    - contrastive_coef: 0.1 (higher than auxiliary approach)
    """

    # Encoder architecture
    hidden_dim: int = Field(
        default=1024,
        gt=0,
        description="Hidden dimension for SA and G encoders (gc-marl uses 1024)"
    )
    embed_dim: int = Field(
        default=64,
        gt=0,
        description="Embedding dimension (gc-marl uses 64)"
    )

    # Loss coefficients
    contrastive_coef: float = Field(
        default=0.1,
        ge=0,
        description="Coefficient for contrastive loss (much higher than auxiliary approach)"
    )
    logsumexp_coef: float = Field(
        default=0.1,
        ge=0,
        description="Logsumexp regularization coefficient (prevents collapse)"
    )

    # Temporal sampling
    discount: float = Field(
        default=0.99,
        ge=0,
        lt=1,
        description="Discount factor (gamma) for geometric goal sampling"
    )

    # Goal specification (for environments with explicit goal structure)
    goal_start_idx: int = Field(
        default=0,
        ge=0,
        description="Start index for goal features in observation (0 = use full obs)"
    )
    goal_end_idx: int = Field(
        default=0,
        ge=0,
        description="End index for goal features in observation (0 = use full obs)"
    )

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "GoalConditionedCRLLoss":
        """Create the goal-conditioned CRL loss instance."""
        return GoalConditionedCRLLoss(policy, trainer_cfg, env, device, instance_name, self)
