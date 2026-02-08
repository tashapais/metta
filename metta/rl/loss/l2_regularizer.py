# metta/rl/loss/l2_regularizer.py
"""
L2 Regularizer Loss.

Adds explicit L2 regularization on the shared encoder's activations,
matching the regularization effect of contrastive loss without the
temporal structure or InfoNCE objective.

This serves as a baseline to test whether the variance reduction from
contrastive learning is specific to InfoNCE or achievable with any regularizer.
"""

from typing import Any, Optional

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss
from metta.rl.training import ComponentContext, TrainingEnvironment


class L2RegularizerLoss(Loss):
    """L2 regularization on encoder activations."""

    _EMBEDDING_CANDIDATES = ("encoder_output", "encoded_obs", "core", "hidden_state", "features")

    __slots__ = ("l2_coef",)

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        cfg: Any,
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, cfg)
        self.l2_coef = self.cfg.l2_coef

    def get_experience_spec(self) -> Composite:
        return Composite()

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        if policy_td is None:
            return {"values"}
        policy_keys = policy_td.keys(True)
        for candidate in self._EMBEDDING_CANDIDATES:
            if candidate in policy_keys:
                return {candidate}
        return {"values"}

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """Compute L2 regularization on encoder activations."""
        policy_td = shared_loss_data["policy_td"]

        # Get embeddings from policy
        embeddings = self._get_embeddings(policy_td)

        # L2 norm of embeddings (activation regularization)
        l2_loss = self.l2_coef * torch.mean(embeddings ** 2)

        self.loss_tracker["l2_reg_loss"].append(float(l2_loss.item()))
        self.loss_tracker["embedding_norm_mean"].append(float(torch.norm(embeddings, dim=-1).mean().item()))

        return l2_loss, shared_loss_data, False

    def _get_embeddings(self, policy_td: TensorDict) -> Tensor:
        """Extract embeddings from policy output."""
        for candidate in self._EMBEDDING_CANDIDATES:
            if candidate in policy_td.keys(True):
                return policy_td[candidate]
        return policy_td["values"]
