# metta/rl/loss/matched_capacity.py
"""
Matched-Capacity Ablation Loss.

Uses the same dual-encoder architecture as GC-CRL (SAEncoder + GEncoder)
but trains with a random-target regression objective instead of contrastive learning.
This isolates whether GC-CRL's effects come from:
1. The additional model capacity (dual encoders)
2. The contrastive objective specifically
3. The goal-conditioning mechanism

If this matched-capacity baseline behaves like GC-CRL, the effect is capacity.
If it behaves like PPO baseline, the effect is the contrastive objective.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.loss.goal_conditioned_crl import GEncoder, SAEncoder
from metta.rl.loss.loss import Loss
from metta.rl.training import ComponentContext, TrainingEnvironment


class MatchedCapacityLoss(Loss):
    """Matched-capacity control: same architecture as GC-CRL, random-target objective.

    Uses SAEncoder and GEncoder (identical to GC-CRL) but trains them to
    predict fixed random projections of the observations. This provides
    the same gradient flow and parameter count without contrastive structure.
    """

    __slots__ = (
        "sa_encoder",
        "g_encoder",
        "random_projection",
        "hidden_dim",
        "embed_dim",
        "capacity_coef",
        "_initialized",
    )

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

        self.hidden_dim = self.cfg.hidden_dim
        self.embed_dim = self.cfg.embed_dim
        self.capacity_coef = self.cfg.capacity_coef

        self.sa_encoder: Optional[SAEncoder] = None
        self.g_encoder: Optional[GEncoder] = None
        self.random_projection: Optional[nn.Linear] = None
        self._initialized = False

    def _initialize(self, state_dim: int, action_dim: int) -> None:
        """Initialize encoders with same architecture as GC-CRL."""
        self.sa_encoder = SAEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim,
        ).to(self.device)

        self.g_encoder = GEncoder(
            goal_dim=state_dim,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim,
        ).to(self.device)

        # Fixed random projection target (not trained)
        self.random_projection = nn.Linear(state_dim, self.embed_dim, bias=False).to(self.device)
        self.random_projection.requires_grad_(False)

        # Register encoder parameters with optimizer (same as GC-CRL)
        context = self._require_context()
        encoder_params = list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters())
        context.optimizer.add_param_group({"params": encoder_params})

        self._initialized = True

    def get_experience_spec(self) -> Composite:
        return Composite()

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"encoded_obs", "values"}

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """Train dual encoders to predict random projections."""
        policy_td = shared_loss_data["policy_td"]
        minibatch = shared_loss_data["sampled_mb"]

        obs = minibatch.get("obs")
        if obs is None:
            obs = minibatch.get("observation")
        if obs is None:
            self.loss_tracker["matched_capacity_loss"].append(0.0)
            return torch.tensor(0.0, device=self.device), shared_loss_data, False

        actions = minibatch.get("actions")
        if actions is None:
            actions = minibatch.get("action")
        if actions is None:
            self.loss_tracker["matched_capacity_loss"].append(0.0)
            return torch.tensor(0.0, device=self.device), shared_loss_data, False

        # Flatten
        obs_flat = obs.reshape(-1, *obs.shape[2:]) if obs.dim() > 2 else obs
        if obs_flat.dim() > 2:
            obs_flat = obs_flat.reshape(obs_flat.shape[0], -1).float()
        else:
            obs_flat = obs_flat.float()

        if actions.dtype in (torch.long, torch.int, torch.int64, torch.int32):
            action_space_size = self.env.policy_env_info.action_space.n
            actions_flat = F.one_hot(actions.reshape(-1).long(), action_space_size).float()
        else:
            actions_flat = actions.reshape(-1, *actions.shape[2:]) if actions.dim() > 2 else actions
            actions_flat = actions_flat.float()

        if not self._initialized:
            self._initialize(obs_flat.shape[-1], actions_flat.shape[-1])

        # Forward through both encoders (same architecture as GC-CRL)
        sa_repr = self.sa_encoder(obs_flat, actions_flat)
        g_repr = self.g_encoder(obs_flat)

        # Train to predict random projection (regression target, not contrastive)
        with torch.no_grad():
            target = self.random_projection(obs_flat)

        sa_loss = F.mse_loss(sa_repr, target)
        g_loss = F.mse_loss(g_repr, target)
        total_loss = self.capacity_coef * (sa_loss + g_loss)

        self.loss_tracker["matched_capacity_loss"].append(float(total_loss.item()))
        self.loss_tracker["sa_mse"].append(float(sa_loss.item()))
        self.loss_tracker["g_mse"].append(float(g_loss.item()))

        return total_loss, shared_loss_data, False
