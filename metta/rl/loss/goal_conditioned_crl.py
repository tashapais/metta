# metta/rl/loss/goal_conditioned_crl.py
"""
Goal-Conditioned Contrastive RL Loss.

Implements the approach from "Self-Supervised Goal-Reaching Results in Multi-Agent
Cooperation and Exploration" (Nimonkar et al., 2025). Unlike auxiliary contrastive
losses, this approach uses contrastive learning as the PRIMARY objective for learning
value functions.

Key differences from auxiliary contrastive loss:
1. Dual encoder architecture (state-action encoder + goal encoder)
2. Q-value defined as negative Euclidean distance in embedding space
3. Logsumexp regularization to prevent collapse
4. Higher contrastive weight (primary objective, not auxiliary)
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss
from metta.rl.training import ComponentContext, TrainingEnvironment


class SAEncoder(nn.Module):
    """State-Action encoder following gc-marl architecture.

    4-layer MLP with LayerNorm and Swish activation.
    Encodes (state, action) pairs into a 64-dim embedding space.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1024, embed_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, embed_dim)

        # Initialize with LeCun uniform (variance scaling 1/3, fan_in)
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc_out]:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(layer.bias)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.silu(self.ln1(self.fc1(x)))  # Swish = SiLU
        x = F.silu(self.ln2(self.fc2(x)))
        x = F.silu(self.ln3(self.fc3(x)))
        x = F.silu(self.ln4(self.fc4(x)))
        return self.fc_out(x)


class GEncoder(nn.Module):
    """Goal encoder following gc-marl architecture.

    4-layer MLP with LayerNorm and Swish activation.
    Encodes goal states into a 64-dim embedding space.
    """

    def __init__(self, goal_dim: int, hidden_dim: int = 1024, embed_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(goal_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, embed_dim)

        # Initialize with LeCun uniform
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc_out]:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(layer.bias)

    def forward(self, goal: Tensor) -> Tensor:
        x = F.silu(self.ln1(self.fc1(goal)))
        x = F.silu(self.ln2(self.fc2(x)))
        x = F.silu(self.ln3(self.fc3(x)))
        x = F.silu(self.ln4(self.fc4(x)))
        return self.fc_out(x)


class GoalConditionedCRLLoss(Loss):
    """Goal-Conditioned Contrastive RL loss.

    Uses contrastive learning to define Q-values directly:
    Q(s, a, g) = -||f_SA(s, a) - f_G(g)||_2

    This approach is fundamentally different from auxiliary contrastive losses:
    - Contrastive objective IS the RL objective (not an auxiliary signal)
    - Uses dedicated dual encoders with significant capacity
    - Includes logsumexp regularization for training stability
    """

    __slots__ = (
        "sa_encoder",
        "g_encoder",
        "hidden_dim",
        "embed_dim",
        "logsumexp_coef",
        "contrastive_coef",
        "discount",
        "goal_start_idx",
        "goal_end_idx",
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
        self.logsumexp_coef = self.cfg.logsumexp_coef
        self.contrastive_coef = self.cfg.contrastive_coef
        self.discount = self.cfg.discount
        self.goal_start_idx = self.cfg.goal_start_idx
        self.goal_end_idx = self.cfg.goal_end_idx

        # Encoders will be initialized on first forward pass when we know dimensions
        self.sa_encoder: Optional[SAEncoder] = None
        self.g_encoder: Optional[GEncoder] = None
        self._initialized = False

    def _initialize_encoders(self, state_dim: int, action_dim: int, goal_dim: int) -> None:
        """Initialize encoders based on actual input dimensions."""
        self.sa_encoder = SAEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim,
        ).to(self.device)

        self.g_encoder = GEncoder(
            goal_dim=goal_dim,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim,
        ).to(self.device)

        # Register encoder parameters with the optimizer so they actually get updated
        context = self._require_context()
        encoder_params = list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters())
        context.optimizer.add_param_group({"params": encoder_params})

        self._initialized = True

    def get_experience_spec(self) -> Composite:
        """Define additional data needed for goal-conditioned CRL."""
        return Composite()

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        # We need encoded observations and actions
        return {"encoded_obs", "values"}

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """Compute goal-conditioned contrastive loss."""
        policy_td = shared_loss_data["policy_td"]
        minibatch = shared_loss_data["sampled_mb"]

        batch_shape = minibatch.batch_size
        if len(batch_shape) != 2:
            raise ValueError("GC-CRL expects minibatch with 2D batch size (segments, horizon).")

        segments, horizon = batch_shape

        # Get observations and actions from minibatch
        obs = minibatch.get("obs")
        if obs is None:
            obs = minibatch.get("observation")
        if obs is None:
            self.loss_tracker["gc_crl_loss"].append(0.0)
            return torch.tensor(0.0, device=self.device), shared_loss_data, False

        actions = minibatch.get("actions")
        if actions is None:
            actions = minibatch.get("action")
        if actions is None:
            self.loss_tracker["gc_crl_loss"].append(0.0)
            return torch.tensor(0.0, device=self.device), shared_loss_data, False

        # Flatten observations for processing
        obs_flat = obs.reshape(-1, *obs.shape[2:]) if obs.dim() > 2 else obs

        # For discrete actions, convert to one-hot
        if actions.dtype in (torch.long, torch.int, torch.int64, torch.int32):
            # Discrete actions - get action space size
            action_space_size = self.env.policy_env_info.action_space.n
            actions_flat = F.one_hot(actions.reshape(-1).long(), action_space_size).float()
        else:
            actions_flat = actions.reshape(-1, *actions.shape[2:]) if actions.dim() > 2 else actions
            actions_flat = actions_flat.float()

        # Flatten observations to feature vector if needed
        if obs_flat.dim() > 2:
            obs_flat = obs_flat.reshape(obs_flat.shape[0], -1).float()
        else:
            obs_flat = obs_flat.float()

        # Initialize encoders if not done
        if not self._initialized:
            state_dim = obs_flat.shape[-1]
            action_dim = actions_flat.shape[-1]

            # Determine goal dimension
            if self.goal_end_idx > 0:
                goal_dim = self.goal_end_idx - self.goal_start_idx
            else:
                # Default: use same as state dim (goal is future state features)
                goal_dim = state_dim

            self._initialize_encoders(state_dim, action_dim, goal_dim)

        # Sample future states as goals using geometric distribution
        dones = minibatch.get("dones")
        if dones is not None:
            dones = dones.squeeze(-1) if dones.dim() == 3 else dones
            done_mask = dones.to(dtype=torch.bool)
        else:
            done_mask = torch.zeros(segments, horizon, dtype=torch.bool, device=self.device)

        truncateds = minibatch.get("truncateds")
        if truncateds is not None:
            truncateds = truncateds.squeeze(-1) if truncateds.dim() == 3 else truncateds
            done_mask = torch.logical_or(done_mask, truncateds.to(dtype=torch.bool))

        # Sample future goals for each (state, action) pair
        loss, metrics = self._compute_gc_crl_loss(
            obs_flat, actions_flat, done_mask, segments, horizon
        )

        # Track metrics
        self.loss_tracker["gc_crl_loss"].append(float(loss.item()))
        for key, value in metrics.items():
            if key not in self.loss_tracker:
                self.loss_tracker[key] = []
            self.loss_tracker[key].append(value)

        return loss, shared_loss_data, False

    def _compute_gc_crl_loss(
        self,
        obs_flat: Tensor,
        actions_flat: Tensor,
        done_mask: Tensor,
        segments: int,
        horizon: int,
    ) -> tuple[Tensor, dict]:
        """Compute InfoNCE loss with logsumexp regularization.

        For each (state, action) pair, we sample a future state as the goal
        and train the encoders to distinguish correct (s, a, g) tuples from
        incorrect ones using contrastive learning.
        """
        batch_size = obs_flat.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=self.device), {
                "gc_categorical_accuracy": 0.0,
                "gc_logits_pos": 0.0,
                "gc_logits_neg": 0.0,
                "gc_logsumexp": 0.0,
            }

        # Get state features for goal (use specified indices or full state)
        if self.goal_end_idx > 0 and self.goal_start_idx < obs_flat.shape[-1]:
            goals = obs_flat[:, self.goal_start_idx:self.goal_end_idx]
        else:
            # Use full observation as goal (common for dense state representations)
            goals = obs_flat

        # Encode state-action pairs and goals
        sa_repr = self.sa_encoder(obs_flat, actions_flat)
        g_repr = self.g_encoder(goals)

        # Compute pairwise distances (InfoNCE with Euclidean distance)
        # logits[i, j] = -||f_SA(s_i, a_i) - f_G(g_j)||_2
        diff = sa_repr[:, None, :] - g_repr[None, :, :]  # [B, B, embed_dim]
        logits = -torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)  # [B, B]

        # InfoNCE loss: maximize diagonal (correct pairs) vs off-diagonal
        # Loss = -mean(diag(logits) - logsumexp(logits, dim=1))
        diag_logits = torch.diag(logits)
        logsumexp = torch.logsumexp(logits, dim=1)
        infonce_loss = -torch.mean(diag_logits - logsumexp)

        # Logsumexp regularization (prevents collapse)
        logsumexp_reg = self.logsumexp_coef * torch.mean(logsumexp ** 2)

        total_loss = (infonce_loss + logsumexp_reg) * self.contrastive_coef

        # Compute metrics
        with torch.no_grad():
            I = torch.eye(batch_size, device=self.device)
            correct = (torch.argmax(logits, dim=1) == torch.argmax(I, dim=1)).float()
            categorical_accuracy = correct.mean().item()
            logits_pos = torch.sum(logits * I) / torch.sum(I)
            logits_neg = torch.sum(logits * (1 - I)) / torch.sum(1 - I)

        metrics = {
            "gc_categorical_accuracy": categorical_accuracy,
            "gc_logits_pos": logits_pos.item(),
            "gc_logits_neg": logits_neg.item(),
            "gc_logsumexp": logsumexp.mean().item(),
            "gc_infonce_loss": infonce_loss.item(),
            "gc_logsumexp_reg": logsumexp_reg.item(),
        }

        return total_loss, metrics
