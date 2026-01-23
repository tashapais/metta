"""
Baseline implementations for comparison with our contrastive learning approach.

TODO: Implement these baselines for fair comparison in the paper.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RepresentationBaseline(ABC):
    """Abstract base class for representation learning baselines."""

    @abstractmethod
    def compute_loss(self, obs: Tensor, next_obs: Tensor, actions: Tensor) -> Tensor:
        """Compute the auxiliary representation learning loss."""
        pass


@dataclass
class CURLConfig:
    """Configuration for CURL baseline."""
    encoder_feature_dim: int = 128
    temperature: float = 0.1
    augmentation_strength: float = 0.5


class CURLBaseline(RepresentationBaseline):
    """
    CURL: Contrastive Unsupervised Representations for RL
    (Srinivas et al., 2020, arXiv:2004.04136)

    Key idea: Use data augmentation to create positive pairs, then apply
    contrastive loss. Augmentations include random crop, color jitter, etc.

    Difference from our approach:
    - CURL uses augmentation-based positives
    - We use temporal proximity-based positives
    - CURL is designed for pixel observations
    - We work with tokenized observations
    """

    def __init__(self, config: CURLConfig, encoder: nn.Module):
        self.config = config
        self.encoder = encoder
        # TODO: Implement momentum encoder for CURL
        self.target_encoder = None

    def compute_loss(self, obs: Tensor, next_obs: Tensor, actions: Tensor) -> Tensor:
        """
        TODO: Implement CURL loss.

        1. Apply random augmentations to obs -> obs_aug1, obs_aug2
        2. Encode both: z1 = encoder(obs_aug1), z2 = target_encoder(obs_aug2)
        3. Compute InfoNCE loss between z1 (query) and z2 (key)
        """
        raise NotImplementedError("CURL baseline not yet implemented")


@dataclass
class SPRConfig:
    """Configuration for SPR baseline."""
    prediction_depth: int = 5
    embedding_dim: int = 128
    projection_dim: int = 256


class SPRBaseline(RepresentationBaseline):
    """
    SPR: Self-Predictive Representations
    (Schwarzer et al., 2020, arXiv:2007.05929)

    Key idea: Predict future latent representations from current state and actions.
    Uses a contrastive loss to match predictions with actual future embeddings.

    Difference from our approach:
    - SPR explicitly predicts future states through a transition model
    - We sample geometric offsets without explicit prediction
    - SPR uses cosine similarity; we use dot product with temperature
    """

    def __init__(self, config: SPRConfig, encoder: nn.Module):
        self.config = config
        self.encoder = encoder
        self.transition_model = None  # TODO: Implement
        self.projection_head = None   # TODO: Implement

    def compute_loss(self, obs: Tensor, next_obs: Tensor, actions: Tensor) -> Tensor:
        """
        TODO: Implement SPR loss.

        1. Encode current state: z_t = encoder(obs)
        2. For k = 1 to K:
           - Predict future: z_hat_{t+k} = transition_model(z_t, actions[t:t+k])
           - Encode actual future: z_{t+k} = target_encoder(obs_{t+k})
           - Compute contrastive loss
        3. Average over all prediction depths
        """
        raise NotImplementedError("SPR baseline not yet implemented")


@dataclass
class ATCConfig:
    """Configuration for ATC baseline."""
    temporal_distance: int = 3
    embedding_dim: int = 128
    temperature: float = 0.1


class ATCBaseline(RepresentationBaseline):
    """
    ATC: Augmented Temporal Contrast
    (Stooke et al., 2020, arXiv:2009.08319)

    Key idea: Decouple representation learning from RL by using temporal
    contrastive learning on a separate representation network.

    Difference from our approach:
    - ATC uses fixed temporal offset
    - We use geometric distribution over offsets
    - ATC decouples representation from policy; we share the encoder
    """

    def __init__(self, config: ATCConfig, encoder: nn.Module):
        self.config = config
        self.encoder = encoder

    def compute_loss(self, obs: Tensor, next_obs: Tensor, actions: Tensor) -> Tensor:
        """
        TODO: Implement ATC loss.

        1. Sample pairs (s_t, s_{t+k}) with fixed k
        2. Apply augmentation to both
        3. Compute contrastive loss
        """
        raise NotImplementedError("ATC baseline not yet implemented")


@dataclass
class ICMConfig:
    """Configuration for ICM baseline."""
    feature_dim: int = 128
    forward_loss_coef: float = 0.2
    inverse_loss_coef: float = 0.8
    intrinsic_reward_scale: float = 0.01


class ICMBaseline(RepresentationBaseline):
    """
    ICM: Intrinsic Curiosity Module
    (Pathak et al., 2017, arXiv:1705.05363)

    Key idea: Learn representations by predicting:
    1. Forward model: Predict next state features given current state and action
    2. Inverse model: Predict action given current and next state features

    Intrinsic reward = prediction error of forward model

    Difference from our approach:
    - ICM provides intrinsic reward signal
    - We provide auxiliary representation loss without reward shaping
    - ICM uses prediction error; we use contrastive similarity
    """

    def __init__(self, config: ICMConfig, encoder: nn.Module):
        self.config = config
        self.encoder = encoder
        self.forward_model = None  # TODO: Implement
        self.inverse_model = None  # TODO: Implement

    def compute_loss(self, obs: Tensor, next_obs: Tensor, actions: Tensor) -> Tensor:
        """
        TODO: Implement ICM loss.

        1. Encode: phi_t = encoder(obs), phi_{t+1} = encoder(next_obs)
        2. Forward loss: ||forward_model(phi_t, a) - phi_{t+1}||^2
        3. Inverse loss: cross_entropy(inverse_model(phi_t, phi_{t+1}), a)
        4. Return weighted sum
        """
        raise NotImplementedError("ICM baseline not yet implemented")

    def compute_intrinsic_reward(self, obs: Tensor, next_obs: Tensor, actions: Tensor) -> Tensor:
        """Compute intrinsic curiosity reward."""
        raise NotImplementedError()


@dataclass
class RNDConfig:
    """Configuration for RND baseline."""
    feature_dim: int = 128
    intrinsic_reward_scale: float = 0.01
    update_proportion: float = 0.25


class RNDBaseline(RepresentationBaseline):
    """
    RND: Random Network Distillation
    (Burda et al., 2018, arXiv:1810.12894)

    Key idea: Train a predictor network to match a random fixed target network.
    Intrinsic reward = prediction error (high for novel states, low for familiar).

    Difference from our approach:
    - RND provides novelty-based intrinsic reward
    - We provide similarity-based representation loss
    - RND doesn't explicitly learn temporal structure
    """

    def __init__(self, config: RNDConfig, obs_dim: int):
        self.config = config
        # Target network is random and fixed
        self.target_network = self._create_network(obs_dim, config.feature_dim)
        for param in self.target_network.parameters():
            param.requires_grad = False
        # Predictor network is trained
        self.predictor_network = self._create_network(obs_dim, config.feature_dim)

    def _create_network(self, input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def compute_loss(self, obs: Tensor, next_obs: Tensor, actions: Tensor) -> Tensor:
        """
        TODO: Implement RND loss.

        loss = ||predictor(obs) - target(obs)||^2
        """
        raise NotImplementedError("RND baseline not yet implemented")

    def compute_intrinsic_reward(self, obs: Tensor) -> Tensor:
        """Compute intrinsic novelty reward."""
        raise NotImplementedError()


# Summary of baselines for paper
BASELINE_SUMMARY = """
Baseline Comparison for Contrastive Learning in Multi-Agent RL
==============================================================

| Baseline | Type          | Positive Pairs    | Key Difference from Ours |
|----------|---------------|-------------------|--------------------------|
| CURL     | Contrastive   | Augmentation      | Augmentation vs temporal |
| SPR      | Contrastive   | Predicted future  | Explicit transition model|
| ATC      | Contrastive   | Fixed temporal    | Fixed vs geometric offset|
| ICM      | Prediction    | Forward/inverse   | Intrinsic reward vs loss |
| RND      | Distillation  | Random target     | Novelty vs similarity    |
| Ours     | Contrastive   | Geometric temporal| Multi-agent negatives    |

TODO: Run all baselines on:
1. MettaGrid Arena (24 agents, complex crafting)
2. MettaGrid Navigation (12 agents, simple collection)
3. Craftax (single agent, procedural generation)
4. Tribal Village (1000 agents, team competition)
"""

if __name__ == "__main__":
    print(BASELINE_SUMMARY)
