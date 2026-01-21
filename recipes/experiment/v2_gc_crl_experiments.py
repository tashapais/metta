"""
Multi-Environment Contrastive Learning Experiments.

This recipe provides comprehensive experiments for the contrastive RL paper,
testing across multiple environments and comparing:
1. PPO Baseline (no contrastive loss)
2. PPO + Auxiliary Contrastive Loss (current approach)
3. PPO + Goal-Conditioned CRL (gc-marl approach)

Environments:
- MettagGrid Arena (current)
- MettagGrid Navigation (simpler, goal-reaching focused)

For Tribal Village experiments, use the separate tribal_village_contrastive.py script.
"""

from typing import Optional

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.rl.loss.contrastive_config import ContrastiveConfig
from metta.rl.loss.goal_conditioned_crl_config import GoalConditionedCRLConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig


# =============================================================================
# Environment Configurations
# =============================================================================


def make_arena_env(num_agents: int = 24) -> MettaGridConfig:
    """Create Arena environment with shaped rewards."""
    env = eb.make_arena(num_agents=num_agents)
    env.game.agent.rewards.inventory["heart"] = 1
    env.game.agent.rewards.inventory_max["heart"] = 100
    env.game.agent.rewards.inventory.update(
        {
            "ore_red": 0.1,
            "battery_red": 0.8,
            "laser": 0.3,
            "armor": 0.3,
        }
    )
    return env


def make_navigation_env(num_agents: int = 12) -> MettaGridConfig:
    """Create simpler navigation environment focused on goal-reaching.

    This environment is more similar to MPE environments used in gc-marl,
    with simpler dynamics and clearer goal structure.
    Uses 12 agents (divisible by instance size of 6).
    """
    env = eb.make_arena(num_agents=num_agents)
    # Simpler rewards: just reaching resources
    env.game.agent.rewards.inventory["ore_red"] = 1.0
    env.game.agent.rewards.inventory_max["ore_red"] = 10
    # Disable complex crafting rewards
    env.game.agent.rewards.inventory["battery_red"] = 0.0
    env.game.agent.rewards.inventory["laser"] = 0.0
    env.game.agent.rewards.inventory["armor"] = 0.0
    env.game.agent.rewards.inventory["heart"] = 0.0
    return env


def arena_simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    """Evaluation simulations for Arena."""
    basic_env = env or make_arena_env()
    return [
        SimulationConfig(suite="contrastive_paper", name="arena_shaped", env=basic_env),
    ]


def navigation_simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    """Evaluation simulations for Navigation."""
    basic_env = env or make_navigation_env()
    return [
        SimulationConfig(suite="contrastive_paper", name="navigation", env=basic_env),
    ]


# =============================================================================
# Arena Experiments
# =============================================================================


def train_arena_baseline() -> TrainTool:
    """PPO baseline on Arena (no contrastive loss)."""
    env = make_arena_env()

    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(enabled=False),
            goal_conditioned_crl=GoalConditionedCRLConfig(enabled=False),
        ),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(curriculum=cc.env_curriculum(env)),
        evaluator=EvaluatorConfig(simulations=arena_simulations(env), epoch_interval=20),
    )


def train_arena_aux_contrastive() -> TrainTool:
    """PPO + Auxiliary Contrastive Loss on Arena (current paper approach)."""
    env = make_arena_env()

    contrastive_config = ContrastiveConfig(
        enabled=True,
        temperature=0.1902943104505539,
        contrastive_coef=0.0006806607125326991,
        discount=0.977,
        embedding_dim=128,
        use_projection_head=True,
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=contrastive_config,
            goal_conditioned_crl=GoalConditionedCRLConfig(enabled=False),
        ),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(curriculum=cc.env_curriculum(env)),
        evaluator=EvaluatorConfig(simulations=arena_simulations(env), epoch_interval=20),
    )


def train_arena_aux_contrastive_high_coef() -> TrainTool:
    """PPO + Auxiliary Contrastive with HIGHER coefficient (0.01 instead of 0.00068)."""
    env = make_arena_env()

    contrastive_config = ContrastiveConfig(
        enabled=True,
        temperature=0.19,
        contrastive_coef=0.01,  # ~15x higher than default
        discount=0.977,
        embedding_dim=128,
        use_projection_head=True,
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=contrastive_config,
            goal_conditioned_crl=GoalConditionedCRLConfig(enabled=False),
        ),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(curriculum=cc.env_curriculum(env)),
        evaluator=EvaluatorConfig(simulations=arena_simulations(env), epoch_interval=20),
    )


def train_arena_gc_crl() -> TrainTool:
    """PPO + Goal-Conditioned CRL on Arena (gc-marl approach)."""
    env = make_arena_env()

    gc_crl_config = GoalConditionedCRLConfig(
        enabled=True,
        hidden_dim=1024,
        embed_dim=64,
        contrastive_coef=0.1,
        logsumexp_coef=0.1,
        discount=0.99,
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(enabled=False),
            goal_conditioned_crl=gc_crl_config,
        ),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(curriculum=cc.env_curriculum(env)),
        evaluator=EvaluatorConfig(simulations=arena_simulations(env), epoch_interval=20),
    )


# =============================================================================
# Navigation Experiments (Simpler, goal-reaching focused)
# =============================================================================


def train_navigation_baseline() -> TrainTool:
    """PPO baseline on Navigation environment."""
    env = make_navigation_env()

    trainer_config = TrainerConfig(
        total_timesteps=int(5e7),  # Shorter, simpler task
        losses=LossesConfig(
            contrastive=ContrastiveConfig(enabled=False),
            goal_conditioned_crl=GoalConditionedCRLConfig(enabled=False),
        ),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(curriculum=cc.env_curriculum(env)),
        evaluator=EvaluatorConfig(simulations=navigation_simulations(env), epoch_interval=20),
    )


def train_navigation_aux_contrastive() -> TrainTool:
    """PPO + Auxiliary Contrastive on Navigation."""
    env = make_navigation_env()

    contrastive_config = ContrastiveConfig(
        enabled=True,
        temperature=0.19,
        contrastive_coef=0.01,  # Higher coefficient for simpler task
        discount=0.977,
        embedding_dim=128,
        use_projection_head=True,
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(5e7),
        losses=LossesConfig(
            contrastive=contrastive_config,
            goal_conditioned_crl=GoalConditionedCRLConfig(enabled=False),
        ),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(curriculum=cc.env_curriculum(env)),
        evaluator=EvaluatorConfig(simulations=navigation_simulations(env), epoch_interval=20),
    )


def train_navigation_gc_crl() -> TrainTool:
    """PPO + Goal-Conditioned CRL on Navigation (should show significant improvements)."""
    env = make_navigation_env()

    gc_crl_config = GoalConditionedCRLConfig(
        enabled=True,
        hidden_dim=1024,
        embed_dim=64,
        contrastive_coef=0.1,
        logsumexp_coef=0.1,
        discount=0.99,
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(5e7),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(enabled=False),
            goal_conditioned_crl=gc_crl_config,
        ),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(curriculum=cc.env_curriculum(env)),
        evaluator=EvaluatorConfig(simulations=navigation_simulations(env), epoch_interval=20),
    )


# =============================================================================
# Experiment Registry
# =============================================================================

EXPERIMENTS = {
    # Arena experiments
    "arena_baseline": train_arena_baseline,
    "arena_aux_contrastive": train_arena_aux_contrastive,
    "arena_aux_contrastive_high": train_arena_aux_contrastive_high_coef,
    "arena_gc_crl": train_arena_gc_crl,
    # Navigation experiments
    "navigation_baseline": train_navigation_baseline,
    "navigation_aux_contrastive": train_navigation_aux_contrastive,
    "navigation_gc_crl": train_navigation_gc_crl,
}


def train(experiment: str = "arena_gc_crl") -> TrainTool:
    """Run a specific experiment by name.

    Args:
        experiment: One of the experiment names from EXPERIMENTS dict

    Available experiments:
        Arena (complex multi-step crafting):
        - arena_baseline: PPO without contrastive loss
        - arena_aux_contrastive: PPO + auxiliary contrastive (current paper)
        - arena_aux_contrastive_high: PPO + auxiliary contrastive (15x higher coef)
        - arena_gc_crl: PPO + goal-conditioned CRL (gc-marl approach)

        Navigation (simpler goal-reaching):
        - navigation_baseline: PPO without contrastive loss
        - navigation_aux_contrastive: PPO + auxiliary contrastive
        - navigation_gc_crl: PPO + goal-conditioned CRL (expected to show biggest gains)
    """
    if experiment not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment: {experiment}. Available: {available}")

    return EXPERIMENTS[experiment]()
