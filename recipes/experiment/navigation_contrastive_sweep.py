"""
Navigation environment experiments with GC-CRL for hyperparameter sweep.

This file creates Navigation experiments with contrastive learning / GC-CRL
to understand why GC-CRL underperforms on simple tasks (Q2 from paper).

Usage with wandb sweep:
    wandb sweep recipes/experiment/navigation_sweep.yaml
    wandb agent <sweep_id>

Or run individual configs:
    uv run ./tools/run.py train navigation_contrastive_sweep run=nav_gccrl_sweep
"""

from typing import Optional

import mettagrid.builder.envs as eb
from metta.rl.loss.contrastive_config import ContrastiveConfig
from metta.rl.loss.goal_conditioned_crl_config import GoalConditionedCRLConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig


def make_navigation_env(num_agents: int = 12) -> MettaGridConfig:
    """Create Navigation environment configuration."""
    env = eb.make_navigation(num_agents=num_agents)
    env.game.max_steps = 500
    # Add rewards
    env.game.agent.rewards.inventory["heart"] = 1
    env.game.agent.rewards.inventory_max["heart"] = 100
    return env


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    """Create evaluation environments."""
    basic_env = env or make_navigation_env()
    return [
        SimulationConfig(suite="contrastive_paper", name="navigation", env=basic_env),
    ]


# ============================================================================
# Baseline: PPO without contrastive loss
# ============================================================================

def train_baseline_ppo() -> TrainTool:
    """Baseline PPO for Navigation."""
    env = make_navigation_env()

    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),  # 100M timesteps
        losses=LossesConfig(
            contrastive=ContrastiveConfig(enabled=False),
            goal_conditioned_crl=GoalConditionedCRLConfig(enabled=False),
        ),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(),
        evaluator=EvaluatorConfig(simulations=simulations(env), epoch_interval=20),
    )


# ============================================================================
# PPO + Auxiliary Contrastive
# ============================================================================

def train_ppo_contrastive(
    temperature: float = 0.19,
    contrastive_coef: float = 0.00068,
    embedding_dim: int = 128,
    discount: float = 0.977,
    use_projection_head: bool = True,
) -> TrainTool:
    """PPO + Auxiliary Contrastive for Navigation with configurable hyperparameters."""
    env = make_navigation_env()

    contrastive_config = ContrastiveConfig(
        enabled=True,
        temperature=temperature,
        contrastive_coef=contrastive_coef,
        discount=discount,
        embedding_dim=embedding_dim,
        use_projection_head=use_projection_head,
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
        training_env=TrainingEnvironmentConfig(),
        evaluator=EvaluatorConfig(simulations=simulations(env), epoch_interval=20),
    )


# ============================================================================
# GC-CRL (Goal-Conditioned Contrastive RL)
# ============================================================================

def train_gccrl(
    embedding_dim: int = 64,
    gc_crl_coef: float = 1.0,
) -> TrainTool:
    """GC-CRL for Navigation with configurable hyperparameters."""
    env = make_navigation_env()

    gccrl_config = GoalConditionedCRLConfig(
        enabled=True,
        embedding_dim=embedding_dim,
        gc_crl_coef=gc_crl_coef,
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(enabled=False),
            goal_conditioned_crl=gccrl_config,
        ),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(),
        evaluator=EvaluatorConfig(simulations=simulations(env), epoch_interval=20),
    )


# ============================================================================
# Sweep configurations for wandb
# ============================================================================

def train_sweep(
    # Sweep parameters (will be set by wandb)
    method: str = "contrastive",  # "baseline", "contrastive", or "gccrl"
    temperature: float = 0.19,
    contrastive_coef: float = 0.00068,
    embedding_dim: int = 128,
    discount: float = 0.977,
    use_projection_head: bool = True,
    gc_crl_coef: float = 1.0,
) -> TrainTool:
    """
    Unified sweep function for wandb.

    This function is called by wandb sweep with different hyperparameter combinations.
    """
    if method == "baseline":
        return train_baseline_ppo()
    elif method == "contrastive":
        return train_ppo_contrastive(
            temperature=temperature,
            contrastive_coef=contrastive_coef,
            embedding_dim=embedding_dim,
            discount=discount,
            use_projection_head=use_projection_head,
        )
    elif method == "gccrl":
        return train_gccrl(
            embedding_dim=embedding_dim,
            gc_crl_coef=gc_crl_coef,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Pre-defined sweep configurations
# ============================================================================

SWEEP_CONFIGS = {
    # Contrastive coefficient sweep
    "contrastive_coef_0.0001": lambda: train_ppo_contrastive(contrastive_coef=0.0001),
    "contrastive_coef_0.0005": lambda: train_ppo_contrastive(contrastive_coef=0.0005),
    "contrastive_coef_0.001": lambda: train_ppo_contrastive(contrastive_coef=0.001),
    "contrastive_coef_0.005": lambda: train_ppo_contrastive(contrastive_coef=0.005),
    "contrastive_coef_0.01": lambda: train_ppo_contrastive(contrastive_coef=0.01),

    # Temperature sweep
    "temp_0.05": lambda: train_ppo_contrastive(temperature=0.05),
    "temp_0.1": lambda: train_ppo_contrastive(temperature=0.1),
    "temp_0.2": lambda: train_ppo_contrastive(temperature=0.2),
    "temp_0.5": lambda: train_ppo_contrastive(temperature=0.5),

    # Embedding dimension sweep
    "embed_32": lambda: train_ppo_contrastive(embedding_dim=32),
    "embed_64": lambda: train_ppo_contrastive(embedding_dim=64),
    "embed_128": lambda: train_ppo_contrastive(embedding_dim=128),
    "embed_256": lambda: train_ppo_contrastive(embedding_dim=256),

    # GC-CRL sweep
    "gccrl_embed_32": lambda: train_gccrl(embedding_dim=32),
    "gccrl_embed_64": lambda: train_gccrl(embedding_dim=64),
    "gccrl_embed_128": lambda: train_gccrl(embedding_dim=128),
    "gccrl_coef_0.1": lambda: train_gccrl(gc_crl_coef=0.1),
    "gccrl_coef_0.5": lambda: train_gccrl(gc_crl_coef=0.5),
    "gccrl_coef_2.0": lambda: train_gccrl(gc_crl_coef=2.0),
}


def train(config_name: str = "contrastive_coef_0.001") -> TrainTool:
    """Run a specific sweep configuration by name."""
    if config_name not in SWEEP_CONFIGS:
        available = ", ".join(SWEEP_CONFIGS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available}")
    return SWEEP_CONFIGS[config_name]()
