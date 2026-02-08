"""
ICML-strength experiments for contrastive regularization paper.

Addresses reviewer concerns:
1. More seeds (10 seeds for key conditions)
2. Standard regularizer baselines (L2 activation regularization)
3. Matched-capacity ablation (same architecture as GC-CRL, random objective)
4. Intermediate therapeutic window values (0.003, 0.005)
5. Intermediate complexity (Arena with 6, 12 agents)

Total new runs: ~56 on Arena (100M steps each)
GPU budget: 2x A100 80GB
"""

from typing import Optional

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.rl.loss.contrastive_config import ContrastiveConfig
from metta.rl.loss.goal_conditioned_crl_config import GoalConditionedCRLConfig
from metta.rl.loss.l2_regularizer_config import L2RegularizerConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.loss.matched_capacity_config import MatchedCapacityConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig


# =============================================================================
# Environment helpers
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


def arena_simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    basic_env = env or make_arena_env()
    return [
        SimulationConfig(suite="icml_contrastive", name="arena_shaped", env=basic_env),
    ]


# =============================================================================
# 1. MORE SEEDS: 10-seed baseline and PPO+C on Arena-24
# =============================================================================

def train_baseline_10seed() -> TrainTool:
    """PPO baseline (no contrastive) - run 10 seeds for statistical power."""
    env = make_arena_env(num_agents=24)
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(enabled=False),
        ),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=[], epoch_interval=9999, parallel_evals=1),
    )


def train_ppoc_10seed() -> TrainTool:
    """PPO + Contrastive (default) - run 10 seeds for statistical power."""
    env = make_arena_env(num_agents=24)
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(
                enabled=True,
                temperature=0.1902943104505539,
                contrastive_coef=0.0006806607125326991,
                discount=0.977,
                embedding_dim=128,
                use_projection_head=True,
            ),
        ),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=[], epoch_interval=9999, parallel_evals=1),
    )


# =============================================================================
# 2. STANDARD REGULARIZER BASELINES
# =============================================================================

def train_l2_regularizer() -> TrainTool:
    """L2 activation regularizer - coefficient matched to contrastive (0.00068)."""
    env = make_arena_env(num_agents=24)
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(enabled=False),
            l2_regularizer=L2RegularizerConfig(
                enabled=True,
                l2_coef=0.00068,
            ),
        ),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=[], epoch_interval=9999, parallel_evals=1),
    )


def train_l2_regularizer_high() -> TrainTool:
    """L2 activation regularizer with higher coefficient (0.01) - matches high contrastive ablation."""
    env = make_arena_env(num_agents=24)
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(enabled=False),
            l2_regularizer=L2RegularizerConfig(
                enabled=True,
                l2_coef=0.01,
            ),
        ),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=[], epoch_interval=9999, parallel_evals=1),
    )


# =============================================================================
# 3. MATCHED-CAPACITY ABLATION
# =============================================================================

def train_matched_capacity() -> TrainTool:
    """Matched-capacity control: same architecture as GC-CRL, random objective.
    Same coef=0.1 as GC-CRL."""
    env = make_arena_env(num_agents=24)
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(enabled=False),
            goal_conditioned_crl=GoalConditionedCRLConfig(enabled=False),
            matched_capacity=MatchedCapacityConfig(
                enabled=True,
                hidden_dim=1024,
                embed_dim=64,
                capacity_coef=0.1,
            ),
        ),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=[], epoch_interval=9999, parallel_evals=1),
    )


# =============================================================================
# 4. INTERMEDIATE THERAPEUTIC WINDOW VALUES
# =============================================================================

def train_ppoc_coef_003() -> TrainTool:
    """PPO + Contrastive with intermediate coefficient (0.003)."""
    env = make_arena_env(num_agents=24)
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(
                enabled=True,
                temperature=0.1902943104505539,
                contrastive_coef=0.003,
                discount=0.977,
                embedding_dim=128,
                use_projection_head=True,
            ),
        ),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=[], epoch_interval=9999, parallel_evals=1),
    )


def train_ppoc_coef_005() -> TrainTool:
    """PPO + Contrastive with intermediate coefficient (0.005)."""
    env = make_arena_env(num_agents=24)
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(
                enabled=True,
                temperature=0.1902943104505539,
                contrastive_coef=0.005,
                discount=0.977,
                embedding_dim=128,
                use_projection_head=True,
            ),
        ),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=[], epoch_interval=9999, parallel_evals=1),
    )


def train_ppoc_coef_05() -> TrainTool:
    """PPO + Contrastive with high-intermediate coefficient (0.05)."""
    env = make_arena_env(num_agents=24)
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(
                enabled=True,
                temperature=0.1902943104505539,
                contrastive_coef=0.05,
                discount=0.977,
                embedding_dim=128,
                use_projection_head=True,
            ),
        ),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=[], epoch_interval=9999, parallel_evals=1),
    )


# =============================================================================
# 5. INTERMEDIATE COMPLEXITY: Arena with fewer agents
# =============================================================================

def train_baseline_6agent() -> TrainTool:
    """PPO baseline on Arena-6 (simpler than 24, more complex than Navigation)."""
    env = make_arena_env(num_agents=6)
    sims = [SimulationConfig(suite="icml_contrastive", name="arena_6agent", env=env)]
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(contrastive=ContrastiveConfig(enabled=False)),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=sims, epoch_interval=20),
    )


def train_ppoc_6agent() -> TrainTool:
    """PPO + Contrastive on Arena-6."""
    env = make_arena_env(num_agents=6)
    sims = [SimulationConfig(suite="icml_contrastive", name="arena_6agent", env=env)]
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(
                enabled=True,
                temperature=0.1902943104505539,
                contrastive_coef=0.0006806607125326991,
                discount=0.977,
                embedding_dim=128,
                use_projection_head=True,
            ),
        ),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=sims, epoch_interval=20),
    )


def train_baseline_12agent() -> TrainTool:
    """PPO baseline on Arena-12 (intermediate complexity)."""
    env = make_arena_env(num_agents=12)
    sims = [SimulationConfig(suite="icml_contrastive", name="arena_12agent", env=env)]
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(contrastive=ContrastiveConfig(enabled=False)),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=sims, epoch_interval=20),
    )


def train_ppoc_12agent() -> TrainTool:
    """PPO + Contrastive on Arena-12."""
    env = make_arena_env(num_agents=12)
    sims = [SimulationConfig(suite="icml_contrastive", name="arena_12agent", env=env)]
    trainer_config = TrainerConfig(
        total_timesteps=int(1e8),
        losses=LossesConfig(
            contrastive=ContrastiveConfig(
                enabled=True,
                temperature=0.1902943104505539,
                contrastive_coef=0.0006806607125326991,
                discount=0.977,
                embedding_dim=128,
                use_projection_head=True,
            ),
        ),
    )
    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(simulations=sims, epoch_interval=20),
    )


# =============================================================================
# Experiment Registry
# =============================================================================

EXPERIMENTS = {
    # 10-seed key conditions (20 runs total)
    "baseline_10seed": train_baseline_10seed,
    "ppoc_10seed": train_ppoc_10seed,
    # Standard regularizer baselines (6 runs: 3 seeds each)
    "l2_regularizer": train_l2_regularizer,
    "l2_regularizer_high": train_l2_regularizer_high,
    # Matched-capacity ablation (3 runs)
    "matched_capacity": train_matched_capacity,
    # Intermediate therapeutic window (9 runs: 3 seeds each)
    "ppoc_coef_003": train_ppoc_coef_003,
    "ppoc_coef_005": train_ppoc_coef_005,
    "ppoc_coef_05": train_ppoc_coef_05,
    # Intermediate complexity (12 runs: 3 seeds each, baseline + PPO+C)
    "baseline_6agent": train_baseline_6agent,
    "ppoc_6agent": train_ppoc_6agent,
    "baseline_12agent": train_baseline_12agent,
    "ppoc_12agent": train_ppoc_12agent,
}


def train(experiment: str = "baseline_10seed") -> TrainTool:
    """Run a specific experiment by name.

    Experiment plan (56 total runs):
    ┌─────────────────────────────────────────────┬───────┬──────────┐
    │ Condition                                    │ Seeds │ Purpose  │
    ├─────────────────────────────────────────────┼───────┼──────────┤
    │ PPO baseline (Arena-24)                      │  10   │ Power    │
    │ PPO+C default (Arena-24)                     │  10   │ Power    │
    │ L2 regularizer (coef=0.00068)                │   3   │ Baseline │
    │ L2 regularizer (coef=0.01)                   │   3   │ Baseline │
    │ Matched capacity (GC-CRL arch, random obj)   │   3   │ Ablation │
    │ PPO+C coef=0.003                             │   3   │ Window   │
    │ PPO+C coef=0.005                             │   3   │ Window   │
    │ PPO+C coef=0.05                              │   3   │ Window   │
    │ PPO baseline (Arena-6)                       │   3   │ Complex  │
    │ PPO+C (Arena-6)                              │   3   │ Complex  │
    │ PPO baseline (Arena-12)                      │   3   │ Complex  │
    │ PPO+C (Arena-12)                             │   3   │ Complex  │
    └─────────────────────────────────────────────┴───────┴──────────┘
    """
    if experiment not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment: {experiment}. Available: {available}")
    return EXPERIMENTS[experiment]()
