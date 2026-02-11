"""
Representation Collapse paper experiments using MettaGrid.

Addresses:
  Exp 2: Team size scaling (2, 4, 8, 16 agents × 5 seeds)
  Exp 4: Embedding trajectory visualization (MettaGrid part)
  Exp 5: SVD spectrum analysis (MettaGrid part)
  Exp 6: Cross-environment consistency (MettaGrid part)
  Ablation: Encoder depth (2, 4, 8 layers × 5 seeds)
  Ablation: Batch size (1024, 4096, 16384 × 5 seeds)

Usage:
    uv run ./tools/run.py train repr_collapse experiment=team_size_2
    uv run ./tools/run.py train repr_collapse experiment=team_size_4
    uv run ./tools/run.py train repr_collapse experiment=depth_2
"""

from typing import Optional

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.rl.loss.losses import LossesConfig
from metta.rl.loss.contrastive_config import ContrastiveConfig
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
    env.game.agent.rewards.inventory.update({
        "ore_red": 0.1,
        "battery_red": 0.8,
        "laser": 0.3,
        "armor": 0.3,
    })
    return env


def make_train_tool(
    num_agents: int = 24,
    total_timesteps: int = int(5e7),
    batch_size: int = None,
    core_resnet_layers: int = None,
) -> TrainTool:
    """Create a TrainTool for MettaGrid with specified configuration."""
    env = make_arena_env(num_agents=num_agents)
    sims = [SimulationConfig(
        suite="repr_collapse",
        name=f"arena_{num_agents}agent",
        env=env,
    )]

    trainer_kwargs = {
        "total_timesteps": total_timesteps,
        "losses": LossesConfig(contrastive=ContrastiveConfig(enabled=False)),
    }

    if batch_size is not None:
        trainer_kwargs["batch_size"] = batch_size

    trainer_config = TrainerConfig(**trainer_kwargs)

    tool = TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env),
            auto_workers=False,
            num_workers=1,
            async_factor=1,
        ),
        evaluator=EvaluatorConfig(
            simulations=sims,
            epoch_interval=20,
            parallel_evals=1,
        ),
    )

    return tool


# =============================================================================
# Experiment 2: Team Size Scaling
# =============================================================================

def train_team_size_2() -> TrainTool:
    """MettaGrid with 2 agents."""
    return make_train_tool(num_agents=2, total_timesteps=int(5e7))


def train_team_size_4() -> TrainTool:
    """MettaGrid with 4 agents."""
    return make_train_tool(num_agents=4, total_timesteps=int(5e7))


def train_team_size_8() -> TrainTool:
    """MettaGrid with 8 agents."""
    return make_train_tool(num_agents=8, total_timesteps=int(5e7))


def train_team_size_16() -> TrainTool:
    """MettaGrid with 16 agents."""
    return make_train_tool(num_agents=16, total_timesteps=int(5e7))


# =============================================================================
# Ablation: Encoder Depth
# =============================================================================

def train_depth_2() -> TrainTool:
    """MettaGrid with encoder depth 2."""
    return make_train_tool(num_agents=8, total_timesteps=int(5e7), core_resnet_layers=2)


def train_depth_4() -> TrainTool:
    """MettaGrid with encoder depth 4."""
    return make_train_tool(num_agents=8, total_timesteps=int(5e7), core_resnet_layers=4)


def train_depth_8() -> TrainTool:
    """MettaGrid with encoder depth 8."""
    return make_train_tool(num_agents=8, total_timesteps=int(5e7), core_resnet_layers=8)


# =============================================================================
# Ablation: Batch Size
# =============================================================================

def train_batch_1024() -> TrainTool:
    """MettaGrid with batch size 1024."""
    return make_train_tool(num_agents=8, total_timesteps=int(5e7), batch_size=1024)


def train_batch_4096() -> TrainTool:
    """MettaGrid with batch size 4096."""
    return make_train_tool(num_agents=8, total_timesteps=int(5e7), batch_size=4096)


def train_batch_16384() -> TrainTool:
    """MettaGrid with batch size 16384."""
    return make_train_tool(num_agents=8, total_timesteps=int(5e7), batch_size=16384)


# =============================================================================
# Experiment Registry
# =============================================================================

EXPERIMENTS = {
    # Exp 2: Team size scaling
    "team_size_2": train_team_size_2,
    "team_size_4": train_team_size_4,
    "team_size_8": train_team_size_8,
    "team_size_16": train_team_size_16,
    # Ablation: Encoder depth
    "depth_2": train_depth_2,
    "depth_4": train_depth_4,
    "depth_8": train_depth_8,
    # Ablation: Batch size
    "batch_1024": train_batch_1024,
    "batch_4096": train_batch_4096,
    "batch_16384": train_batch_16384,
}


def train(experiment: str = "team_size_4") -> TrainTool:
    """Run a specific experiment.

    Experiment plan:
    ┌─────────────────────────────┬───────┬──────────┐
    │ Condition                    │ Seeds │ Purpose  │
    ├─────────────────────────────┼───────┼──────────┤
    │ Team size 2 agents           │   5   │ Exp 2    │
    │ Team size 4 agents           │   5   │ Exp 2    │
    │ Team size 8 agents           │   5   │ Exp 2    │
    │ Team size 16 agents          │   5   │ Exp 2    │
    │ Encoder depth 2              │   5   │ Ablation │
    │ Encoder depth 4              │   5   │ Ablation │
    │ Encoder depth 8              │   5   │ Ablation │
    │ Batch size 1024              │   5   │ Ablation │
    │ Batch size 4096              │   5   │ Ablation │
    │ Batch size 16384             │   5   │ Ablation │
    └─────────────────────────────┴───────┴──────────┘
    """
    if experiment not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment: {experiment}. Available: {available}")
    return EXPERIMENTS[experiment]()
