"""
Experiment configurations for the contrastive learning paper.

This file contains all baseline and ablation experiments described in the paper:
- Baseline: PPO without contrastive loss
- PPO + Contrastive (default configuration)
- Ablation 1: No projection head
- Ablation 2: Temperature = 0.05
- Ablation 3: Temperature = 0.5
- Ablation 4: Higher contrastive coefficient (0.01)
- Ablation 5: Fixed temporal offset (Δt = 10) instead of geometric sampling

All experiments use the Tribal Village environment with matched training budgets.
"""

from typing import Optional

from metta.rl.loss.contrastive_config import ContrastiveConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig


def make_tribal_village_env(num_agents: int = 24) -> MettaGridConfig:
    """Create Tribal Village environment configuration."""
    # Import the tribal village environment builder
    # You may need to adjust this import based on your environment setup
    from mettagrid.builder import envs as eb

    env = eb.make_tribal_village(num_agents=num_agents)
    return env


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    """Create evaluation environments."""
    basic_env = env or make_tribal_village_env()

    return [
        SimulationConfig(suite="contrastive_paper", name="tribal_basic", env=basic_env),
    ]


# ============================================================================
# Baseline: PPO without contrastive loss
# ============================================================================

def train_baseline_ppo() -> TrainTool:
    """
    Baseline: PPO without contrastive loss.

    This is the control condition for measuring the impact of contrastive learning.
    """
    env = make_tribal_village_env()

    contrastive_config = ContrastiveConfig(
        enabled=False,  # Disabled for baseline
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(1e9),  # 1B timesteps as per paper
        losses=LossesConfig(contrastive=contrastive_config),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(),
        evaluator=EvaluatorConfig(simulations=simulations(env)),
    )


# ============================================================================
# Main: PPO + Contrastive (default configuration from paper)
# ============================================================================

def train_ppo_plus_contrastive() -> TrainTool:
    """
    PPO + Contrastive with default configuration from the paper.

    Default hyperparameters:
    - temperature: 0.1902943104505539
    - contrastive_coef: 0.0006806607125326991
    - discount (gamma): 0.977
    - embedding_dim: 128
    - use_projection_head: True
    """
    env = make_tribal_village_env()

    contrastive_config = ContrastiveConfig(
        enabled=True,
        temperature=0.1902943104505539,  # Default from config
        contrastive_coef=0.0006806607125326991,  # Default from config
        discount=0.977,  # Geometric sampling parameter
        embedding_dim=128,
        use_projection_head=True,
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(1e9),  # 1B timesteps
        losses=LossesConfig(contrastive=contrastive_config),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(),
        evaluator=EvaluatorConfig(simulations=simulations(env)),
    )


# ============================================================================
# Ablation 1: No projection head
# ============================================================================

def train_ablation_no_projection_head() -> TrainTool:
    """
    Ablation: No projection head (use_projection_head=False).

    Tests whether the projection head is necessary for good performance.
    """
    env = make_tribal_village_env()

    contrastive_config = ContrastiveConfig(
        enabled=True,
        temperature=0.1902943104505539,
        contrastive_coef=0.0006806607125326991,
        discount=0.977,
        embedding_dim=128,
        use_projection_head=False,  # ABLATION: No projection head
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(1e9),
        losses=LossesConfig(contrastive=contrastive_config),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(),
        evaluator=EvaluatorConfig(simulations=simulations(env)),
    )


# ============================================================================
# Ablation 2: Temperature = 0.05 (lower temperature)
# ============================================================================

def train_ablation_temperature_low() -> TrainTool:
    """
    Ablation: Lower temperature (τ = 0.05).

    Lower temperature creates sharper distinctions between positive and negative pairs.
    """
    env = make_tribal_village_env()

    contrastive_config = ContrastiveConfig(
        enabled=True,
        temperature=0.05,  # ABLATION: Lower temperature
        contrastive_coef=0.0006806607125326991,
        discount=0.977,
        embedding_dim=128,
        use_projection_head=True,
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(1e9),
        losses=LossesConfig(contrastive=contrastive_config),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(),
        evaluator=EvaluatorConfig(simulations=simulations(env)),
    )


# ============================================================================
# Ablation 3: Temperature = 0.5 (higher temperature)
# ============================================================================

def train_ablation_temperature_high() -> TrainTool:
    """
    Ablation: Higher temperature (τ = 0.5).

    Higher temperature softens the distinctions between positive and negative pairs.
    """
    env = make_tribal_village_env()

    contrastive_config = ContrastiveConfig(
        enabled=True,
        temperature=0.5,  # ABLATION: Higher temperature
        contrastive_coef=0.0006806607125326991,
        discount=0.977,
        embedding_dim=128,
        use_projection_head=True,
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(1e9),
        losses=LossesConfig(contrastive=contrastive_config),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(),
        evaluator=EvaluatorConfig(simulations=simulations(env)),
    )


# ============================================================================
# Ablation 4: Higher contrastive coefficient (0.01)
# ============================================================================

def train_ablation_higher_coefficient() -> TrainTool:
    """
    Ablation: Higher contrastive coefficient (α_c = 0.01).

    Tests whether increasing the weight of contrastive loss improves or destabilizes training.
    Default coefficient is ~0.00068, so this is about 15x higher.
    """
    env = make_tribal_village_env()

    contrastive_config = ContrastiveConfig(
        enabled=True,
        temperature=0.1902943104505539,
        contrastive_coef=0.01,  # ABLATION: Higher coefficient
        discount=0.977,
        embedding_dim=128,
        use_projection_head=True,
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(1e9),
        losses=LossesConfig(contrastive=contrastive_config),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(),
        evaluator=EvaluatorConfig(simulations=simulations(env)),
    )


# ============================================================================
# Ablation 5: Fixed temporal offset (Δt = 10)
# ============================================================================

def train_ablation_fixed_temporal_offset() -> TrainTool:
    """
    Ablation: Fixed temporal offset (Δt = 10) instead of geometric sampling.

    Tests the importance of geometric sampling for temporal coherence.
    To achieve a fixed offset, we set discount very close to 0, which makes
    the geometric distribution very peaked at the first few values.

    NOTE: This requires modifying the contrastive loss implementation to support
    a fixed offset mode. For now, we approximate with a very low discount factor.
    """
    env = make_tribal_village_env()

    # With discount = 0.1, the expected delta is 1/(1-0.1) = ~1.1 steps
    # We need to modify the implementation to support fixed offsets properly
    # For now, this serves as a placeholder
    contrastive_config = ContrastiveConfig(
        enabled=True,
        temperature=0.1902943104505539,
        contrastive_coef=0.0006806607125326991,
        discount=0.1,  # ABLATION: Low discount to approximate fixed offset
        embedding_dim=128,
        use_projection_head=True,
    )

    trainer_config = TrainerConfig(
        total_timesteps=int(1e9),
        losses=LossesConfig(contrastive=contrastive_config),
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(),
        evaluator=EvaluatorConfig(simulations=simulations(env)),
    )


# ============================================================================
# Experiment runners - for easy execution
# ============================================================================

# Map experiment names to their training functions
EXPERIMENTS = {
    "baseline_ppo": train_baseline_ppo,
    "ppo_plus_contrastive": train_ppo_plus_contrastive,
    "ablation_no_projection": train_ablation_no_projection_head,
    "ablation_temp_0.05": train_ablation_temperature_low,
    "ablation_temp_0.5": train_ablation_temperature_high,
    "ablation_coef_0.01": train_ablation_higher_coefficient,
    "ablation_fixed_offset": train_ablation_fixed_temporal_offset,
}


def train(experiment_name: str = "ppo_plus_contrastive") -> TrainTool:
    """
    Run a specific experiment by name.

    Args:
        experiment_name: One of the experiment names from EXPERIMENTS dict

    Available experiments:
        - baseline_ppo: PPO without contrastive loss
        - ppo_plus_contrastive: PPO + Contrastive (default)
        - ablation_no_projection: No projection head
        - ablation_temp_0.05: Temperature = 0.05
        - ablation_temp_0.5: Temperature = 0.5
        - ablation_coef_0.01: Higher contrastive coefficient
        - ablation_fixed_offset: Fixed temporal offset
    """
    if experiment_name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Available experiments: {available}"
        )

    return EXPERIMENTS[experiment_name]()


# ============================================================================
# Metrics tracked by wandb (for reference)
# ============================================================================

"""
The following metrics are automatically logged to wandb under the 'losses/' prefix:

1. losses/contrastive_loss - The weighted contrastive loss contribution
2. losses/positive_sim_mean - Mean similarity between anchor and positive pairs
3. losses/positive_sim_std - Standard deviation of positive similarities
4. losses/negative_sim_mean - Mean similarity between anchor and negative pairs
5. losses/negative_sim_std - Standard deviation of negative similarities
6. losses/num_pairs - Number of valid contrastive pairs sampled
7. losses/delta_mean - Average temporal distance (Δt) between anchor and positive

Additional RL metrics logged:
- overview/reward - Episode return
- overview/sps - Steps per second
- experience/* - Various experience statistics
- And many more (see metta/rl/training/stats_reporter.py)

Interpretation (from paper Section 4.1):
- Healthy training: positive_sim_mean > negative_sim_mean with growing gap
- Representation collapse: positive_sim_mean ≈ negative_sim_mean
- Insufficient capacity: High positive_sim_std
"""
