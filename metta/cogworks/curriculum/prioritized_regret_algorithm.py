"""Prioritized regret curriculum algorithm."""

from typing import Dict

import numpy as np

from .curriculum import CurriculumAlgorithmConfig
from .regret_algorithm_base import RegretAlgorithmBase


class PrioritizedRegretConfig(CurriculumAlgorithmConfig):
    """Configuration for prioritized regret."""

    type: str = "prioritized_regret"

    optimal_value: float = 1.0
    regret_ema_timescale: float = 0.01
    exploration_bonus: float = 0.1
    temperature: float = 1.0
    min_samples_for_prioritization: int = 2
    max_memory_tasks: int = 1000
    max_slice_axes: int = 3
    enable_detailed_slice_logging: bool = False

    def algorithm_type(self) -> str:
        return "prioritized_regret"

    def create(self, num_tasks: int) -> "PrioritizedRegretAlgorithm":
        return PrioritizedRegretAlgorithm(num_tasks, self)


class PrioritizedRegretAlgorithm(RegretAlgorithmBase):
    """Select tasks with highest regret."""

    def __init__(self, num_tasks: int, hypers: PrioritizedRegretConfig):
        super().__init__(num_tasks, hypers)
        self.hypers: PrioritizedRegretConfig = hypers

    def _score_task(self, task_id: int) -> float:
        """Calculate regret-based score for a task."""
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = self.regret_tracker.get_task_stats(task_id)

        if not task_stats or task_stats["completion_count"] < self.hypers.min_samples_for_prioritization:
            score = self.hypers.exploration_bonus
        else:
            regret = task_stats["ema_regret"]
            score = regret / max(self.hypers.temperature, 0.01)
            score += self._exploration_boost(task_stats["completion_count"])

        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def _eviction_fraction(self) -> float:
        return 0.3

    def _get_detailed_stats(self) -> Dict[str, float]:
        all_tasks = self.regret_tracker.get_all_tracked_tasks()
        if not all_tasks:
            return {
                "num_high_regret_tasks": 0.0,
                "num_low_regret_tasks": 0.0,
                "regret_std": 0.0,
            }

        regrets = [
            task_stats["ema_regret"]
            for task_id in all_tasks
            if (task_stats := self.regret_tracker.get_task_stats(task_id))
        ]
        if not regrets:
            return {
                "num_high_regret_tasks": 0.0,
                "num_low_regret_tasks": 0.0,
                "regret_std": 0.0,
            }

        regrets_array = np.array(regrets)
        median_regret = np.median(regrets_array)

        return {
            "num_high_regret_tasks": float(np.sum(regrets_array > median_regret)),
            "num_low_regret_tasks": float(np.sum(regrets_array <= median_regret)),
            "regret_std": float(np.std(regrets_array)),
        }
