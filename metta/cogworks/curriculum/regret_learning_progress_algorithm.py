"""Regret-based learning progress curriculum algorithm."""

from typing import Any, Dict, List, Optional

import numpy as np

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .regret_tracker import RegretTracker
from .task_tracker import TaskTracker


class RegretLearningProgressConfig(CurriculumAlgorithmConfig):
    """Configuration for regret learning progress."""

    type: str = "regret_learning_progress"

    optimal_value: float = 1.0
    regret_ema_timescale: float = 0.001
    use_bidirectional: bool = True
    exploration_bonus: float = 0.1
    progress_smoothing: float = 0.05
    invert_regret_progress: bool = True
    min_samples_for_lp: int = 2
    memory: int = 25
    max_memory_tasks: int = 1000
    max_slice_axes: int = 3
    enable_detailed_slice_logging: bool = False

    def algorithm_type(self) -> str:
        return "regret_learning_progress"

    def create(self, num_tasks: int) -> "RegretLearningProgressAlgorithm":
        return RegretLearningProgressAlgorithm(num_tasks, self)


class RegretLearningProgressAlgorithm(CurriculumAlgorithm):
    """Prioritize tasks where regret decreases fastest."""

    def __init__(self, num_tasks: int, hypers: RegretLearningProgressConfig):
        super().__init__(num_tasks, hypers)

        self.num_tasks = num_tasks
        self.hypers: RegretLearningProgressConfig = hypers

        self.regret_tracker = RegretTracker(
            max_memory_tasks=hypers.max_memory_tasks,
            optimal_value=hypers.optimal_value,
            regret_ema_timescale=hypers.regret_ema_timescale,
        )
        self.task_tracker = TaskTracker(max_memory_tasks=hypers.max_memory_tasks)

        if hypers.use_bidirectional:
            self._init_bidirectional_regret_tracking()

        self._score_cache: Dict[int, float] = {}
        self._cache_valid_tasks: set[int] = set()
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_valid = False

    def _init_bidirectional_regret_tracking(self):
        """Initialize bidirectional regret tracking."""
        self._regret_outcomes: Dict[int, List[float]] = {}
        self._r_fast: Optional[np.ndarray] = None
        self._r_slow: Optional[np.ndarray] = None
        self._task_dist: Optional[np.ndarray] = None
        self._stale_dist = True
        self._update_mask: np.ndarray = np.array([])
        self._counter: Dict[int, int] = {}
        self._task_ids: List[int] = []

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks by regret learning progress."""
        return {task_id: self._get_regret_learning_progress_score(task_id) for task_id in task_ids}

    def _get_regret_learning_progress_score(self, task_id: int) -> float:
        """Calculate regret learning progress score for a task."""
        if self.hypers.use_bidirectional and self._stale_dist:
            self._invalidate_score_cache()

        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = self.task_tracker.get_task_stats(task_id)

        if not task_stats or task_stats["completion_count"] < self.hypers.min_samples_for_lp:
            score = self.hypers.exploration_bonus
        elif self.hypers.use_bidirectional:
            score = self._compute_bidirectional_score(task_id)
        else:
            regret_progress = self.regret_tracker.get_regret_progress(task_id)
            if regret_progress is None:
                score = self.hypers.exploration_bonus
            else:
                score = max(0.0, -regret_progress) if self.hypers.invert_regret_progress else abs(regret_progress)
                if task_stats["completion_count"] < 10:
                    score += self.hypers.exploration_bonus * (10 - task_stats["completion_count"]) / 10

        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def _compute_bidirectional_score(self, task_id: int) -> float:
        """Compute bidirectional learning progress score."""
        if task_id not in self._regret_outcomes or len(self._regret_outcomes[task_id]) < 2:
            return self.hypers.exploration_bonus

        self._update_regret_progress()

        if self._task_dist is None or self._stale_dist:
            self._calculate_regret_task_distribution()

        task_indices = self._task_ids or sorted(self._regret_outcomes.keys())
        if task_id in task_indices and self._task_dist is not None:
            task_idx = task_indices.index(task_id)
            if task_idx < len(self._task_dist):
                return float(self._task_dist[task_idx])

        return self.hypers.exploration_bonus

    def _update_regret_progress(self):
        """Update bidirectional regret progress tracking."""
        if not self._regret_outcomes:
            return

        task_ids = sorted(self._regret_outcomes.keys())
        if not task_ids:
            return
        num_tasks = len(task_ids)

        mean_regrets = np.array(
            [np.mean(self._regret_outcomes[task_id]) if self._regret_outcomes[task_id] else 0.0 for task_id in task_ids]
        )

        self._update_mask = np.array([len(self._regret_outcomes[task_id]) >= 2 for task_id in task_ids])
        old_task_ids = self._task_ids
        old_index = {task_id: idx for idx, task_id in enumerate(old_task_ids)}
        self._task_ids = task_ids

        new_r_fast = np.zeros(num_tasks)
        new_r_slow = np.zeros(num_tasks)

        if self._r_fast is None or self._r_slow is None:
            new_r_fast = mean_regrets.copy()
            new_r_slow = mean_regrets.copy()
        else:
            for idx, task_id in enumerate(task_ids):
                old_idx = old_index.get(task_id)
                if old_idx is not None:
                    new_r_fast[idx] = self._r_fast[old_idx]
                    new_r_slow[idx] = self._r_slow[old_idx]
                else:
                    new_r_fast[idx] = mean_regrets[idx]
                    new_r_slow[idx] = mean_regrets[idx]

        self._r_fast = new_r_fast
        self._r_slow = new_r_slow

        if np.any(self._update_mask):
            self._r_fast[self._update_mask] = mean_regrets[
                self._update_mask
            ] * self.hypers.regret_ema_timescale + self._r_fast[self._update_mask] * (
                1.0 - self.hypers.regret_ema_timescale
            )
            slow_timescale = self.hypers.regret_ema_timescale * 0.2
            self._r_slow[self._update_mask] = mean_regrets[self._update_mask] * slow_timescale + self._r_slow[
                self._update_mask
            ] * (1.0 - slow_timescale)

        self._stale_dist = True

    def _regret_learning_progress(self) -> np.ndarray:
        """Calculate regret learning progress as slow - fast."""
        if self._r_fast is None or self._r_slow is None:
            return np.array([])

        regret_lp = self._r_slow - self._r_fast

        lp = np.abs(regret_lp)

        if self.hypers.invert_regret_progress:
            improvement_bonus = np.maximum(regret_lp, 0) * 0.2
            lp = lp + improvement_bonus

        return lp

    def _calculate_regret_task_distribution(self):
        """Calculate task distribution based on regret learning progress."""
        if not self._regret_outcomes:
            self._task_dist = np.array([])
            self._stale_dist = False
            return

        num_tasks = len(self._regret_outcomes)
        task_dist = np.ones(num_tasks) / num_tasks

        regret_lp = self._regret_learning_progress()
        if regret_lp.size == 0:
            self._task_dist = task_dist
            self._stale_dist = False
            return

        posidxs = [i for i, lp in enumerate(regret_lp) if lp > 0]

        any_progress = len(posidxs) > 0
        subprobs = regret_lp[posidxs] if any_progress else regret_lp

        subprobs = subprobs - np.mean(subprobs)
        std = np.std(subprobs)
        if std > 0:
            subprobs = subprobs / std

        subprobs = self._sigmoid(subprobs)

        sum_probs = np.sum(subprobs)
        if sum_probs > 0:
            subprobs = subprobs / sum_probs
        else:
            subprobs = np.ones_like(subprobs) / len(subprobs)

        if any_progress:
            task_dist = np.zeros(len(regret_lp))
            task_dist[posidxs] = subprobs
        else:
            task_dist = subprobs

        self._task_dist = task_dist.astype(np.float32)
        self._stale_dist = False

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function to array values."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """Evict tasks with lowest regret learning progress."""
        if not task_ids:
            return None

        scores = self.score_tasks(task_ids)
        min_task_id = min(task_ids, key=lambda tid: scores.get(tid, 0.0))
        return min_task_id

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        """Check if a task should be evicted."""
        task_stats = self.task_tracker.get_task_stats(task_id)
        if task_stats is None:
            return False

        if task_stats["completion_count"] < min_presentations:
            return False

        all_task_ids = self.task_tracker.get_all_tracked_tasks()
        if len(all_task_ids) <= 1:
            return False

        scores = self.score_tasks(all_task_ids)
        task_score = scores.get(task_id, 0.0)

        sorted_scores = sorted(scores.values())
        threshold_index = max(0, int(len(sorted_scores) * 0.4))
        threshold_score = sorted_scores[threshold_index] if sorted_scores else 0.0

        return task_score <= threshold_score

    def on_task_evicted(self, task_id: int) -> None:
        """Clean up when a task is evicted."""
        self.regret_tracker.remove_task(task_id)
        self.task_tracker.remove_task(task_id)

        if self.hypers.use_bidirectional:
            self._regret_outcomes.pop(task_id, None)
            self._counter.pop(task_id, None)
            self._stale_dist = True

        if self.hypers.use_bidirectional:
            self._invalidate_score_cache()
        else:
            self._score_cache.pop(task_id, None)
            self._cache_valid_tasks.discard(task_id)
        self.invalidate_cache()

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance and regret tracking."""
        self.task_tracker.update_task_performance(task_id, score)
        self.regret_tracker.update_task_performance(task_id, score)

        if self.hypers.use_bidirectional:
            regret = self.regret_tracker.compute_regret(score)

            if task_id not in self._regret_outcomes:
                self._regret_outcomes[task_id] = []

            self._regret_outcomes[task_id].append(regret)
            self._regret_outcomes[task_id] = self._regret_outcomes[task_id][-self.hypers.memory :]

            if task_id not in self._counter:
                self._counter[task_id] = 0
            self._counter[task_id] += 1

            self._update_regret_progress()
            self._stale_dist = True
            self._invalidate_score_cache()

        if not self.hypers.use_bidirectional:
            self._cache_valid_tasks.discard(task_id)
        self.invalidate_cache()

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle task creation."""
        self.task_tracker.track_task_creation(task._task_id)
        self.regret_tracker.track_task_creation(task._task_id)

        slice_values = task.get_slice_values()
        if slice_values:
            self.slice_analyzer.update_task_completion(task._task_id, slice_values, 0.5)

        if self.hypers.use_bidirectional:
            self._stale_dist = True
            self._invalidate_score_cache()

        self.invalidate_cache()

    def update_task_with_slice_values(self, task_id: int, score: float, slice_values: Dict[str, Any]) -> None:
        """Update task performance including slice values."""
        self.update_task_performance(task_id, score)
        self.slice_analyzer.update_task_completion(task_id, slice_values, score)

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics."""
        base_stats = {"num_tasks": self.num_tasks, **self.slice_analyzer.get_base_stats()}

        regret_stats = self.regret_tracker.get_global_stats()
        for key, value in regret_stats.items():
            base_stats[f"regret/{key}"] = value

        task_stats = self.task_tracker.get_global_stats()
        for key, value in task_stats.items():
            base_stats[f"tracker/{key}"] = value

        return base_stats

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get all statistics with optional prefix."""
        cache_key = prefix if prefix else "_default"

        if self._stats_cache_valid and cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        stats = self.get_base_stats()

        detailed_stats = self._get_detailed_regret_lp_stats()
        for key, value in detailed_stats.items():
            stats[f"regret_lp/{key}"] = value

        if self.enable_detailed_logging:
            detailed = self.get_detailed_stats()
            stats.update(detailed)

        if prefix:
            stats = {f"{prefix}{k}": v for k, v in stats.items()}

        self._stats_cache[cache_key] = stats
        self._stats_cache_valid = True

        return stats

    def _get_detailed_regret_lp_stats(self) -> Dict[str, float]:
        """Get detailed regret learning progress statistics."""
        if not self._regret_outcomes:
            return {
                "num_tracked_tasks": 0.0,
                "mean_regret_lp": 0.0,
                "num_decreasing_regret": 0.0,
                "num_increasing_regret": 0.0,
            }

        self._update_regret_progress()

        stats = {
            "num_tracked_tasks": float(len(self._regret_outcomes)),
        }

        if self._r_fast is not None and self._r_slow is not None:
            regret_lp = self._regret_learning_progress()
            if len(regret_lp) > 0:
                stats["mean_regret_lp"] = float(np.mean(regret_lp))

                # Count tasks with decreasing vs increasing regret
                regret_changes = self._r_slow - self._r_fast
                stats["num_decreasing_regret"] = float(np.sum(regret_changes > 0))
                stats["num_increasing_regret"] = float(np.sum(regret_changes < 0))
            else:
                stats["mean_regret_lp"] = 0.0
                stats["num_decreasing_regret"] = 0.0
                stats["num_increasing_regret"] = 0.0
        else:
            stats["mean_regret_lp"] = 0.0
            stats["num_decreasing_regret"] = 0.0
            stats["num_increasing_regret"] = 0.0

        return stats

    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state for checkpointing."""
        state = {
            "type": self.hypers.algorithm_type(),
            "hypers": self.hypers.model_dump(),
            "regret_tracker": self.regret_tracker.get_state(),
            "task_tracker": self.task_tracker.get_state(),
        }

        if self.hypers.use_bidirectional:
            state.update(
                {
                    "regret_outcomes": {k: v for k, v in self._regret_outcomes.items()},
                    "counter": self._counter,
                    "r_fast": self._r_fast.tolist() if self._r_fast is not None else None,
                    "r_slow": self._r_slow.tolist() if self._r_slow is not None else None,
                    "task_dist": self._task_dist.tolist() if self._task_dist is not None else None,
                    "stale_dist": self._stale_dist,
                    "update_mask": self._update_mask.tolist(),
                    "task_ids": list(self._task_ids),
                    "score_cache": self._score_cache,
                    "cache_valid_tasks": list(self._cache_valid_tasks),
                }
            )

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load algorithm state from checkpoint."""
        self.regret_tracker.load_state(state["regret_tracker"])
        self.task_tracker.load_state(state["task_tracker"])

        if "regret_outcomes" in state:
            self._regret_outcomes = state["regret_outcomes"]
            self._counter = state["counter"]
            self._r_fast = np.array(state["r_fast"]) if state["r_fast"] is not None else None
            self._r_slow = np.array(state["r_slow"]) if state["r_slow"] is not None else None
            self._task_dist = np.array(state["task_dist"]) if state["task_dist"] is not None else None
            self._stale_dist = state["stale_dist"]
            self._update_mask = np.array(state["update_mask"])
            self._task_ids = state.get("task_ids", sorted(self._regret_outcomes.keys()))
            self._score_cache = state.get("score_cache", {})
            self._cache_valid_tasks = set(state.get("cache_valid_tasks", []))

        self._stats_cache_valid = False

    def _invalidate_score_cache(self) -> None:
        """Invalidate cached scores for all tasks (distribution-dependent)."""
        self._score_cache.clear()
        self._cache_valid_tasks.clear()
