"""Base class for regret-driven curriculum algorithms."""

from typing import Any, Dict, Optional

from .curriculum import CurriculumAlgorithm, CurriculumTask
from .regret_tracker import RegretTracker
from .task_tracker import TaskTracker


class RegretAlgorithmBase(CurriculumAlgorithm):
    """Shared plumbing for regret-based algorithms."""

    def __init__(self, num_tasks: int, hypers, use_task_tracker: bool = False):
        super().__init__(num_tasks, hypers)
        self.num_tasks = num_tasks
        self.hypers = hypers
        self.regret_tracker = RegretTracker(
            max_memory_tasks=hypers.max_memory_tasks,
            optimal_value=hypers.optimal_value,
            regret_ema_timescale=hypers.regret_ema_timescale,
        )
        self.task_tracker: Optional[TaskTracker] = (
            TaskTracker(max_memory_tasks=hypers.max_memory_tasks) if use_task_tracker else None
        )

        self._score_cache: Dict[int, float] = {}
        self._cache_valid_tasks: set[int] = set()

    def score_tasks(self, task_ids):
        return {task_id: self._score_task(task_id) for task_id in task_ids}

    def _score_task(self, task_id: int) -> float:
        raise NotImplementedError

    def _invalidate_score_cache(self) -> None:
        self._score_cache.clear()
        self._cache_valid_tasks.clear()

    def _exploration_boost(self, count: int, threshold: int = 10) -> float:
        if count >= threshold:
            return 0.0
        return self.hypers.exploration_bonus * (threshold - count) / threshold

    def recommend_eviction(self, task_ids):
        if not task_ids:
            return None
        scores = self.score_tasks(task_ids)
        return min(task_ids, key=lambda tid: scores.get(tid, 0.0))

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        stats = self._get_task_stats(task_id)
        if stats is None or stats["completion_count"] < min_presentations:
            return False

        task_ids = self._get_tracked_task_ids()
        if len(task_ids) <= 1:
            return False

        scores = self.score_tasks(task_ids)
        task_score = scores.get(task_id, 0.0)
        sorted_scores = sorted(scores.values())
        threshold_index = max(0, int(len(sorted_scores) * self._eviction_fraction()))
        threshold = sorted_scores[threshold_index] if sorted_scores else 0.0
        return task_score <= threshold

    def update_task_performance(self, task_id: int, score: float):
        if self.task_tracker is not None:
            self.task_tracker.update_task_performance(task_id, score)
        self.regret_tracker.update_task_performance(task_id, score)
        self._after_update_task_performance(task_id, score)
        self._cache_valid_tasks.discard(task_id)
        self.invalidate_cache()

    def _after_update_task_performance(self, task_id: int, score: float) -> None:
        pass

    def on_task_created(self, task: CurriculumTask) -> None:
        if self.task_tracker is not None:
            self.task_tracker.track_task_creation(task._task_id)
        self.regret_tracker.track_task_creation(task._task_id)

        slice_values = task.get_slice_values()
        if slice_values:
            self.slice_analyzer.update_task_completion(task._task_id, slice_values, 0.5)

        self._after_task_created(task)
        self.invalidate_cache()

    def _after_task_created(self, task: CurriculumTask) -> None:
        pass

    def on_task_evicted(self, task_id: int) -> None:
        if self.task_tracker is not None:
            self.task_tracker.remove_task(task_id)
        self.regret_tracker.remove_task(task_id)

        self._after_task_evicted(task_id)
        self._score_cache.pop(task_id, None)
        self._cache_valid_tasks.discard(task_id)
        self.invalidate_cache()

    def _after_task_evicted(self, task_id: int) -> None:
        pass

    def update_task_with_slice_values(self, task_id: int, score: float, slice_values: Dict[str, Any]) -> None:
        self.update_task_performance(task_id, score)
        self.slice_analyzer.update_task_completion(task_id, slice_values, score)

    def get_base_stats(self) -> Dict[str, float]:
        stats = {"num_tasks": self.num_tasks, **self.slice_analyzer.get_base_stats()}
        for key, value in self.regret_tracker.get_global_stats().items():
            stats[f"regret/{key}"] = value
        if self.task_tracker is not None:
            for key, value in self.task_tracker.get_global_stats().items():
                stats[f"tracker/{key}"] = value
        return stats

    def _detailed_stats_prefix(self) -> str:
        return "regret"

    def _get_detailed_stats(self) -> Dict[str, float]:
        return {}

    def stats(self, prefix: str = "") -> Dict[str, float]:
        cache_key = prefix if prefix else "_default"
        if self._stats_cache_valid and cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        stats = self.get_base_stats()
        detailed = self._get_detailed_stats()
        if detailed:
            detail_prefix = self._detailed_stats_prefix()
            for key, value in detailed.items():
                stats[f"{detail_prefix}/{key}"] = value

        if self.enable_detailed_logging:
            stats.update(self.get_detailed_stats())

        if prefix:
            stats = {f"{prefix}{k}": v for k, v in stats.items()}

        self._stats_cache[cache_key] = stats
        self._stats_cache_valid = True
        return stats

    def get_state(self) -> Dict[str, Any]:
        state = {
            "type": self.hypers.algorithm_type(),
            "hypers": self.hypers.model_dump(),
            "regret_tracker": self.regret_tracker.get_state(),
            "score_cache": self._score_cache,
            "cache_valid_tasks": list(self._cache_valid_tasks),
        }
        if self.task_tracker is not None:
            state["task_tracker"] = self.task_tracker.get_state()
        state.update(self._get_extra_state())
        return state

    def _get_extra_state(self) -> Dict[str, Any]:
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.regret_tracker.load_state(state["regret_tracker"])
        if self.task_tracker is not None and "task_tracker" in state:
            self.task_tracker.load_state(state["task_tracker"])
        self._score_cache = state.get("score_cache", {})
        self._cache_valid_tasks = set(state.get("cache_valid_tasks", []))
        self._load_extra_state(state)
        self._stats_cache_valid = False

    def _load_extra_state(self, state: Dict[str, Any]) -> None:
        pass

    def _get_task_stats(self, task_id: int):
        return self.regret_tracker.get_task_stats(task_id)

    def _get_tracked_task_ids(self):
        return self.regret_tracker.get_all_tracked_tasks()

    def _eviction_fraction(self) -> float:
        return 0.3
