"""Analyze role-like behavior specialization in Tribal Village rollouts."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_experiments.tribal_behavior import SIMULATOR_STAT_COLUMNS  # noqa: E402

CHAIN_STAGE_COLUMNS = ("resource_ore", "craft_battery", "deposit_heart")
CHAIN_STAGE_NAMES = ("ore", "battery", "heart")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze_paths(
        [Path(path) for path in args.paths],
        specialization_threshold=args.specialization_threshold,
        min_chain_events=args.min_chain_events,
        role_stage_gap_threshold=args.role_stage_gap_threshold,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="rollout_metrics.json files")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--specialization-threshold",
        type=float,
        default=0.60,
        help="Agent is specialized when this fraction of its chain events is one stage.",
    )
    parser.add_argument(
        "--min-chain-events",
        type=int,
        default=10,
        help="Minimum ore+battery+heart events before an agent can be called specialized.",
    )
    parser.add_argument(
        "--role-stage-gap-threshold",
        type=float,
        default=0.15,
        help="Largest agent_id %% 3 stage-share gap needed to flag fixed-role behavior separation.",
    )
    return parser.parse_args(argv)


def analyze_paths(
    paths: Sequence[Path],
    *,
    specialization_threshold: float = 0.60,
    min_chain_events: int = 10,
    role_stage_gap_threshold: float = 0.15,
) -> dict[str, Any]:
    records = [
        analyze_rollout_record(
            json.loads(path.read_text()),
            path=path,
            specialization_threshold=specialization_threshold,
            min_chain_events=min_chain_events,
            role_stage_gap_threshold=role_stage_gap_threshold,
        )
        for path in sorted(paths)
    ]
    return {
        "schema_version": "tribal_behavior_role_analysis_v1",
        "specialization_threshold": specialization_threshold,
        "min_chain_events": min_chain_events,
        "role_stage_gap_threshold": role_stage_gap_threshold,
        "records": records,
        "condition_summary": _condition_summary(records),
    }


def analyze_rollout_record(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    specialization_threshold: float = 0.60,
    min_chain_events: int = 10,
    role_stage_gap_threshold: float = 0.15,
) -> dict[str, Any]:
    per_agent_stats = _sum_agent_stats(record)
    chain_counts = np.stack(
        [
            np.asarray([agent_stats[column] for column in CHAIN_STAGE_COLUMNS], dtype=np.float64)
            for agent_stats in per_agent_stats
        ]
    )
    agent_chain_totals = chain_counts.sum(axis=1)
    chain_stage_totals = chain_counts.sum(axis=0)
    stage_shares = np.divide(
        chain_counts,
        agent_chain_totals[:, None],
        out=np.zeros_like(chain_counts),
        where=agent_chain_totals[:, None] > 0,
    )
    dominant_stage_indexes = np.argmax(stage_shares, axis=1)
    max_stage_shares = stage_shares.max(axis=1)
    specialized_mask = (max_stage_shares >= specialization_threshold) & (agent_chain_totals >= min_chain_events)
    all_chain_stages_mask = (chain_counts > 0).all(axis=1)
    fixed_role_sums = _fixed_role_sums(chain_counts)
    fixed_role_stage_shares = np.divide(
        fixed_role_sums,
        fixed_role_sums.sum(axis=1, keepdims=True),
        out=np.zeros_like(fixed_role_sums),
        where=fixed_role_sums.sum(axis=1, keepdims=True) > 0,
    )
    max_fixed_role_stage_gap = _max_column_gap(fixed_role_stage_shares)
    return {
        "path": None if path is None else str(path),
        "condition": _infer_condition(path, record),
        "rollout_name": None if path is None else path.parent.name,
        "policy": record.get("policy"),
        "checkpoint_path": record.get("checkpoint_path"),
        "seed": record.get("seed"),
        "episodes": record.get("episodes"),
        "steps_per_episode": record.get("steps_per_episode"),
        "chain_stage_names": list(CHAIN_STAGE_NAMES),
        "chain_stage_totals": _named_float_dict(CHAIN_STAGE_NAMES, chain_stage_totals),
        "per_agent_chain_counts": [
            _named_float_dict(CHAIN_STAGE_NAMES, chain_counts[agent_id]) for agent_id in range(chain_counts.shape[0])
        ],
        "per_agent_chain_totals": [float(value) for value in agent_chain_totals],
        "per_agent_stage_shares": [
            _named_float_dict(CHAIN_STAGE_NAMES, stage_shares[agent_id]) for agent_id in range(stage_shares.shape[0])
        ],
        "dominant_stage_by_agent": [CHAIN_STAGE_NAMES[int(index)] for index in dominant_stage_indexes],
        "dominant_stage_counts": dict(Counter(CHAIN_STAGE_NAMES[int(index)] for index in dominant_stage_indexes)),
        "mean_max_stage_share": float(max_stage_shares.mean()) if max_stage_shares.size else 0.0,
        "specialized_agent_count": int(specialized_mask.sum()),
        "agents_with_all_chain_stages": int(all_chain_stages_mask.sum()),
        "stage_cv": _named_float_dict(
            CHAIN_STAGE_NAMES,
            [_coefficient_of_variation(chain_counts[:, i]) for i in range(3)],
        ),
        "fixed_role_stage_sums": [
            _named_float_dict(CHAIN_STAGE_NAMES, fixed_role_sums[role_id])
            for role_id in range(fixed_role_sums.shape[0])
        ],
        "fixed_role_stage_shares": [
            _named_float_dict(CHAIN_STAGE_NAMES, fixed_role_stage_shares[role_id])
            for role_id in range(fixed_role_stage_shares.shape[0])
        ],
        "max_fixed_role_stage_share_gap": max_fixed_role_stage_gap,
        "fixed_role_behavior_separation": max_fixed_role_stage_gap >= role_stage_gap_threshold,
    }


def _sum_agent_stats(record: dict[str, Any]) -> list[dict[str, float]]:
    expected_agents = None
    sums: list[dict[str, float]] | None = None
    for episode in record.get("episode_metrics", []):
        metrics = episode.get("behavior_metrics", {})
        stats = metrics.get("simulator_action_stats_by_agent")
        if not stats:
            continue
        if sums is None:
            expected_agents = len(stats)
            sums = [{column: 0.0 for column in SIMULATOR_STAT_COLUMNS} for _ in range(expected_agents)]
        if len(stats) != expected_agents:
            raise ValueError("all episodes must have the same number of agents")
        for agent_id, row in enumerate(stats):
            for column in SIMULATOR_STAT_COLUMNS:
                sums[agent_id][column] += float(row.get(column, 0.0))
    if sums is None:
        raise ValueError("rollout record has no simulator_action_stats_by_agent data")
    return sums


def _fixed_role_sums(chain_counts: np.ndarray, num_roles: int = 3) -> np.ndarray:
    role_sums = np.zeros((num_roles, chain_counts.shape[1]), dtype=np.float64)
    for agent_id, row in enumerate(chain_counts):
        role_sums[agent_id % num_roles] += row
    return role_sums


def _max_column_gap(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values.max(axis=0) - values.min(axis=0)))


def _coefficient_of_variation(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    if mean == 0.0:
        return 0.0
    return float(arr.std(ddof=0) / mean)


def _named_float_dict(names: Sequence[str], values: Iterable[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _infer_condition(path: Path | None, record: dict[str, Any]) -> str:
    text = "" if path is None else str(path)
    checkpoint_path = str(record.get("checkpoint_path") or "")
    combined = f"{text} {checkpoint_path}"
    if "baseline_" in combined:
        return "baseline"
    if "alpha0p6" in combined or "alpha0_to_alpha0p6" in combined:
        return "alpha0p6_promotion"
    if "alpha0_source" in combined or "alpha0_sources" in combined:
        return "alpha0_source"
    return "unknown"


def _condition_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["condition"])].append(record)
    return {condition: _summarize_group(items) for condition, items in sorted(grouped.items())}


def _summarize_group(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return float(statistics.fmean(float(record[key]) for record in records)) if records else 0.0

    stage_cvs = {
        stage: float(statistics.fmean(record["stage_cv"][stage] for record in records)) if records else 0.0
        for stage in CHAIN_STAGE_NAMES
    }
    fixed_role_separation_count = sum(1 for record in records if record["fixed_role_behavior_separation"])
    dominant_stage_counts: Counter[str] = Counter()
    for record in records:
        dominant_stage_counts.update(record["dominant_stage_counts"])
    return {
        "records": len(records),
        "mean_specialized_agent_count": mean("specialized_agent_count"),
        "mean_agents_with_all_chain_stages": mean("agents_with_all_chain_stages"),
        "mean_max_stage_share": mean("mean_max_stage_share"),
        "mean_max_fixed_role_stage_share_gap": mean("max_fixed_role_stage_share_gap"),
        "fixed_role_behavior_separation_count": fixed_role_separation_count,
        "dominant_stage_counts_total": dict(dominant_stage_counts),
        "mean_stage_cv": stage_cvs,
    }


if __name__ == "__main__":
    raise SystemExit(main())
