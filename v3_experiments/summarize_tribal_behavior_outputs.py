"""Summarize Tribal Village behavior rollout outputs into gate-friendly rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_experiments.tribal_behavior import validate_behavior_record, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = []
    for path in _collect_paths(args):
        record = json.loads(path.read_text())
        issues = validate_behavior_record(record)
        if issues:
            raise RuntimeError(f"{path}: " + "; ".join(issues))
        records.append((path, record))

    summary = {
        "schema_version": "tribal_behavior_summary_v1",
        "rows": [summarize_record(path, record) for path, record in records],
    }
    if args.output:
        write_json(args.output, summary)
    print(_format_table(summary["rows"]))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def summarize_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    aggregate = record["aggregate_behavior_metrics"]
    action_attempts = aggregate["action_attempts_by_verb_total"]
    invalid = aggregate["invalid_actions_by_verb_total"].get("invalid", 0)
    total_attempts = sum(int(value) for value in action_attempts.values())
    invalid_fraction = invalid / total_attempts if total_attempts else 0.0

    resource_events = sum(aggregate["resource_pickups_by_type_total"].values())
    craft_events = sum(aggregate["crafting_outputs_by_type_total"].values())
    deposit_events = sum(aggregate["deposits_by_type_total"].values())
    handoff_events = sum(aggregate["handoffs_by_type_total"].values())
    combat_events = sum(aggregate["combat_events_total"].values())
    task_events = aggregate["task_event_count_mean"]

    flags = []
    if task_events <= 0:
        flags.append("no_task_events")
    if aggregate["unique_joint_action_count_mean"] <= 1:
        flags.append("single_joint_action")
    if aggregate["noop_fraction_mean"] >= 0.95:
        flags.append("mostly_noop")
    if invalid_fraction >= 0.5:
        flags.append("mostly_invalid")

    return {
        "path": str(path),
        "policy": record["policy"],
        "environment_backend": record["environment_backend"],
        "episodes": record["episodes"],
        "steps_per_episode": record["steps_per_episode"],
        "raw_reward_mean": aggregate["raw_env_reward_total_mean"],
        "task_event_count_mean": task_events,
        "resource_events": resource_events,
        "craft_events": craft_events,
        "deposit_events": deposit_events,
        "handoff_events": handoff_events,
        "combat_events": combat_events,
        "unique_actions_mean": aggregate["unique_action_count_mean"],
        "unique_joint_actions_mean": aggregate["unique_joint_action_count_mean"],
        "noop_fraction_mean": aggregate["noop_fraction_mean"],
        "invalid_fraction": invalid_fraction,
        "flags": flags,
    }


def _collect_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.paths)
    if args.results_dir is not None:
        paths.extend(sorted(args.results_dir.rglob("rollout_metrics.json")))
    if not paths:
        raise SystemExit("pass rollout JSON paths or --results-dir")
    return paths


def _format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no rows"
    headers = (
        "policy",
        "reward",
        "task",
        "resource",
        "craft",
        "deposit",
        "combat",
        "uniq_joint",
        "noop",
        "invalid",
        "flags",
    )
    table_rows = [
        (
            row["policy"],
            f"{row['raw_reward_mean']:.3f}",
            f"{row['task_event_count_mean']:.1f}",
            str(row["resource_events"]),
            str(row["craft_events"]),
            str(row["deposit_events"]),
            str(row["combat_events"]),
            f"{row['unique_joint_actions_mean']:.1f}",
            f"{row['noop_fraction_mean']:.2f}",
            f"{row['invalid_fraction']:.2f}",
            ",".join(row["flags"]) or "ok",
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for row in table_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True))]
    lines.append("  ".join("-" * width for width in widths))
    for row in table_rows:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
