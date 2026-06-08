"""Validate canonical Tribal Village reward-geometry result JSON files."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_experiments.canonical_reward_geometry import (  # noqa: E402
    CANONICAL_NUM_AGENTS,
    ROLE_PROBE_CV,
    role_labels,
    role_probe_chance,
)

SCHEMA_VERSION = "canonical_reward_geometry_v1"
FULL_RUN_AGENT_STEPS = 4_000_000
FULL_RUN_EVAL_TRIALS = 10
REQUIRED_FIELDS = (
    "schema_version",
    "condition_group",
    "shared_frac",
    "seed",
    "num_agents",
    "num_teams",
    "map_width",
    "map_height",
    "role_assignment",
    "role_names",
    "role_labels",
    "role_shaping_enabled",
    "separate_encoders",
    "total_agent_steps",
    "eval_trials",
    "metta_git_sha",
    "tribal_village_git_sha",
    "tribal_village_build_id",
    "command",
    "wandb_entity",
    "wandb_project",
    "wandb_run_id",
    "wandb_url",
    "checkpoint_path",
    "obs_shape",
    "action_space_size",
    "metric_schema",
    "effrank_per_agent",
    "d_act_ordered_kl",
    "d_act_js",
    "role_probe_acc",
    "role_probe_chance",
    "role_probe_cv",
    "eval_trial_metrics",
)
METRIC_KEYS = ("effrank_per_agent", "d_act_ordered_kl", "d_act_js", "role_probe_acc")
CONDITION_GROUPS = ("primary", "no_role_shaping", "separate_encoders")


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    field: str
    message: str


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    issues: list[ValidationIssue] = []
    for path_str in args.paths:
        path = Path(path_str)
        try:
            record = json.loads(path.read_text())
        except Exception as exc:
            issues.append(ValidationIssue(str(path), "json", f"could not read JSON: {exc}"))
            continue
        issues.extend(
            validate_record(
                record,
                path=path,
                allow_smoke=args.allow_smoke,
                allow_missing_wandb=args.allow_missing_wandb,
                allow_missing_checkpoint=args.allow_missing_checkpoint,
            )
        )

    if issues:
        for issue in issues:
            print(f"{issue.path}: {issue.field}: {issue.message}", file=sys.stderr)
        return 1
    print(f"validated {len(args.paths)} canonical reward-geometry result file(s)")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--allow-smoke", action="store_true", help="permit mock backend, short runs, and short evals")
    parser.add_argument("--allow-missing-wandb", action="store_true", help="permit missing W&B run id/url")
    parser.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        help="permit checkpoint_path that is not local",
    )
    return parser.parse_args(argv)


def validate_record(
    record: Mapping[str, Any],
    *,
    path: Path | None = None,
    allow_smoke: bool = False,
    allow_missing_wandb: bool = False,
    allow_missing_checkpoint: bool = False,
) -> list[ValidationIssue]:
    path_label = str(path) if path is not None else "<record>"
    issues: list[ValidationIssue] = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            issues.append(ValidationIssue(path_label, field, "missing required field"))
    if issues:
        return issues

    _expect_equal(issues, path_label, record, "schema_version", SCHEMA_VERSION)
    _expect_equal(issues, path_label, record, "num_agents", CANONICAL_NUM_AGENTS)
    _expect_equal(issues, path_label, record, "num_teams", 1)
    _expect_equal(issues, path_label, record, "map_width", 80)
    _expect_equal(issues, path_label, record, "map_height", 80)
    _expect_equal(issues, path_label, record, "role_assignment", "agent_id % 3")
    _expect_equal(issues, path_label, record, "role_probe_cv", ROLE_PROBE_CV)

    if record["condition_group"] not in CONDITION_GROUPS and not allow_smoke:
        issues.append(ValidationIssue(path_label, "condition_group", f"must be one of {CONDITION_GROUPS}"))
    if record["condition_group"] == "primary":
        _expect_equal(issues, path_label, record, "role_shaping_enabled", True)
        _expect_equal(issues, path_label, record, "separate_encoders", False)
    if record["condition_group"] == "no_role_shaping":
        _expect_equal(issues, path_label, record, "role_shaping_enabled", False)
    if record["condition_group"] == "separate_encoders":
        _expect_equal(issues, path_label, record, "separate_encoders", True)

    if list(record["role_labels"]) != role_labels(CANONICAL_NUM_AGENTS).astype(int).tolist():
        issues.append(ValidationIssue(path_label, "role_labels", "must match interleaved agent_id % 3 labels"))
    if not isinstance(record["role_names"], list) or len(record["role_names"]) != 3:
        issues.append(ValidationIssue(path_label, "role_names", "must list the three role names"))

    _expect_float_in_range(issues, path_label, record, "shared_frac", minimum=0.0, maximum=1.0)
    _expect_float_in_range(issues, path_label, record, "role_probe_acc", minimum=0.0, maximum=1.0)
    probe_chance = _float_or_issue(issues, path_label, record, "role_probe_chance")
    if probe_chance is not None and not math.isclose(probe_chance, role_probe_chance(), rel_tol=0.0, abs_tol=1e-12):
        issues.append(ValidationIssue(path_label, "role_probe_chance", "must be exactly 1/3 for the fixed-role probe"))
    for key in ("effrank_per_agent", "d_act_ordered_kl", "d_act_js"):
        _expect_finite_nonnegative(issues, path_label, record, key)

    eval_trials = _int_or_issue(issues, path_label, record, "eval_trials")
    if not allow_smoke:
        _expect_minimum(issues, path_label, record, "total_agent_steps", FULL_RUN_AGENT_STEPS)
        _expect_equal(issues, path_label, record, "eval_trials", FULL_RUN_EVAL_TRIALS)
        if record.get("environment_backend") != "tribal":
            issues.append(ValidationIssue(path_label, "environment_backend", "full validation requires tribal backend"))
    elif eval_trials is not None and eval_trials < 1:
        issues.append(
            ValidationIssue(path_label, "eval_trials", "smoke validation still requires at least one eval trial")
        )

    eval_trial_metrics = record["eval_trial_metrics"]
    if not isinstance(eval_trial_metrics, list):
        issues.append(ValidationIssue(path_label, "eval_trial_metrics", "must be a list"))
    elif eval_trials is not None and len(eval_trial_metrics) != eval_trials:
        issues.append(ValidationIssue(path_label, "eval_trial_metrics", "length must equal eval_trials"))
    else:
        for idx, trial in enumerate(eval_trial_metrics):
            if not isinstance(trial, Mapping):
                issues.append(ValidationIssue(path_label, f"eval_trial_metrics[{idx}]", "must be an object"))
                continue
            for key in METRIC_KEYS:
                _expect_finite_nonnegative(
                    issues,
                    path_label,
                    trial,
                    f"eval_trial_metrics[{idx}].{key}",
                    source_key=key,
                )

    metric_schema = record["metric_schema"]
    if not isinstance(metric_schema, Mapping):
        issues.append(ValidationIssue(path_label, "metric_schema", "must be an object"))
    else:
        for key in (*METRIC_KEYS, "role_probe_chance"):
            if key not in metric_schema:
                issues.append(ValidationIssue(path_label, "metric_schema", f"missing metric description for {key}"))

    for key in ("metta_git_sha", "tribal_village_git_sha"):
        if not _is_git_sha(record[key]):
            issues.append(ValidationIssue(path_label, key, "must be a full git SHA"))
    if "train_canonical_reward_geometry.py" not in str(record["command"]):
        issues.append(ValidationIssue(path_label, "command", "must invoke the canonical runner"))
    if not isinstance(record["obs_shape"], list) or not record["obs_shape"]:
        issues.append(ValidationIssue(path_label, "obs_shape", "must be a non-empty list"))
    _expect_minimum(issues, path_label, record, "action_space_size", 1)

    if not allow_missing_wandb and not allow_smoke:
        for key in ("wandb_run_id", "wandb_url"):
            if not record[key]:
                issues.append(ValidationIssue(path_label, key, "must be present for full runs"))
    if not allow_missing_checkpoint:
        checkpoint_path = _resolve_checkpoint_path(str(record["checkpoint_path"]), path)
        if checkpoint_path is None or not checkpoint_path.exists():
            issues.append(ValidationIssue(path_label, "checkpoint_path", "checkpoint does not exist locally"))

    return issues


def _expect_equal(
    issues: list[ValidationIssue],
    path_label: str,
    record: Mapping[str, Any],
    field: str,
    expected: object,
) -> None:
    if record[field] != expected:
        issues.append(ValidationIssue(path_label, field, f"expected {expected!r}, got {record[field]!r}"))


def _expect_minimum(
    issues: list[ValidationIssue],
    path_label: str,
    record: Mapping[str, Any],
    field: str,
    minimum: int,
) -> None:
    value = _int_or_issue(issues, path_label, record, field)
    if value is not None and value < minimum:
        issues.append(ValidationIssue(path_label, field, f"must be >= {minimum}"))


def _expect_float_in_range(
    issues: list[ValidationIssue],
    path_label: str,
    record: Mapping[str, Any],
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> None:
    value = _float_or_issue(issues, path_label, record, field)
    if value is None:
        return
    if not math.isfinite(value) or value < minimum or value > maximum:
        issues.append(ValidationIssue(path_label, field, f"must be finite and in [{minimum}, {maximum}]"))


def _expect_finite_nonnegative(
    issues: list[ValidationIssue],
    path_label: str,
    record: Mapping[str, Any],
    field: str,
    *,
    source_key: str | None = None,
) -> None:
    key = source_key or field
    value = _float_or_issue(issues, path_label, record, key, issue_field=field)
    if value is None:
        return
    if not math.isfinite(value) or value < 0.0:
        issues.append(ValidationIssue(path_label, field, "must be finite and non-negative"))


def _int_or_issue(
    issues: list[ValidationIssue],
    path_label: str,
    record: Mapping[str, Any],
    field: str,
) -> int | None:
    try:
        return int(record[field])
    except (TypeError, ValueError):
        issues.append(ValidationIssue(path_label, field, "must be an integer"))
        return None


def _float_or_issue(
    issues: list[ValidationIssue],
    path_label: str,
    record: Mapping[str, Any],
    field: str,
    *,
    issue_field: str | None = None,
) -> float | None:
    try:
        return float(record[field])
    except (TypeError, ValueError):
        issues.append(ValidationIssue(path_label, issue_field or field, "must be numeric"))
        return None


def _is_git_sha(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _resolve_checkpoint_path(value: str, result_path: Path | None) -> Path | None:
    checkpoint_path = Path(value)
    if checkpoint_path.exists() or checkpoint_path.is_absolute() or result_path is None:
        return checkpoint_path
    return result_path.parent / checkpoint_path


if __name__ == "__main__":
    raise SystemExit(main())
