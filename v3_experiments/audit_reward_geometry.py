"""Audit provenance for the canonical Tribal reward-geometry reruns.

The audit intentionally treats checked-in ``results_reward_*.json`` files as
non-canonical unless they satisfy the fixed 3-way role-probe contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_experiments.canonical_reward_geometry import (
    audit_fixed_role_probe_records,
    summarize_probe_audit,
)


DEFAULT_BRANCHES = (
    "main",
    "origin/main",
    "origin/tashapais-gpu0-smac-experiments",
    "origin/tashapais-gpu1-craftax-experiments",
    "origin/tashapais-gpu2-mettagrid-experiments",
    "origin/tashapais-gpu3-ablation-experiments",
)
INTERESTING_PATH_PARTS = (
    "paper_exp_reward_type",
    "paper_exp_tribal",
    "results_reward",
    "results_tribal",
    "launch_reward",
    "launch_sepenc",
    "run_smac",
)
WANDB_NAME_HINTS = ("tribal", "reward", "mixed80", "shared", "individual", "geometry")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--entity", default="tashapais")
    parser.add_argument("--project", default="representation-collapse")
    parser.add_argument("--include-wandb", action="store_true", help="Query W&B in addition to local/git metadata")
    parser.add_argument("--max-wandb-runs", type=int, default=250)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    report: dict[str, Any] = {
        "purpose": "canonical_tribal_reward_geometry_provenance_audit",
        "repo_root": str(repo_root),
        "local_reward_result_audit": audit_local_reward_results(repo_root),
        "git_provenance_candidates": audit_git_branches(repo_root),
        "wandb_runs": [],
    }
    if args.include_wandb:
        report["wandb_runs"] = audit_wandb_runs(args.entity, args.project, args.max_wandb_runs)

    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


def audit_local_reward_results(repo_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted((repo_root / "v3_experiments").glob("results_reward_*.json")):
        records = _load_json_records(path)
        issues = audit_fixed_role_probe_records(records)
        results.append(
            {
                "path": str(path.relative_to(repo_root)),
                "record_count": len(records),
                "issue_summary": summarize_probe_audit(issues),
                "issues": [asdict(issue) for issue in issues[:20]],
                "canonical_role_probe_compatible": not issues,
            }
        )
    return results


def audit_git_branches(repo_root: Path) -> list[dict[str, Any]]:
    branches: list[dict[str, Any]] = []
    for branch in DEFAULT_BRANCHES:
        if not _git_exists(repo_root, branch):
            continue
        files = _git_lines(repo_root, "ls-tree", "-r", "--name-only", branch)
        interesting = [
            path for path in files
            if path.startswith("v3_experiments/") and any(part in path for part in INTERESTING_PATH_PARTS)
        ]
        branches.append(
            {
                "branch": branch,
                "head": _git_text(repo_root, "rev-parse", "--short", branch),
                "recent_commits": _git_lines(repo_root, "log", "--oneline", "-5", branch),
                "interesting_files": interesting[:200],
            }
        )
    return branches


def audit_wandb_runs(entity: str, project: str, max_runs: int) -> list[dict[str, Any]]:
    try:
        import wandb
    except ImportError as exc:
        return [{"error": f"wandb is not importable: {exc}"}]

    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
    except Exception as exc:
        return [{"error": f"failed to query {entity}/{project}: {exc}"}]

    matches: list[dict[str, Any]] = []
    try:
        for idx, run in enumerate(runs):
            if idx >= max_runs:
                break
            name = run.name or ""
            config = dict(run.config or {})
            if not _looks_relevant(name, config):
                continue
            matches.append(
                {
                    "id": run.id,
                    "name": name,
                    "state": run.state,
                    "created_at": str(run.created_at),
                    "url": run.url,
                    "config_subset": {
                        key: config[key]
                        for key in sorted(config)
                        if key in {"seed", "num_agents", "num_teams", "reward_type", "shared_frac", "num_seeds"}
                    },
                    "summary_subset": {
                        key: run.summary.get(key)
                        for key in sorted(run.summary.keys())
                        if key.startswith("final/") or key in {"probe_accuracy", "probe_chance", "effrank_per_agent"}
                    },
                }
            )
    except Exception as exc:
        return [{"error": f"failed while reading {entity}/{project}: {exc}"}]
    return matches


def _looks_relevant(name: str, config: dict[str, Any]) -> bool:
    lowered = name.lower()
    if any(hint in lowered for hint in WANDB_NAME_HINTS):
        return True
    return any(key in config for key in ("shared_frac", "reward_type", "num_teams"))


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _git_exists(repo_root: Path, ref: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", ref],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _git_text(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo_root), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _git_lines(repo_root: Path, *args: str) -> list[str]:
    text = _git_text(repo_root, *args)
    return [line for line in text.splitlines() if line]


if __name__ == "__main__":
    main()
