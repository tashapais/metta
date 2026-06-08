"""Validate Tribal Village behavior rollout JSON outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_experiments.tribal_behavior import validate_behavior_record  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _collect_paths(args)
    all_issues = []
    for path in paths:
        record = json.loads(path.read_text())
        for issue in validate_behavior_record(record):
            all_issues.append(f"{path}: {issue}")

    if all_issues:
        for issue in all_issues:
            print(issue, file=sys.stderr)
        return 1

    print(f"validated {len(paths)} Tribal behavior rollout file(s)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--results-dir", type=Path)
    return parser.parse_args(argv)


def _collect_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.paths)
    if args.results_dir is not None:
        paths.extend(sorted(args.results_dir.rglob("rollout_metrics.json")))
    if not paths:
        raise SystemExit("pass rollout JSON paths or --results-dir")
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
