"""Audit the trained Tribal Village package against the Coworld repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_experiments.tribal_behavior import ACTION_ARGUMENT_COUNT, action_verb_names, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = audit_tribal_versions(Path(args.metta_root), Path(args.coworld_root))
    output = Path(args.output)
    write_json(output, record)
    print(json.dumps({"output": str(output), "differences": len(record["differences"])}, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metta-root", default=str(REPO_ROOT))
    parser.add_argument("--coworld-root", default="/Users/relh/Code/coworld-tribal-village")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def audit_tribal_versions(metta_root: Path, coworld_root: Path) -> dict[str, Any]:
    trained_root = metta_root / "packages" / "tribal_village"
    trained = audit_package_contract(
        trained_root,
        label="trained_metta_package",
        canonical_reward_geometry=True,
    )
    coworld = audit_package_contract(
        coworld_root,
        label="metta_ai_coworld_repo",
        canonical_reward_geometry=False,
    )
    return {
        "schema_version": "tribal_version_audit_v1",
        "trained_metta_package": trained,
        "metta_ai_coworld_repo": coworld,
        "differences": compare_contracts(trained, coworld),
        "compatibility": {
            "checkpoint_compatible": trained["action_space_size"] == coworld["action_space_size"]
            and trained["num_agents"] == coworld["num_agents"],
            "reason": (
                "Existing checkpoints are only directly compatible when agent count and action space match."
            ),
        },
    }


def audit_package_contract(root: Path, *, label: str, canonical_reward_geometry: bool) -> dict[str, Any]:
    common_path = root / "src" / "common.nim"
    environment_path = root / "src" / "environment.nim"
    constants = {
        **_extract_constants(common_path),
        **_extract_constants(environment_path),
    }

    action_verbs = _int_constant(constants, "ActionVerbCount")
    action_args = _int_constant(constants, "ActionArgumentCount", ACTION_ARGUMENT_COUNT)
    observation_layers = _int_constant(constants, "ObservationLayers")
    observation_width = _int_constant(constants, "ObservationWidth")
    observation_height = _int_constant(constants, "ObservationHeight")

    if canonical_reward_geometry and _has_canonical_reward_geometry(environment_path):
        num_teams = 1
        agents_per_team = 12
        map_room_width = 76
        map_room_height = 76
    else:
        num_teams = _int_constant(constants, "MapRoomObjectsHouses")
        agents_per_team = _int_constant(constants, "MapAgentsPerHouse")
        map_room_width = _int_constant(constants, "MapRoomWidth")
        map_room_height = _int_constant(constants, "MapRoomHeight")

    map_border = _int_constant(constants, "MapBorder", 0)
    map_room_border = _int_constant(constants, "MapRoomBorder", 0)
    layout_x = _int_constant(constants, "MapLayoutRoomsX", 1)
    layout_y = _int_constant(constants, "MapLayoutRoomsY", 1)
    map_width = layout_x * (map_room_width + map_room_border) + map_border
    map_height = layout_y * (map_room_height + map_room_border) + map_border
    action_space_size = action_verbs * action_args

    return {
        "label": label,
        "root": str(root),
        "exists": root.exists(),
        "git_sha": _git_sha(root),
        "canonical_reward_geometry_define": bool(
            canonical_reward_geometry and _has_canonical_reward_geometry(environment_path)
        ),
        "num_teams": num_teams,
        "agents_per_team": agents_per_team,
        "num_agents": num_teams * agents_per_team,
        "map_width": map_width,
        "map_height": map_height,
        "observation_shape": [observation_layers, observation_width, observation_height],
        "action_verb_count": action_verbs,
        "action_argument_count": action_args,
        "action_space_size": action_space_size,
        "action_verb_names": action_verb_names(action_space_size, argument_count=action_args),
        "has_coworld_runtime": (root / "tribal_village_env" / "coworld" / "server.py").exists(),
        "has_rules_doc": (root / "docs" / "rules.md").exists(),
        "has_play_doc": (root / "docs" / "play_tribal_village.md").exists(),
        "has_equipment_module": (root / "src" / "equipment.nim").exists(),
        "has_bridge_asset": (root / "data" / "objects" / "bridge.png").exists(),
        "has_fertile_asset": (root / "data" / "objects" / "fertile.png").exists(),
        "source_files": {
            "common": str(common_path),
            "environment": str(environment_path),
        },
    }


def compare_contracts(trained: dict[str, Any], coworld: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "num_agents",
        "num_teams",
        "agents_per_team",
        "map_width",
        "map_height",
        "observation_shape",
        "action_verb_count",
        "action_space_size",
        "has_coworld_runtime",
        "has_equipment_module",
        "has_bridge_asset",
        "has_fertile_asset",
    )
    differences = []
    for key in keys:
        if trained.get(key) != coworld.get(key):
            differences.append(
                {
                    "field": key,
                    "trained_metta_package": trained.get(key),
                    "metta_ai_coworld_repo": coworld.get(key),
                }
            )
    return differences


def _extract_constants(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    constants: dict[str, str] = {}
    pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\*\s*=\s*([^#\n]+)")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match and match.group(1) not in constants:
            constants[match.group(1)] = match.group(2).strip()
    return constants


def _int_constant(constants: dict[str, str], key: str, default: int | None = None) -> int:
    if key not in constants:
        if default is not None:
            return default
        raise KeyError(f"missing Nim constant {key}")
    raw = constants[key].strip()
    match = re.match(r"^-?\d+", raw)
    if not match:
        if default is not None:
            return default
        raise ValueError(f"constant {key} is not an integer literal: {raw!r}")
    return int(match.group(0))


def _has_canonical_reward_geometry(path: Path) -> bool:
    return path.exists() and "canonicalRewardGeometry" in path.read_text()


def _git_sha(root: Path) -> str | None:
    if not root.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
