"""Export Tribal Village representation clouds and old-style PCA plots.

This script reloads canonical reward-geometry checkpoints, runs deterministic
evaluation rollouts, stores embeddings/logits/actions as ``.npz`` artifacts,
and regenerates the paper's PCA role-geometry plot contract:

- one panel per reward condition;
- x/y axes are PC1/PC2 with explained variance;
- point colors are fixed ``agent_id % 3`` roles;
- large outlined markers are per-agent embedding means;
- panel subtitles report EffRank/n, D_act, JS, and role-probe accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_experiments.canonical_reward_geometry import (  # noqa: E402
    effective_rank,
    fixed_role_probe_accuracy,
    js_action_diversity,
    ordered_kl_action_diversity,
    role_labels,
)
from v3_experiments.train_canonical_reward_geometry import (  # noqa: E402
    ActorCritic,
    RunnerConfig,
    _action_mask_tensor,
    _agent_ids,
    _make_env,
    _torch_load_checkpoint,
    _validate_env_contract,
    parse_args as parse_train_args,
)

ROLE_COLORS = ("#e74c3c", "#2ecc71", "#3498db")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True, help="Canonical result JSON files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-trials", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=240)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint-prefix-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite checkpoint paths from JSON records before loading.",
    )
    parser.add_argument("--max-points-per-condition", type=int, default=6000)
    parser.add_argument("--plot-filename", default="tribal_pca_role_geometry")
    parser.add_argument("--plot-title", default="Tribal Village Role Geometry")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_maps = _parse_prefix_maps(args.checkpoint_prefix_map)

    artifacts = []
    for record_path in _expand_record_paths(args.records):
        record = json.loads(record_path.read_text())
        artifact = export_record(
            record,
            record_path,
            output_dir,
            prefix_maps,
            eval_trials=args.eval_trials,
            eval_steps=args.eval_steps,
            device_name=args.device,
        )
        artifacts.append(artifact)

    summary = {
        "schema_version": "tribal_representation_artifacts_v1",
        "records": [artifact["summary"] for artifact in artifacts],
    }
    (output_dir / "representation_artifacts_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    plot_conditions(
        artifacts,
        output_dir,
        filename=args.plot_filename,
        title=args.plot_title,
        max_points_per_condition=args.max_points_per_condition,
    )
    print(json.dumps({"output_dir": str(output_dir), "records": len(artifacts)}, indent=2))
    return 0


def export_record(
    record: dict[str, Any],
    record_path: Path,
    output_dir: Path,
    prefix_maps: list[tuple[str, str]],
    *,
    eval_trials: int,
    eval_steps: int,
    device_name: str,
) -> dict[str, Any]:
    checkpoint_path = _resolve_checkpoint_path(str(record["checkpoint_path"]), prefix_maps)
    device = torch.device(device_name)
    checkpoint = _torch_load_checkpoint(checkpoint_path, device)
    config = _config_from_checkpoint(checkpoint, record, device_name, eval_trials, eval_steps)

    env = _make_env(config)
    try:
        _validate_env_contract(env, config)
        policy = ActorCritic(
            env.obs_shape,
            env.action_space_size,
            n_agents=env.num_agents,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            separate_encoders=config.separate_encoders,
        ).to(device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        rollout = _collect_deterministic_rollout(policy, env, config, device, eval_trials, eval_steps)
    finally:
        env.close()

    embeddings = rollout["embeddings"]
    logits = rollout["logits"]
    flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    flat_logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
    probe_acc, probe_meta = fixed_role_probe_accuracy(embeddings.reshape(-1, embeddings.shape[-2], embeddings.shape[-1]))
    metrics = {
        "effrank_per_agent": effective_rank(flat_embeddings) / embeddings.shape[-2],
        "d_act_ordered_kl": ordered_kl_action_diversity(flat_logits),
        "d_act_js": js_action_diversity(flat_logits),
        "role_probe_acc": probe_acc,
        "role_probe_chance": probe_meta["chance"],
    }

    label = _condition_label(record)
    stem = _artifact_stem(record)
    npz_path = output_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        logits=logits,
        actions=rollout["actions"],
        role_labels=role_labels(embeddings.shape[-2]),
        record_json=json.dumps(record, sort_keys=True),
        metrics_json=json.dumps(metrics, sort_keys=True),
    )

    summary = {
        "record_path": str(record_path),
        "checkpoint_path": str(checkpoint_path),
        "artifact_path": str(npz_path),
        "label": label,
        "shared_frac": record.get("shared_frac"),
        "seed": record.get("seed"),
        **metrics,
    }
    return {
        "label": label,
        "record": record,
        "summary": summary,
        "embeddings": embeddings,
        "logits": logits,
        "metrics": metrics,
        "artifact_path": npz_path,
    }


def _collect_deterministic_rollout(
    policy: ActorCritic,
    env: Any,
    config: RunnerConfig,
    device: torch.device,
    eval_trials: int,
    eval_steps: int,
) -> dict[str, np.ndarray]:
    policy.eval()
    all_embeddings = []
    all_logits = []
    all_actions = []
    with torch.no_grad():
        for trial in range(eval_trials):
            obs = env.reset(seed=config.seed + 700_000 + trial)
            embeddings = []
            logits = []
            actions_out = []
            for _step in range(eval_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action_mask = _action_mask_tensor(env, config, device)
                actions, _logprob, _entropy, _value, emb, logit = policy.get_action_and_value(
                    obs_t,
                    agent_ids=_agent_ids(env.num_agents, device) if config.separate_encoders else None,
                    action_mask=action_mask,
                    deterministic=True,
                )
                actions_np = actions.cpu().numpy().astype(np.int64)
                obs, _rewards, done = env.step(actions_np)
                embeddings.append(emb.cpu().numpy())
                logits.append(logit.cpu().numpy())
                actions_out.append(actions_np)
                if done:
                    obs = env.reset(seed=config.seed + 710_000 + trial * eval_steps + _step)
            all_embeddings.append(np.stack(embeddings))
            all_logits.append(np.stack(logits))
            all_actions.append(np.stack(actions_out))
    policy.train()
    return {
        "embeddings": np.stack(all_embeddings),
        "logits": np.stack(all_logits),
        "actions": np.stack(all_actions),
    }


def plot_conditions(
    artifacts: list[dict[str, Any]],
    output_dir: Path,
    *,
    filename: str,
    title: str,
    max_points_per_condition: int,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        grouped.setdefault(str(artifact["label"]), []).append(artifact)
    labels = sorted(grouped, key=_label_sort_key)
    if not labels:
        return

    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 5), squeeze=False)
    for panel_index, label in enumerate(labels):
        ax = axes[0][panel_index]
        condition_artifacts = grouped[label]
        embeddings = np.concatenate([item["embeddings"].reshape(-1, *item["embeddings"].shape[-2:]) for item in condition_artifacts], axis=0)
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        pca = PCA(n_components=2)
        projected = pca.fit_transform(flat).reshape(embeddings.shape[0], embeddings.shape[1], 2)
        labels_by_agent = role_labels(embeddings.shape[1])
        sample_idx = _sample_indices(projected.shape[0], max_points_per_condition // embeddings.shape[1])
        for role_id, color in enumerate(ROLE_COLORS):
            agent_ids = np.flatnonzero(labels_by_agent == role_id)
            points = projected[sample_idx][:, agent_ids, :].reshape(-1, 2)
            ax.scatter(points[:, 0], points[:, 1], s=9, c=color, alpha=0.25, label=f"Role {role_id}")
        per_agent_means = projected.mean(axis=0)
        for agent_id, point in enumerate(per_agent_means):
            color = ROLE_COLORS[int(labels_by_agent[agent_id]) % len(ROLE_COLORS)]
            ax.scatter(point[0], point[1], s=100, c=color, edgecolors="black", linewidths=1.1, zorder=5)

        metrics = _mean_metrics(condition_artifacts)
        ax.set_title(
            f"{label}\n"
            f"EffRank/n={metrics['effrank_per_agent']:.3f}  "
            f"D_act={metrics['d_act_ordered_kl']:.3f}  "
            f"JS={metrics['d_act_js']:.3f}  "
            f"Probe={metrics['role_probe_acc']:.3f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.grid(alpha=0.2)
        if panel_index == 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{filename}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _config_from_checkpoint(
    checkpoint: dict[str, Any],
    record: dict[str, Any],
    device_name: str,
    eval_trials: int,
    eval_steps: int,
) -> RunnerConfig:
    raw_config = dict(checkpoint.get("config") or {})
    defaults = vars(
        parse_train_args(
            [
                "--shared-frac",
                str(raw_config.get("shared_frac", record.get("shared_frac", 0.0))),
                "--seed",
                str(raw_config.get("seed", record.get("seed", 0))),
                "--output",
                "unused.json",
            ]
        )
    )
    field_names = {field.name for field in fields(RunnerConfig)}
    merged = {key: value for key, value in defaults.items() if key in field_names}
    merged.update({key: value for key, value in raw_config.items() if key in field_names})
    merged["eval_trials"] = eval_trials
    merged["eval_steps"] = eval_steps
    merged["device"] = device_name
    return RunnerConfig(**merged)


def _parse_prefix_maps(values: list[str]) -> list[tuple[str, str]]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--checkpoint-prefix-map must be OLD=NEW, got {value!r}")
        old, new = value.split("=", 1)
        out.append((old, new))
    return out


def _resolve_checkpoint_path(raw_path: str, prefix_maps: list[tuple[str, str]]) -> Path:
    path = raw_path
    for old, new in prefix_maps:
        if path.startswith(old):
            path = new + path[len(old) :]
            break
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved


def _expand_record_paths(values: list[str]) -> list[Path]:
    paths = [Path(value) for value in values]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("record JSON not found: " + ", ".join(missing))
    return paths


def _condition_label(record: dict[str, Any]) -> str:
    shared_frac = record.get("shared_frac")
    if shared_frac is not None:
        return f"alpha{str(shared_frac).replace('.', 'p')}"
    return str(record.get("condition_group") or record.get("run_name") or "condition")


def _label_sort_key(label: str) -> tuple[int, str]:
    if label.startswith("alpha"):
        try:
            return (0, f"{float(label[5:].replace('p', '.')):08.4f}")
        except ValueError:
            pass
    return (1, label)


def _artifact_stem(record: dict[str, Any]) -> str:
    run_name = str(record.get("run_name") or "record")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_name)
    return safe[:160]


def _sample_indices(total: int, requested: int) -> np.ndarray:
    if total <= max(1, requested):
        return np.arange(total)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(total, size=max(1, requested), replace=False))


def _mean_metrics(artifacts: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("effrank_per_agent", "d_act_ordered_kl", "d_act_js", "role_probe_acc")
    return {
        key: float(np.mean([float(artifact["metrics"][key]) for artifact in artifacts]))
        for key in keys
    }


if __name__ == "__main__":
    raise SystemExit(main())
