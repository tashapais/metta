# Canonical Tribal Reward-Geometry Reruns

This is the canonical protocol for the paper section about MAPPO representation
geometry under shared rewards. It is separate from the later
`results_reward_*.json` stream, which uses binary top/bottom return probes and
must not be cited as fixed-role Tribal Village evidence.

## Research Question

Does increasing reward mixing in 12-agent Tribal Village reduce role-separable
geometry in a shared MAPPO encoder?

The pilot result in the Overleaf checkout used one training seed and five eval
trials. Treat it as provisional until the reruns below replace it.

## Canonical Protocol

- Environment: Tribal Village, 1 team x 12 agents, 80x80 map.
- Policy: one shared MAPPO policy; no role input and no agent-specific parameters.
- Roles: `role = agent_id % 3`, balanced as four gatherers, four explorers, four guardians.
- Reward mixing: `r_i_alpha = (1 - alpha) * r_i_ind + alpha * mean_j(r_j_ind)`.
- Primary conditions: `alpha in {0.0, 0.8, 1.0}`.
- Primary seeds: 5 independent training seeds per condition.
- Evaluation: 10 eval trials per final checkpoint.
- Table uncertainty: mean and std across training seeds, not eval trials.

The machine-readable version is
`v3_experiments/canonical_reward_geometry_protocol.json`.

## Canonical Metrics

- `effrank_per_agent`: effective rank over flattened eval embeddings divided by 12.
- `d_act_ordered_kl`: ordered off-diagonal KL averaged over `n * (n - 1)` agent pairs.
- `d_act_js`: Jensen-Shannon robustness check, not the primary paper metric.
- `role_probe_acc`: 3-way probe with labels from `agent_id % 3`; chance is `1/3`.

## Provenance Audit

Run the local/git audit:

```bash
uv run python v3_experiments/audit_reward_geometry.py --repo-root .
```

Include W&B metadata when authenticated:

```bash
uv run python v3_experiments/audit_reward_geometry.py \
  --repo-root . \
  --include-wandb \
  --output v3_experiments/canonical_reward_geometry_provenance.json
```

Expected audit behavior:

- `v3_experiments/results_reward_*.json` should be flagged as non-canonical for
  the fixed-role probe if `probe_chance != 1/3`.
- W&B candidates should be inspected for git SHA, checkpoint artifacts,
  environment dimensions, seed, command, and metric definitions.

## Focused Reruns

After recovering or reconstructing the closest Tribal Village checkout:

- Primary: `alpha in {0.0, 0.8, 1.0}`, seeds `0..4`, 4M agent-steps each.
- No-role-shaping mechanism check: `alpha in {0.0, 1.0}`, seeds `0..2`.
- Separate-encoder mechanism check: `alpha in {0.0, 1.0}`, seeds `0..2`.

Do not update the paper table until every table cell traces to:

- raw per-seed JSON,
- W&B run ID,
- git SHA,
- exact command,
- checkpoint path,
- and the metric schema above.
