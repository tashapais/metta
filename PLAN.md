# Canonical MAPPO Shared-Reward Geometry Plan

## Summary

The paper section on MAPPO representation geometry under shared rewards should be
rebuilt around one canonical result stream: 12-agent Tribal Village, fixed
3-role assignment, shared MAPPO policy, and a reward-mixing sweep.

The current Overleaf Tribal Village numbers are pilot evidence only. They came
from one training seed per condition and five eval trials, so they are useful for
framing the hypothesis but not sufficient for final paper claims. The later
`v3_experiments/results_reward_*.json` stream is also non-canonical for this
paper section because it uses a binary top/bottom return probe rather than the
fixed 3-way Tribal role probe.

## Research Question

Does increasing shared reward mixing in 12-agent Tribal Village reduce
role-separable geometry and behavioral diversity in a shared MAPPO encoder?

Primary hypothesis:

- With fixed roles and a shared encoder, increasing `shared_frac` from `0.0` to
  `0.8` to `1.0` should reduce role-probe accuracy, EffRank per agent, and
  action-distribution diversity.

Mechanism hypothesis:

- If the effect is primarily shared-encoder gradient homogenization, then a
  separate-encoder ablation should rescue role-probe accuracy, especially under
  fully shared rewards.

Main confound to isolate:

- Role-specific shaping rewards are currently applied before reward mixing.
  A no-role-shaping ablation is needed to separate reward-mixing effects from
  hand-designed role-reward effects.

## Current Evidence

- Overleaf pilot notes report one training seed per condition, 4M agent-steps,
  and five eval trials per final checkpoint.
- Pilot reward sweep:
  - Individual `shared_frac=0.0`: EffRank/n `0.224 +/- 0.004`, `D_act=0.239 +/- 0.032`, role probe `1.000 +/- 0.000`.
  - Mixed `shared_frac=0.8`: EffRank/n `0.118 +/- 0.001`, `D_act=0.004 +/- 0.000`, role probe `0.902 +/- 0.049`.
  - Shared `shared_frac=1.0`: EffRank/n `0.092 +/- 0.001`, `D_act=0.000 +/- 0.000`, role probe `0.289 +/- 0.144`.
- This monotonic pattern is promising but provisional because eval-trial
  variation is weaker evidence than independent training-seed variation.
- Existing `results_reward_*.json` files should not be used as fixed-role
  evidence. Their stored probe baselines are incompatible with a 3-way role
  probe, and several have negative lift against their own recorded baseline.

## Provenance Gate

Before rerunning or replacing paper tables, recover or reconstruct the exact
training context:

- Query W&B for candidate runs from `tashapais/representation-collapse`, using
  names and config hints such as `tribal`, `reward`, `mixed80`, `shared`,
  `individual`, `shared_frac`, `num_teams`, and `num_agents`.
- Inspect git history and branches:
  - `main`
  - `origin/main`
  - `origin/tashapais-gpu0-smac-experiments`
  - `origin/tashapais-gpu1-craftax-experiments`
  - `origin/tashapais-gpu2-mettagrid-experiments`
  - `origin/tashapais-gpu3-ablation-experiments`
- Recover or record:
  - metta git SHA
  - Tribal Village package/build SHA
  - exact command
  - map/team config
  - action and observation dimensions
  - seed list
  - checkpoint paths
  - W&B run IDs
  - metric schema
- If exact old provenance is not recoverable, rerun from the nearest reproducible
  checkout that matches the pilot setup: 1 team x 12 agents, 80x80 map, fixed
  roles `agent_id % 3`, and action space matching the pilot notes.

Useful audit command:

```bash
uv run python v3_experiments/audit_reward_geometry.py --repo-root .
```

With W&B auth:

```bash
uv run python v3_experiments/audit_reward_geometry.py \
  --repo-root . \
  --include-wandb \
  --output v3_experiments/canonical_reward_geometry_provenance.json
```

## Canonical Experiment Protocol

Environment and policy:

- Environment: Tribal Village.
- Team config: 1 team x 12 agents.
- Map: 80x80.
- Policy: one shared MAPPO policy.
- No role input.
- No agent-specific parameters.
- Roles: `role = agent_id % 3`, giving four gatherers, four explorers, and four guardians.

Role shaping before reward mixing:

- Gatherer: `0.3 * visible_gold_tiles`.
- Explorer: `0.5` if no gold or altar is visible.
- Guardian: `2.0 * visible_altar_tiles`.

Reward mixing:

```text
r_i_alpha = (1 - alpha) * r_i_ind + alpha * mean_j(r_j_ind)
```

Primary rerun:

- `alpha in {0.0, 0.8, 1.0}`.
- Training seeds `0..4` for each condition.
- 4M agent-steps per seed.
- 10 fixed eval trials per final checkpoint.
- Table uncertainty is mean/std across independent training seeds, not eval trials.

## Mechanism Checks

No-role-shaping ablation:

- Conditions: `alpha in {0.0, 1.0}`.
- Seeds: `0..2`.
- Disable role-specific shaping before reward mixing.
- Purpose: determine whether the reward-mixing pattern survives without
  hand-designed role rewards.

Separate-encoder ablation:

- Conditions: `alpha in {0.0, 1.0}`.
- Seeds: `0..2`.
- Give each agent its own encoder while keeping the rest of the training setup
  matched.
- Purpose: test whether shared-encoder gradient homogenization is the proximate
  mechanism.

Robustness metric:

- Compute Jensen-Shannon action diversity offline as a secondary robustness
  metric.
- Keep ordered KL as the primary paper metric for continuity with the pilot.

## Metric Contract

Canonical metrics:

- `effrank_per_agent`: effective rank over flattened eval embeddings
  `[eval_steps * agents, embedding_dim]`, divided by `n_agents`.
- `d_act_ordered_kl`: mean ordered off-diagonal KL over `n_agents * (n_agents - 1)` pairs.
- `d_act_js`: unordered Jensen-Shannon action diversity for robustness.
- `role_probe_acc`: 3-way fixed-role probe using `agent_id % 3`; chance is `1/3`.

Non-canonical for this paper section:

- `return_probe_acc` or `probe_accuracy` from binary top/bottom return labels.
- Any probe result whose stored `probe_chance` is not `1/3`.
- Any table cell mixing eval-trial std with training-seed std.

## Paper Update Policy

The Overleaf paper has already been updated to mark the Tribal table as pilot
evidence and to fix the ordered-KL denominator.

When primary reruns complete:

- Replace the pilot Tribal table only with metrics traceable to per-seed JSON,
  W&B run IDs, git SHA, exact command, and checkpoint paths.
- If the monotonic pattern reproduces across seeds, claim that reward mixing
  tracks reduced role geometry and behavioral diversity in this setup.
- If it does not reproduce, rewrite the results around the stable finding and
  state that the single-seed pilot was not sufficient evidence.
- Keep SMACv2 as a boundary-condition section only if its provenance and metric
  schema pass the same audit.

## Acceptance Criteria

The final results section is canonical only when:

- Every table cell has raw per-seed JSON.
- Every per-seed JSON has W&B run ID, git SHA, command, environment config, seed,
  and checkpoint path.
- Role-probe chance is exactly `1/3`.
- `D_act` uses ordered off-diagonal KL normalized by `n * (n - 1)`.
- Existing `results_reward_*.json` files are not cited as fixed-role evidence.
- The paper clearly distinguishes pilot results, canonical reruns, and mechanism
  ablations.

## Implemented Support In This Branch

- `v3_experiments/canonical_reward_geometry.py`
  - role labels
  - reward mixing
  - ordered-KL action diversity
  - JS action diversity
  - fixed-role probe audit
- `v3_experiments/audit_reward_geometry.py`
  - local result audit
  - git branch provenance audit
  - optional W&B provenance query
- `v3_experiments/canonical_reward_geometry_protocol.json`
  - machine-readable protocol
- `v3_experiments/README_canonical_reward_geometry.md`
  - runbook for audit and reruns
- `tests/v3_experiments/test_canonical_reward_geometry.py`
  - tests for reward mixing, role labels, `D_act`, JS, and stale probe audits

Focused verification already run:

```bash
uv run pytest tests/v3_experiments/test_canonical_reward_geometry.py -q
```

Result:

```text
7 passed
```
