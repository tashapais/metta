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

## Preparation Before Reruns

Do not launch the new sandbox jobs until the items in this section are handled.
The current branch has protocol, metric helpers, and audit helpers, but it does
not yet have a canonical training launcher for the fixed-role Tribal Village
reward-mixing experiment. The later `paper_exp_reward_type.py` script should not
be used for this paper section because it trains the non-canonical binary
top/bottom return-probe stream.

### 1. Finish The Provenance Recovery Decision

First decide whether the pilot artifacts can be recovered or whether the rerun
will be an explicit reconstruction from the nearest reproducible setup.

Known pilot artifact names from Overleaf:

- `train_condition.py`
- `results_individual.json`
- `results_mixed80.json`
- `results_shared.json`
- `log_individual.txt`
- `log_mixed80.txt`
- `log_shared.txt`
- `best_model_{label}.pt`
- `final_model_{label}.pt`

Current audit state:

- W&B auth works locally, but `tashapais/representation-collapse` may not be
  visible to the currently authenticated account. If that remains true, record
  this as a provenance gap rather than silently substituting another project.
- The visible `tashapais/metta` and `metta-research/*` projects did not expose
  the expected `paper_tribal_*` or `paper_reward_*` run names in the first audit.
- The checked-in `results_reward_*.json` files are not canonical because their
  probe schema is binary return top/bottom, not the fixed 3-way role probe.
- The old pilot artifact names were not found in the fetched git refs checked so
  far.

Decision rule:

- If the old `train_condition.py`, result JSONs, logs, and model checkpoints are
  recovered, use them only to reconstruct the exact runner and provenance. Do
  not promote the one-seed pilot numbers to canonical results.
- If they are not recovered, create a new canonical runner in this branch and
  mark the experiments as reconstructed reruns that match the protocol below.

### 2. Build Or Recover The Canonical Runner

Add or recover a runner with an explicit entrypoint, tentatively:

```bash
uv run python v3_experiments/train_canonical_reward_geometry.py \
  --shared-frac 0.8 \
  --seed 0 \
  --total-agent-steps 4000000 \
  --eval-trials 10 \
  --output v3_experiments/canonical_results/primary_alpha0.8_seed0.json
```

The runner must implement the canonical environment and policy contract:

- Tribal Village, 1 team x 12 agents, 80x80 map.
- Fixed roles from `agent_id % 3`: four gatherers, four explorers, four
  guardians.
- No role observation, role id feature, or agent-specific policy parameter.
- One shared MAPPO encoder for the primary sweep.
- Optional separate-encoder mode only for the mechanism ablation.
- Role-specific shaping applied before reward mixing:
  - gatherer: `0.3 * visible_gold_tiles`
  - explorer: `0.5` if no gold or altar is visible
  - guardian: `2.0 * visible_altar_tiles`
- Reward mixing exactly:

```text
r_i_alpha = (1 - alpha) * r_i_ind + alpha * mean_j(r_j_ind)
```

The runner must also expose mechanism flags, but those flags should not change
the primary sweep by accident:

- `--disable-role-shaping` for the no-role-shaping ablation.
- `--separate-encoders` for the separate-encoder ablation.
- An explicit run name suffix such as `primary`, `no_role_shaping`, or
  `separate_encoders`.

Implementation guardrails:

- Reuse `v3_experiments/canonical_reward_geometry.py` for role labels, reward
  mixing, ordered KL, JS diversity, and stale-probe audit behavior.
- Keep the canonical runner separate from `paper_exp_reward_type.py` unless a
  small shared helper is genuinely useful.
- Do not add role labels to observations or training inputs. Role labels are for
  reward shaping and final probe/eval only.
- Record action and observation dimensions at runtime, because the pilot notes
  require matching the old action space if possible.

### 3. Define The Per-Seed Output Contract

Every training seed must write one raw JSON file. Aggregation should only read
these raw JSON files; it should not scrape W&B summaries as the source of truth.

Required per-seed fields:

- `schema_version`
- `condition_group`: `primary`, `no_role_shaping`, or `separate_encoders`
- `shared_frac`
- `seed`
- `num_agents`
- `num_teams`
- `map_width`
- `map_height`
- `role_assignment`
- `role_shaping_enabled`
- `separate_encoders`
- `total_agent_steps`
- `eval_trials`
- `metta_git_sha`
- `tribal_village_git_sha` or `tribal_village_build_id`
- `command`
- `wandb_entity`
- `wandb_project`
- `wandb_run_id`
- `wandb_url`
- `checkpoint_path`
- `obs_shape`
- `action_space_size`
- `metric_schema`
- `effrank_per_agent`
- `d_act_ordered_kl`
- `d_act_js`
- `role_probe_acc`
- `role_probe_chance`
- `role_probe_cv`
- `eval_trial_metrics`

Metric requirements:

- `role_probe_chance` must be exactly `1/3`.
- `role_probe_acc` must be a 3-way fixed-role probe using labels
  `agent_id % 3`.
- `d_act_ordered_kl` must be normalized by `n_agents * (n_agents - 1)`.
- `eval_trial_metrics` should store the 10 eval-trial values for traceability,
  but table-level uncertainty must be across training seeds.

### 4. Prepare W&B And Artifact Logging

Before launching any long run, confirm where the new canonical runs will log.

Required decisions:

- W&B entity and project for the new reruns. Prefer
  `tashapais/representation-collapse` if access is restored; otherwise choose a
  visible project and record the deviation in every per-seed JSON.
- Run naming convention. Suggested:
  `canonical_reward_geometry_{group}_alpha{alpha}_seed{seed}`.
- Whether checkpoint files are uploaded to W&B artifacts, S3, or both.

Required W&B config fields:

- full CLI command
- git SHA
- sandbox name
- seed
- `shared_frac`
- mechanism flags
- environment dimensions
- role-shaping coefficients
- action/observation dimensions
- checkpoint directory

Do not rely on committed or pasted W&B API keys in scripts. Use the sandbox
environment, `wandb login`, or a secret manager mechanism.

### 5. Prepare The Sandboxes

Use the sandboxes only after SkyPilot reports them as `UP`.

Status checks:

```bash
uv run sky status relh-sandbox-1 --all-users
uv run sky status relh-sandbox-2 --all-users
uv run sky queue relh-sandbox-1 --all-users --skip-finished
uv run sky queue relh-sandbox-2 --all-users --skip-finished
```

Remote readiness checks for each sandbox:

```bash
uv run sky exec <sandbox> -- env -C /workspace/metta git status --short --branch
uv run sky exec <sandbox> -- env -C /workspace/metta git rev-parse HEAD
uv run sky exec <sandbox> -- nvidia-smi
ssh <sandbox> "pgrep -af '[t]orchrun|[t]rain_canonical_reward_geometry.py|[t]ools/run.py' || true"
```

Remote setup checklist:

- `/workspace/metta` is on the intended branch or a clean worktree at the
  intended commit.
- The checkout is not dirty. If it is dirty, do not reset it without explicit
  approval; create a separate clean worktree for the experiment.
- `uv run python -c "import wandb, torch, sklearn; print('OK')"` works.
- Tribal Village dependencies and any native library build needed by the runner
  are present.
- The runner can create a 1 team x 12 agents, 80x80 environment and report
  observation/action dimensions.
- W&B login works from the sandbox and creates runs in the chosen project.
- Checkpoint and output directories exist and have enough disk space.
- Autostop behavior is understood before starting long runs.

### 6. Run Short Smoke Tests Before 4M-Step Jobs

Before launching the full matrix, run one very short smoke per condition family:

```bash
uv run python v3_experiments/train_canonical_reward_geometry.py \
  --shared-frac 0.0 \
  --seed 0 \
  --total-agent-steps 12000 \
  --eval-trials 1 \
  --output /tmp/canonical_reward_geometry_smoke.json
```

Smoke acceptance criteria:

- The runner completes without traceback.
- The JSON contains all required provenance fields.
- `role_probe_chance` is `1/3`.
- `obs_shape` and `action_space_size` are present and stable.
- A checkpoint is written and loadable.
- A W&B run is created with the expected config fields.
- `d_act_ordered_kl`, `d_act_js`, and `effrank_per_agent` are finite numbers.

Only after these pass should the long jobs start.

### 7. Launch The Primary Matrix

Primary matrix:

- `shared_frac=0.0`, seeds `0..4`
- `shared_frac=0.8`, seeds `0..4`
- `shared_frac=1.0`, seeds `0..4`

That is 15 primary training jobs. With two 4-GPU L4 sandboxes, launch at most
eight jobs at once, one per visible GPU, then launch the remaining seven after
the first batch clears.

Suggested first wave:

- `relh-sandbox-1`: alpha `0.0` seeds `0..3`
- `relh-sandbox-2`: alpha `0.8` seeds `0..3`

Suggested second wave:

- alpha `0.0` seed `4`
- alpha `0.8` seed `4`
- alpha `1.0` seeds `0..4`

For each detached launch, record:

- sandbox name
- GPU id
- shell/tmux session name
- command
- log path
- W&B URL
- output JSON path
- checkpoint path

Use tmux or SkyPilot task logs, but verify that the process passes environment
startup and begins training. Do not treat job submission alone as success.

### 8. Validate Finished Runs Before Aggregation

After each run finishes:

- Confirm process exit status and final log lines.
- Confirm the final checkpoint exists.
- Confirm the per-seed JSON exists and validates against the output contract.
- Confirm W&B has the matching run ID and final metrics.
- Confirm `role_probe_chance == 1/3`.
- Confirm `eval_trials == 10` for full runs.
- Confirm no table summary has been computed from eval-trial std alone.

Add a small validator if needed, for example:

```bash
uv run python v3_experiments/validate_canonical_reward_geometry_results.py \
  v3_experiments/canonical_results/*.json
```

### 9. Run Mechanism Checks Only After Primary Health

Do not start mechanism ablations until at least one full seed from each primary
alpha has completed cleanly and passed validation.

No-role-shaping ablation:

- `shared_frac in {0.0, 1.0}`
- seeds `0..2`
- `--disable-role-shaping`

Separate-encoder ablation:

- `shared_frac in {0.0, 1.0}`
- seeds `0..2`
- `--separate-encoders`

The mechanism ablations should use the same output contract and validation
script as the primary runs.

### 10. Aggregate And Prepare Paper Inputs

Aggregation should produce a separate canonical summary file, not mutate the raw
per-seed files.

Required summary outputs:

- mean/std across training seeds for each metric and condition
- seed list included for each table cell
- W&B run IDs included for each table cell
- checkpoint paths included for each seed
- explicit statement that uncertainty is across training seeds
- copy-paste-ready table values for Overleaf

The paper should not be updated until every primary table cell has five
validated training seeds. If a condition fails to reproduce the pilot pattern,
rewrite the paper around the reproduced result rather than preserving the pilot
claim.

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
