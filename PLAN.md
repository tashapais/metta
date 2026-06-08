# Behavioral Reward Redesign and MAPPO Geometry Rerun Plan

## Summary

We are pivoting the Tribal Village representation experiment from a provenance
recovery rerun into a behavior-first reward redesign. The previous canonical
reward-geometry rerun produced a clean monotonic geometry result across reward
mixing levels, but replay and baseline audits showed that the learned policies
were not doing meaningful Tribal Village work. The next phase is therefore:

1. Redesign the shaped rewards so they are tied to successful task events rather
   than passive observations.
2. Add instrumentation that can prove agents are moving, using objects,
   transferring resources, crafting, defending, and improving task score.
3. Pass a behavioral validity gate against no-op, random, and simple scripted
   baselines before running another expensive seed sweep.
4. Once behavior is meaningful, rerun the shared-reward geometry analysis and
   update the paper only from the new canonical result stream.

The paper should not currently claim that MAPPO learned meaningful Tribal
Village roles. The defensible current interpretation is narrower: under the
old shaped rewards, shared reward mixing strongly affected representation
geometry and action diversity metrics, but the policies failed the behavioral
sanity check.

## Research Question

Can shared-parameter MAPPO learn task-relevant differentiated behavior in Tribal
Village under event-based shaped rewards, and if it does, does increasing shared
reward mixing reduce role-separable geometry and behavioral diversity?

Primary hypothesis:

- With event-based role shaping, the individual-reward condition
  `shared_frac=0.0` should learn differentiated task behavior and role-separable
  representations.
- Increasing reward sharing from `0.0` to `0.8` to `1.0` should reduce
  role-separable geometry and action-distribution diversity, but only after the
  behavior gate confirms that the underlying policies are doing real work.

Mechanism hypothesis:

- The old passive shaping terms were exploitable because rewards could be earned
  from what an agent observed rather than what it successfully did.
- Event-based shaping should force reward to track actual environment progress:
  resource pickup, resource delivery, crafting, handoffs, defense, survival, and
  team score.

Main paper risk:

- If behavior remains trivial after event-based shaping, the current Tribal
  Village setup should be treated as a failed training environment for the paper,
  not as positive evidence about role learning.

## Current Evidence

The completed canonical rerun produced stable seed-level geometry trends:

| Condition | EffRank/n | Action diversity | Role-probe accuracy |
| --- | ---: | ---: | ---: |
| `shared_frac=0.0` | `0.177 +/- 0.006` | `3.141 +/- 0.690` | `0.853 +/- 0.041` |
| `shared_frac=0.8` | `0.0888 +/- 0.0006` | `0.056 +/- 0.011` | `0.678 +/- 0.016` |
| `shared_frac=1.0` | `0.0836 +/- 0.0001` | `0.000057 +/- 0.000014` | `0.447 +/- 0.017` |

Those numbers are useful for diagnosing how reward mixing interacts with the
encoder, but they are not enough for the paper because the behavioral audit
failed:

- Deterministic rollouts often collapsed to a single repeated action.
- Raw environment returns were flat or negative.
- Shaped reward could be high even when task progress was poor.
- No-op and random baselines were competitive with, or better than, trained
  checkpoints on several sanity metrics.
- Visual replays did not show robust gathering, crafting, defense, or team-level
  task progress.

Current replay and audit artifacts:

- Replay index:
  `../canonical-reward-geometry-results/replays/index.html`
- Deterministic checkpoint scan:
  `../canonical-reward-geometry-results/replays/all_checkpoint_deterministic_scan_120steps.json`

Paper interpretation for now:

- Do not use the current rerun as positive MAPPO behavioral evidence.
- Do not claim learned specialization, learned roles, or successful Tribal
  Village task competence.
- The current rerun can be cited internally as a failure mode: representation
  and probe metrics can look coherent while behavior is not meaningful.

## Behavioral Validity Gate

No full seed sweep should run until a candidate reward design passes this gate.
The gate should be run for at least `shared_frac=0.0` before testing the full
reward-mixing sweep.

Required baselines:

- No-op policy.
- Uniform random policy.
- Simple scripted policy or policies, even if weak.
- Current old-reward checkpoint as a negative-control reference when useful.

Required rollout artifacts:

- Per-policy JSON metrics.
- Per-policy replay files.
- At least three visually inspected replays for each candidate reward design.
- A compact comparison table against no-op and random.

Required metrics:

- Raw environment reward or task score.
- Shaped reward, logged separately from raw reward.
- Movement distance and coverage.
- Unique action count and action entropy.
- Unique joint-action count.
- Repeated-action streak length.
- Successful action counts by verb.
- Failed or invalid action counts by verb.
- Resource collection counts.
- Inventory deltas.
- Resource delivery or deposit counts.
- Handoff or put/give counts.
- Crafting or conversion outputs.
- Tumor attack, damage, and kill counts when available.
- Lantern planting or equivalent protective action counts.
- Agent health, death, and survival metrics when available.
- Team score or cumulative objective progress.

Minimum pass criteria:

- Trained policy beats no-op and random on raw task progress, not just shaped
  reward.
- Trained policy has nonzero successful task events across multiple categories.
- Deterministic rollouts do not collapse into a single fixed action for the
  whole episode.
- Shaped reward correlates with raw task progress and event counters.
- Visual replays show interpretable behavior: movement toward objects, object
  interaction, resource flow, crafting or depositing, and defense where relevant.
- The same checkpoint should not look good only because a passive reward term is
  triggered repeatedly while the task state remains unchanged.

## Reward Redesign

The new shaping should be event-based, success-gated, and capped. Avoid passive
visibility rewards as primary shaping signals.

### Global Principles

- Reward successful state changes, not observations alone.
- Log raw task reward and shaped role reward separately.
- Use potential or delta rewards where possible.
- Cap dense shaping so it cannot dominate the task.
- Decay or suppress repeated rewards for the same unchanged state.
- Penalize invalid repeated action only if the behavior gate shows action
  collapse after the main reward fix.
- Keep role labels out of observations and policy inputs unless explicitly
  running a role-observation ablation.
- Keep the same reward-mixing formula:

```text
r_i_alpha = (1 - alpha) * r_i_ind + alpha * mean_j(r_j_ind)
```

### Candidate Roles

The old experiment used three artificial labels from `agent_id % 3`:
gatherer, explorer, guardian. That is still useful for continuity, but the
reward semantics should change.

Candidate role definitions:

- Gatherer/harvester:
  - successful pickup or harvest event;
  - inventory increase for wood, ore, wheat, water, or equivalent resources;
  - delivery of raw resources to a relevant building or teammate;
  - movement toward known resource sites only as a weak auxiliary term.
- Crafter/logistics:
  - successful use of converter, assembler, forge, loom, oven, or equivalent
    production station;
  - creation of battery, spear, lantern, armor, bread, or equivalent item;
  - successful handoff to a teammate;
  - deposit or scoring event.
- Guardian/defender:
  - successful attack;
  - tumor damage or tumor kill;
  - lantern placement that changes the environment or improves protection;
  - survival, healing, or protection-related events if exposed by the runtime.

Explorer is the weakest old role because the previous reward encouraged
"nothing useful visible." If we keep an explorer label, it should receive only
capped novelty rewards:

- first visit to a new local region;
- discovery of resource, building, or threat tiles;
- map coverage increase;
- no reward for simply seeing an empty view repeatedly.

Preferred first implementation:

- Replace explorer with crafter/logistics for the controlled 3-role experiment.
- Keep the 12-agent shape as four gatherers, four crafters/logistics agents, and
  four guardians.
- If we need strict continuity with the old paper text, call this a revised
  three-role Tribal Village task and explicitly describe the role redesign.

### Coworld-Derived `event_v1` Setup

The standalone Coworld Tribal Village AI gives us a better behavioral template
than the old passive gatherer/explorer/guardian shaping. It has six scripted
team roles:

- `Hearter`: ore -> battery -> assembler heart workflow.
- `Armorer`: wood -> armor -> teammate armor handoff.
- `Hunter`: wood -> spear -> tumor/spawner/enemy defense.
- `Baker`: wheat -> bread -> teammate bread handoff.
- `Lighter`: wheat -> lantern -> lantern planting and territory protection.
- `Farmer`: water/fertile-ground/planting loop.

We should not port that AI directly into the 12-agent canonical experiment,
because the Coworld environment has a different contract: 48 agents, 8 teams,
6 agents per team, and 64 actions including the extra `plant_resource` verb.
The canonical trained package has 12 agents and 56 actions. Direct policy or
checkpoint mixing would invalidate the geometry comparison.

Instead, use Coworld as the semantic reference and collapse its six roles into
three canonical role labels:

| Canonical role | Agent label | Coworld sources | Rewarded event counters |
| --- | --- | --- | --- |
| `supplier` | `agent_id % 3 == 0` | Hearter/Farmer/Armorer/Baker/Lighter/Hunter collection phases | `resource_water`, `resource_wheat`, `resource_wood`, `resource_ore` |
| `crafter_logistics` | `agent_id % 3 == 1` | Hearter battery/heart workflow, Armorer/Baker handoffs, Lighter/Hunter production | `craft_battery`, `craft_spear`, `craft_lantern`, `craft_armor`, `craft_bread`, `deposit_heart`, `put_armor`, `put_bread` |
| `defender_territory` | `agent_id % 3 == 2` | Hunter combat, Lighter lantern protection | `tumor_kill`, `spawner_kill`, `agent_kill`, `lantern_plant` |

All roles also receive a small common bonus for successful interactive verbs:
`action_use`, `action_put`, `action_attack`, and `action_plant`. Invalid action
attempts receive a small penalty. This common term is intentionally weak; it
encourages interaction with the simulator without turning every role into the
same generic "press use" policy.

Initial `event_v1` coefficients:

| Component | Counter | Coefficient |
| --- | --- | ---: |
| Common interaction | `action_use`, `action_put`, `action_attack`, `action_plant` | `0.02` |
| Common invalid penalty | `action_invalid` | `-0.01` |
| Supplier resource pickup | `resource_water`, `resource_wheat`, `resource_wood`, `resource_ore` | `0.20` |
| Crafter/logistics production | `craft_battery`, `craft_spear`, `craft_lantern`, `craft_armor`, `craft_bread` | `0.45` |
| Crafter/logistics scoring | `deposit_heart` | `0.80` |
| Crafter/logistics handoff | `put_armor`, `put_bread` | `0.30` |
| Defender/territory threat removal | `tumor_kill`, `spawner_kill` | `0.90` |
| Defender/territory combat | `agent_kill` | `0.40` |
| Defender/territory protection | `lantern_plant` | `0.45` |

This setup is cleaner than the old passive reward for two reasons:

- Rewards fire from per-step deltas of simulator counters, so unchanged
  observations cannot pay repeated shaping.
- The three labels describe real task subgraphs: supply, production/logistics,
  and defense/territory. This is closer to the Coworld role structure while
  preserving the 12-agent canonical geometry protocol.

First training command shape:

```bash
uv run python v3_experiments/train_canonical_reward_geometry.py \
  --reward-design event_v1 \
  --shared-frac 0.0 \
  --seed 0 \
  --total-agent-steps 1000000 \
  --eval-trials 10 \
  --output v3_experiments/behavioral_reward_results/event_v1_alpha0_seed0.json
```

Negative-control command shape:

```bash
uv run python v3_experiments/train_canonical_reward_geometry.py \
  --reward-design passive_v0 \
  --shared-frac 0.0 \
  --seed 0 \
  --total-agent-steps 1000000 \
  --eval-trials 10 \
  --output v3_experiments/behavioral_reward_results/passive_v0_alpha0_seed0.json
```

Result JSON must record `reward_design`, `reward_design_details`, raw
environment return, role-shaping return, individual pre-mix return, and final
mixed return. W&B should use the same separation so shaped reward cannot be
mistaken for task progress.

## Instrumentation Plan

The next implementation task is not just changing reward constants. We need
event counters that make the behavior gate mechanically checkable.

Required instrumentation surfaces:

- Environment step output should expose per-agent and team-level event counters.
- Training logs should record raw task reward and shaped reward separately.
- Rollout/eval scripts should save compact JSON summaries.
- Replay generation should be part of the validation loop.

Required event counters:

- `action_attempts_by_verb`
- `action_successes_by_verb`
- `invalid_actions_by_verb`
- `movement_steps`
- `unique_positions`
- `coverage_tiles`
- `resource_pickups_by_type`
- `resource_drops_by_type`
- `inventory_delta_by_type`
- `crafting_outputs_by_type`
- `handoffs_by_type`
- `deposits_by_type`
- `tumor_damage`
- `tumor_kills`
- `lantern_plants`
- `health_delta`
- `deaths`
- `survival_steps`
- `raw_env_reward`
- `shaped_role_reward`
- `team_score`

Implementation notes:

- Prefer extracting event counters from the Nim environment boundary rather than
  inferring everything from pixels or observations.
- If full event exposure is too large for the first pass, implement a smaller
  event schema first: movement, successful verb counts, inventory deltas,
  crafting outputs, tumor kills, and raw score.
- Keep the metrics schema stable across no-op, random, scripted, and trained
  policies.
- Add an output validator that rejects missing keys, `NaN`, incompatible action
  dimensions, or zero-length rollouts.

## Tribal Village Version Comparison

There are two relevant Tribal Village implementations. They are similar enough
to compare conceptually but different enough that checkpoints and metrics should
not be mixed without an explicit compatibility layer.

### A. Trained Metta Package

Location:

- `packages/tribal_village` in this Metta branch.

Observed contract:

- The canonical reward-geometry setup uses `-d:canonicalRewardGeometry`.
- Environment shape: 1 team x 12 agents.
- Map size: 80 x 80.
- Observation tensor: 21 layers, 11 x 11 local view.
- Action space: 56 discrete actions from 7 verbs x 8 arguments.
- Old role labels: `agent_id % 3`.
- Old role shaping was implemented outside the core task as passive
  observation-based terms.
- Replay artifacts were generated for analysis, but this package does not have
  the same Coworld browser/player/replay stack as the Metta-AI repository.

Strengths for the paper:

- Direct continuity with the already-run canonical geometry sweep.
- Existing MAPPO runner and metrics are wired around this action and agent
  contract.
- Lowest-risk path for a controlled reward-redesign rerun.

Weaknesses:

- The old shaping failed the behavioral sanity check.
- Event counters are currently insufficient for proving meaningful behavior.
- Artificial 3-role labels are a paper construct, not the richer role structure
  described by the standalone game.
- Existing checkpoints are tied to the 56-action, 12-agent setup.

### B. Metta-AI `coworld-tribal-village`

Location:

- `/Users/relh/Code/coworld-tribal-village`
- Remote: `git@github.com:Metta-AI/coworld-tribal-village.git`
- Local `main` observed at `7e43e806` on 2026-06-08 after fetching.

Observed contract:

- Public game shape: 48 player slots.
- Team shape: 8 teams x 6 agents.
- Observation tensor: 21 layers, 11 x 11 local view.
- Action space: 64 discrete actions from 8 verbs x 8 arguments.
- The extra verb is `plant_resource`.
- Includes Coworld server/player/replay support.
- Includes richer mechanics such as water, fertile ground, bridges, equipment,
  health UI, and more explicit team scoring.
- Contains richer scripted role logic for 6-agent teams, including farmer-like
  and logistics/defense behaviors.
- Has docs and protocol files that make browser replay and human inspection
  easier.

Strengths for the paper:

- More complete Tribal Village game semantics.
- Better aligned with the standalone game the paper text may be describing.
- Better replay and inspection path for showing meaningful behavior.
- Richer mechanics could support clearer task roles.

Weaknesses:

- Not checkpoint-compatible with the trained 12-agent, 56-action Metta package.
- Team and agent counts differ from the canonical experiment.
- Action space differs because of `plant_resource`.
- Directly comparing old geometry numbers to this version would be invalid.
- It may require either a new training runner or a compatibility mode.

### Contrast Table

| Dimension | Trained Metta package | Metta-AI Coworld repo |
| --- | --- | --- |
| Current research role | Controlled geometry rerun target | Richer game/reference target |
| Agents | 12 | 48 |
| Teams | 1 | 8 |
| Agents per team | 12 in canonical mode | 6 |
| Map | 80 x 80 in canonical mode | game-defined Coworld map |
| Observation | 21 x 11 x 11 | 21 x 11 x 11 |
| Actions | 56 = 7 verbs x 8 args | 64 = 8 verbs x 8 args |
| Extra mechanics | smaller experimental package | water, fertile ground, bridges, equipment, richer UI |
| Role structure | artificial 3-role labels | richer 6-agent team logic |
| Replay/browser | analysis artifacts | Coworld server/player/replay |
| Checkpoint compatibility | compatible with current rerun | incompatible without adaptation |
| Paper risk | may be too artificial | requires new controlled protocol |

### Version Decision

Immediate path:

- Implement event-based rewards and behavior instrumentation in the trained
  Metta package first, because that preserves continuity with the existing
  MAPPO geometry runner.
- Use the Coworld repo as the reference implementation for richer mechanics,
  replay expectations, and role semantics.
- Do not mix numerical results across the two versions unless we build an
  explicit compatibility mode and document it.

Parallel comparison path:

- Write a short environment-contract audit for both versions:
  - agent/team count;
  - action verbs and action-space size;
  - observation shape;
  - scoring definition;
  - available event hooks;
  - replay support;
  - supported scripted policies.
- Run the no-op/random/scripted baseline gate in both environments if feasible.
- Decide whether the paper should describe:
  - the reconstructed 12-agent Metta package experiment;
  - the richer Coworld Tribal Village experiment;
  - or both, with the Coworld version as an external validity check.

Important compatibility rule:

- Existing 56-action checkpoints from the trained package cannot be evaluated
  directly in the 64-action Coworld environment.
- Existing 12-agent geometry metrics cannot be interpreted as 48-agent Coworld
  Tribal Village results.

## Experimental Phases

### Phase 0: Version and Baseline Audit

Goal:

- Lock down what environment is being trained and what behavior the baseline
  policies achieve.

Tasks:

- Record the Metta git SHA and Tribal package state.
- Record the Coworld repo SHA and action/observation/team contract.
- Produce a one-page contrast summary from the table above.
- Run no-op and random policies in the trained Metta package.
- Run simple scripted policies if available.
- Save metrics and replays.
- Confirm the action and observation dimensions used by the runner at runtime.

Exit criteria:

- Baseline JSON files exist.
- Replays can be opened.
- The behavior metrics schema is stable.
- We know whether the Coworld repo can be run locally or on a sandbox with the
  same validation harness.

### Phase 1: Event Instrumentation

Goal:

- Make behavior measurable before changing reward design.

Tasks:

- Add per-agent and team-level event counters.
- Add a rollout metrics writer.
- Add an output validator.
- Add a replay-generation command that can be run for trained, no-op, random,
  and scripted policies.
- Ensure W&B logs include raw reward, shaped reward, and behavior counters.

Exit criteria:

- No-op/random/scripted rollouts produce comparable JSON.
- Metrics do not require manual replay inspection to detect total collapse.
- Replay inspection remains available as a final sanity check.

### Phase 2: Reward Redesign

Goal:

- Replace passive shaping with event-based shaping.

Tasks:

- Implement gatherer reward from successful resource pickup and delivery.
- Implement crafter/logistics reward from successful production, handoff, and
  deposit events.
- Implement guardian reward from successful defense and protection events.
- Optionally implement capped explorer novelty if we keep explorer as a role.
- Add config flags so old rewards can be run as a negative control.
- Add reward component logging.

Exit criteria:

- Unit or smoke tests prove reward components only fire on successful events.
- No repeated passive reward can be earned from an unchanged observation.
- Rollout metrics show shaped reward and event counters move together.

### Phase 3: Short Training Gate

Goal:

- Find out quickly whether the new reward design produces meaningful behavior.

Initial run shape:

- `shared_frac=0.0`.
- 2 or 3 seeds.
- Short horizon first, for example 0.5M to 1M agent steps.
- Evaluate deterministic and stochastic policies.

Tasks:

- Train short candidate runs.
- Generate no-op/random/scripted/trained comparisons.
- Save replays for each seed.
- Inspect at least three replays.
- Compare raw task score, event counters, action entropy, and repeated-action
  streaks.

Exit criteria:

- Trained policies beat no-op and random on raw task progress.
- At least two major event categories are nonzero and visibly meaningful.
- Deterministic policy does not collapse into a fixed action loop.
- If this fails, revise rewards before any full sweep.

### Phase 4: Reward-Mixing Pilot

Goal:

- Check whether the geometry effect survives once behavior is meaningful.

Initial run shape:

- `shared_frac in {0.0, 0.8, 1.0}`.
- 2 seeds per condition.
- Intermediate budget.
- Same behavior gate for every condition.

Tasks:

- Run the three-condition pilot.
- Aggregate behavior metrics.
- Aggregate geometry metrics.
- Compare raw reward, event success, EffRank/n, action diversity, and role-probe
  accuracy.
- Save replays for every seed and condition.

Exit criteria:

- At least the individual-reward condition passes the behavior gate.
- Shared-reward conditions can be interpreted clearly as either collapsed
  behavior, reduced differentiation, or both.
- The metrics and replays agree.

### Phase 5: Canonical Rerun

Goal:

- Produce final paper-quality numbers only after behavior is validated.

Run shape:

- `shared_frac in {0.0, 0.8, 1.0}`.
- 5 independent seeds per condition.
- 4M agent steps unless the short gate shows a better justified budget.
- Fixed eval protocol.
- Fixed replay protocol.

Required outputs:

- One raw JSON file per seed.
- One W&B run per seed.
- Checkpoint paths.
- Replay artifacts.
- Aggregated behavior table.
- Aggregated geometry table.
- Statistical summary with seed-level means and standard errors.

Exit criteria:

- Behavior gate passes for the relevant claim.
- Aggregation reads raw JSON files, not W&B summaries as source of truth.
- Every table row has provenance: git SHA, command, config, seed list, and
  artifact path.

### Phase 6: Paper Update

Goal:

- Update the paper only from the canonical result stream.

Rules:

- Do not mix pilot, temporary, failed-reward, and final canonical results in the
  Overleaf text.
- Keep temporary debugging details out of the paper.
- Use the paper to report the final validated experiment and the relevant
  caveats.
- If the behavior redesign fails, report that as a limitation or remove the
  positive Tribal Village MAPPO claim.

Possible final claim if the redesigned runs pass:

- "In a behavior-validated Tribal Village setting with event-based role rewards,
  increasing shared reward mixing reduced role-separable representation geometry
  and action diversity in a shared MAPPO encoder."

Possible final claim if behavior still fails:

- "In our Tribal Village reconstruction, reward mixing affected representation
  metrics, but the policies failed behavioral validation; we therefore do not
  interpret the probe results as evidence of learned task roles."

## Live Sandbox Ramp: 2026-06-08

Goal:

- Use `relh-sandbox-1` and `relh-sandbox-2` to run the `event_v1` reward ramp
  in increasing budgets, with behavior-gate evidence before promoting to larger
  representation experiments.

Branch and code state:

- Local branch: `canonical-reward-geometry-plan`.
- Launch branch: `tashapais/metta` branch `canonical-reward-geometry-plan`.
- Exact launch commit should be recorded in every result JSON.
- Remote sandboxes may have `/workspace/metta` on Metta-AI upstream branches or
  dirty local state. Do not reset those checkouts. Use clean git worktrees under
  `/workspace/tribal_event_v1_<sha>` instead.

Sandbox allocation:

| Sandbox | Stage-1 allocation | Notes |
| --- | --- | --- |
| `relh-sandbox-1` | `event_v1`, `shared_frac=0.0`, seeds `0` and `1` | Main checkout is clean but on another branch; use a clean worktree. |
| `relh-sandbox-2` | `event_v1`, `shared_frac=0.0`, seed `2` | Main checkout has dirty `MODULE.bazel.lock`; use a clean worktree. |

Stage 1: short behavior discovery.

- Run `event_v1`, `shared_frac=0.0`, seeds `0,1,2`.
- Initial budget: `1,000,000` agent steps per seed.
- Log to W&B if the sandbox is authenticated; otherwise write full local JSON
  and checkpoint artifacts and continue with offline/local provenance.
- After completion, run checkpoint rollouts with deterministic and stochastic
  actions when feasible.
- Compare against no-op/random/use-sweep baselines using the behavior summary
  tooling.

Stage-1 promotion gate:

- At least one seed must show nonzero task events beyond invalid/no-op/action
  attempts.
- Behavior metrics must include resource, craft, deposit, handoff, combat, or
  lantern events; one isolated accidental event is not enough.
- Deterministic replay should not be a single fixed joint action loop.
- Raw environment return and event counters should not contradict the shaped
  reward story.

Stage 2: longer single-condition ramp.

- Promote only the best Stage-1 seed or seeds.
- Budget: start with `10,000,000` agent steps, then consider `50,000,000` if
  counters continue improving.
- Before runs much larger than this, add or enable periodic checkpointing and
  resume support. The current runner writes a final checkpoint; that is too
  brittle for billion-step jobs.
- Save behavior rollouts and summaries at each budget boundary.

Stage 3: shared-fraction pilot.

- Run `event_v1` with `shared_frac in {0.0, 0.8, 1.0}`.
- Use two seeds per condition if Stage 2 behavior is meaningful.
- Compare representation metrics and behavior metrics together; do not interpret
  representation collapse if behavior is trivial.

Stage 4: canonical-scale decision.

- Do not launch billion-step or final canonical sweeps until:
  - Stage 3 produces at least one behavior-valid condition;
  - periodic checkpoints/resume are in place;
  - replay and JSON artifacts are harvestable without manual reconstruction;
  - W&B or equivalent local provenance is confirmed for every run.
- If these pass, choose the final budget empirically from Stage 2 and Stage 3
  learning curves. Billions of timesteps may be appropriate, but only after the
  reward and infrastructure prove they are worth scaling.

Live monitoring loop:

- Poll SkyPilot queue state, remote process state, GPU utilization, local run
  logs, and output JSON/checkpoint files.
- Use direct SSH for tmux/process/log probes and `uv run sky` for queue/log
  truth.
- Keep temporary pilot details in this plan and local artifacts, not in the
  Overleaf paper.
- Update this section with concrete run names, artifact paths, and pass/fail
  status as jobs finish.

### Stage-1 Status: Failed Behavior Gate

Run date:

- 2026-06-08.

Sandbox staging:

- `relh-sandbox-1` and `relh-sandbox-2` were both up and SSH reachable.
- Both sandboxes used clean detached worktrees at
  `/workspace/tribal_event_v1_0741537b`.
- Exact commit: `0741537b6d231640382307f3dbf294dbbe0afa32`.
- `/workspace/metta` was not reset. This mattered because `relh-sandbox-2`
  had a dirty `MODULE.bazel.lock`.

Launch note:

- W&B online launch failed with a 404 for project
  `tashapais/representation-collapse`.
- Stage-1 runs were relaunched with `--wandb-mode offline`.
- Offline W&B directories were written under each run directory and can be
  synced later if the project/access issue is fixed.

Stage-1 training runs:

| Seed | Sandbox | Run directory | Steps | Status |
| ---: | --- | --- | ---: | --- |
| `0` | `relh-sandbox-1` | `/workspace/tribal_event_v1_runs/stage1/event_v1_stage1_alpha0_seed0_1m_0741537b_offline` | `1,000,008` | complete |
| `1` | `relh-sandbox-1` | `/workspace/tribal_event_v1_runs/stage1/event_v1_stage1_alpha0_seed1_1m_0741537b_offline` | `1,000,008` | complete |
| `2` | `relh-sandbox-2` | `/workspace/tribal_event_v1_runs/stage1/event_v1_stage1_alpha0_seed2_1m_0741537b_offline` | `1,000,008` | complete |

Stage-1 training summary:

| Seed | EffRank/n | Ordered KL | JS diversity | Role probe | Eval raw return | Eval shaping return |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0847` | `0.00112` | `0.000279` | `0.412` | `-11.07` | `0.0` |
| `1` | `0.0840` | `0.000121` | `0.000030` | `0.395` | `-10.90` | `0.0` |
| `2` | `0.0842` | `0.000224` | `0.000056` | `0.290` | `-10.03` | `0.0` |

Interpretation:

- Representation and action-diversity metrics are already collapsed-looking at
  this budget.
- More importantly, `mean_role_shaping_return` is zero in deterministic eval for
  all three seeds. The event reward is not being discovered by the learned
  policy.

Behavior-gate rollouts:

- Baselines and checkpoint rollouts were saved under
  `/workspace/tribal_event_v1_runs/behavior_gate`.
- Replays were saved as JSONL files under each rollout directory's `replays/`
  subdirectory.
- Baselines were run on `relh-sandbox-1`; seed-2 checkpoint rollouts were run on
  `relh-sandbox-2`.

Behavior summary:

| Policy | Raw reward | Task events | Resources | Crafts | Deposits | Combat | Unique joint actions | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `no_op` | `-14.400` | `0.0` | `0` | `0` | `0` | `0` | `1.0` | `no_task_events,single_joint_action,mostly_noop` |
| `random` | `-17.733` | `16.7` | `45` | `3` | `0` | `0` | `120.0` | `mostly_invalid` |
| `move_sweep` | `-14.400` | `0.0` | `0` | `0` | `0` | `0` | `8.0` | `no_task_events` |
| `use_sweep` | `-14.400` | `10.3` | `23` | `8` | `0` | `0` | `8.0` | `mostly_invalid` |
| seed-0 deterministic checkpoint | `-14.400` | `0.0` | `0` | `0` | `0` | `0` | `1.0` | `no_task_events,single_joint_action` |
| seed-0 stochastic checkpoint | `-14.400` | `0.0` | `0` | `0` | `0` | `0` | `120.0` | `no_task_events` |
| seed-1 deterministic checkpoint | `-14.400` | `0.0` | `0` | `0` | `0` | `0` | `1.0` | `no_task_events,single_joint_action` |
| seed-1 stochastic checkpoint | `-16.067` | `0.0` | `0` | `0` | `0` | `0` | `120.0` | `no_task_events` |
| seed-2 deterministic checkpoint | `-16.067` | `0.0` | `0` | `0` | `0` | `0` | `1.0` | `no_task_events,single_joint_action` |
| seed-2 stochastic checkpoint | `-14.400` | `0.0` | `0` | `0` | `0` | `0` | `120.0` | `no_task_events` |

Stage-1 decision:

- Failed. Do not promote these runs to Stage 2, Stage 3, or billion-step
  experiments.
- The counters are working: `random` and `use_sweep` can trigger resource and
  craft events.
- The learned policy is not discovering those event rewards under the current
  sparse `event_v1` setup.

Immediate next research action:

- Do not spend more compute on the current `event_v1` reward alone.
- Add a discovery aid before the next ramp. The likely candidates are:
  - a curriculum or scripted-start distribution that places agents near useful
    resource/building interactions;
  - a weak dense auxiliary for valid `use`/resource-proximity that decays after
    event discovery;
  - scripted or imitation warm starts from the Coworld AI role logic;
  - or a short behavior-cloning/pretraining phase followed by MAPPO.
- Re-run Stage 1 only after at least one of those changes makes task events
  discoverable by learning, not only by random/use-sweep baselines.

### Reward-Debug Continuation: `event_v2_breadcrumbs`

Interpretation of the Stage-1 failure:

- We did train on `event_v1`; this was not just an evaluation failure.
- The training runs sampled for `1,000,008` agent steps per seed, but PPO did
  not discover positive event shaping in a stable way.
- The evidence is different from the original passive-reward failure:
  - original passive rewards paid for observations without meaningful work;
  - `event_v1` only pays for real task events, but the learned policies failed
    to discover those events.
- The behavior counters are not the bottleneck because random/use-sweep
  baselines produced resource and craft events.

Next reward-debug design:

- Add `event_v2_breadcrumbs` as an exploratory reward scaffold.
- Keep the same three role names: `supplier`, `crafter_logistics`,
  `defender_territory`.
- Preserve `event_v1` as the sparse role-specialized baseline.
- Pay all agents a role-agnostic breadcrumb for successful task events, then add
  role-specific bonuses on top. This lets learning reinforce "resource pickup
  happened" even if the first agent to discover it is not assigned to the
  supplier label.
- Increase rewards for successful interactive verbs and reduce the invalid
  penalty. The previous invalid penalty plus sparse role-specific rewards likely
  made exploratory `use` behavior unattractive before task events were found.

Initial `event_v2_breadcrumbs` intent:

| Component | Purpose |
| --- | --- |
| valid interaction breadcrumb | keep agents trying successful `use`, `put`, `attack`, `plant`, and `swap` actions |
| small no-op/invalid penalty | discourage collapse without overwhelming rare discoveries |
| role-agnostic task-event reward | reinforce resource/craft/deposit/handoff/combat/lantern events for every role |
| extra role-specific reward | keep pressure toward the Coworld-derived three-role decomposition |

Ramp rules for `event_v2_breadcrumbs`:

- First run a tiny native smoke to confirm reward metadata and event deltas.
- Re-run Stage 1 with seeds `0,1,2`, `shared_frac=0.0`, and a higher entropy
  coefficient such as `--ent-coef 0.05`.
- Promote only if at least one checkpoint rollout has nonzero task events and
  nonzero shaping return, and preferably beats random/use-sweep on at least one
  meaningful task category.
- If `event_v2_breadcrumbs` still fails, the next step should be curriculum or
  scripted/imitation warm starts, not larger budgets.

### Reward-Debug Continuation: `event_v3_navigation_breadcrumbs`

`event_v2_breadcrumbs` was implemented and tested on the same Stage-1 ramp.

Run date:

- 2026-06-08.

Sandbox staging:

- `relh-sandbox-1` and `relh-sandbox-2` used clean detached worktrees at
  `/workspace/tribal_event_v2_66b093eaf`.
- Exact commit: `66b093eafcb8bcb637eb47118238307f07e9bd43`.
- Local smoke and sandbox GPU smokes succeeded with
  `--reward-design event_v2_breadcrumbs`.

Stage-1 `event_v2_breadcrumbs` runs:

| Seed | Sandbox | Run directory | Steps | Status |
| ---: | --- | --- | ---: | --- |
| `0` | `relh-sandbox-1` | `/workspace/tribal_event_v2_runs/stage1/stage1_event_v2_breadcrumbs_alpha0_seed0_1m_66b093eaf` | `1,000,008` | complete |
| `1` | `relh-sandbox-1` | `/workspace/tribal_event_v2_runs/stage1/stage1_event_v2_breadcrumbs_alpha0_seed1_1m_66b093eaf` | `1,000,008` | complete |
| `2` | `relh-sandbox-2` | `/workspace/tribal_event_v2_runs/stage1/stage1_event_v2_breadcrumbs_alpha0_seed2_1m_66b093eaf` | `1,000,008` | complete |

Stage-1 `event_v2_breadcrumbs` training summary:

| Seed | EffRank/n | Ordered KL | JS diversity | Role probe | Eval raw return | Eval shaping return | Train shaping return |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0937` | `0.000088` | `0.000022` | `0.425` | `-10.33` | `-0.982` | `0.00092` |
| `1` | `0.0902` | `0.000168` | `0.000042` | `0.565` | `-10.33` | `-0.933` | `0.00274` |
| `2` | `0.0920` | `0.000249` | `0.000062` | `0.351` | `-10.82` | `-0.981` | `0.02108` |

Behavior-gate artifacts:

- Baselines and seed-0/seed-1 checkpoint rollouts:
  `/workspace/tribal_event_v2_runs/behavior_gate` on `relh-sandbox-1`.
- Seed-2 checkpoint rollouts:
  `/workspace/tribal_event_v2_runs/behavior_gate` on `relh-sandbox-2`.
- Replays are JSONL files under each rollout directory's `replays/`
  subdirectory.

Behavior summary:

| Policy | Raw reward | Task events | Resources | Crafts | Deposits | Combat | Unique joint actions | Invalid frac | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `no_op` | `-14.400` | `0.0` | `0` | `0` | `0` | `0` | `1.0` | `0.00` | `no_task_events,single_joint_action,mostly_noop` |
| `random` | `-17.733` | `15.0` | `37` | `5` | `0` | `1` | `120.0` | `0.75` | `mostly_invalid` |
| `move_sweep` | `-16.067` | `0.0` | `0` | `0` | `0` | `0` | `8.0` | `0.07` | `no_task_events` |
| `use_sweep` | `-14.400` | `9.3` | `21` | `7` | `0` | `0` | `8.0` | `0.99` | `mostly_invalid` |
| seed-0 deterministic checkpoint | `-14.400` | `0.7` | `2` | `0` | `0` | `0` | `1.0` | `1.00` | `single_joint_action,mostly_invalid` |
| seed-0 stochastic checkpoint | `-14.333` | `25.0` | `74` | `1` | `0` | `0` | `120.0` | `0.77` | `mostly_invalid` |
| seed-1 deterministic checkpoint | `-14.400` | `1.7` | `5` | `0` | `0` | `0` | `1.0` | `1.00` | `single_joint_action,mostly_invalid` |
| seed-1 stochastic checkpoint | `-16.033` | `30.3` | `88` | `3` | `0` | `0` | `120.0` | `0.78` | `mostly_invalid` |
| seed-2 deterministic checkpoint | `-14.400` | `1.3` | `4` | `0` | `0` | `0` | `1.0` | `1.00` | `single_joint_action,mostly_invalid` |
| seed-2 stochastic checkpoint | `-16.067` | `50.0` | `135` | `7` | `0` | `0` | `120.0` | `0.77` | `mostly_invalid` |

Interpretation:

- Training did attempt to discover the shaped rewards. `event_v1` gave zero
  deterministic shaping return, and `event_v2_breadcrumbs` found only a bad
  local optimum.
- The deterministic `event_v2` checkpoints always collapsed to one repeated
  `use` action: seed 0 used action `27`, seed 1 used action `28`, and seed 2
  used action `29`.
- Those fixed `use` policies occasionally collect a resource by accident, but
  almost every action is invalid. This is not meaningful Tribal Village
  behavior and does not pass the behavior gate.
- Stochastic checkpoint rollouts produce task events only because they are noisy
  and random-like; the invalid fraction remains about `0.77`, so these are not
  behavior-valid policies either.

Next reward-debug design:

- Add `event_v3_navigation_breadcrumbs`.
- Keep all reward terms tied to simulator event deltas.
- Add a small successful-movement breadcrumb capped per agent per episode. This
  is an exploration scaffold, not a final task reward.
- Increase task-event rewards so resource/craft/deposit/defense events dominate
  blind action spam.
- Increase the invalid-action penalty relative to `event_v2`, because the
  observed failure mode is repeated invalid `use`.

Promotion rule:

- Do not promote `event_v3_navigation_breadcrumbs` only because shaped return is
  positive.
- Promote only if deterministic checkpoint rollouts no longer collapse to a
  single joint action, invalid fraction drops materially below random/use-sweep,
  and task-event counts beat no-op plus the fixed sweep baselines.
- If `event_v3_navigation_breadcrumbs` still collapses, stop reward-only
  debugging and switch to a curriculum, action masking, or scripted/imitation
  warm start.

`event_v3_navigation_breadcrumbs` Stage-1 outcome:

- `event_v3_navigation_breadcrumbs` was implemented and pushed at
  `d87a6086f535ee4fd892e40638437d7dc873c8cd`.
- Stage-1 ran seeds `0,1,2` for `1,000,008` agent steps with
  `--ent-coef 0.03` and offline W&B.
- Run directories:
  - `/workspace/tribal_event_v3_runs/stage1/stage1_event_v3_navigation_alpha0_seed0_1m_d87a6086f`
  - `/workspace/tribal_event_v3_runs/stage1/stage1_event_v3_navigation_alpha0_seed1_1m_d87a6086f`
  - `/workspace/tribal_event_v3_runs/stage1/stage1_event_v3_navigation_alpha0_seed2_1m_d87a6086f`
- Final deterministic eval still failed:
  - seed 0: eval raw `-10.98`, eval shaping `-3.82`, JS diversity `0.000053`;
  - seed 1: eval raw `-10.28`, eval shaping `-3.91`, JS diversity `0.000122`;
  - seed 2: eval raw `-10.06`, eval shaping `-3.82`, JS diversity `0.000209`.
- Behavior-gate rollouts under `/workspace/tribal_event_v3_runs/behavior_gate`
  showed the same core failure:
  - seed 0 deterministic mostly repeated move action `15`, with invalid
    fraction `0.92`;
  - seed 1 deterministic mostly repeated use action `31`, with invalid
    fraction `1.00`;
  - seed 2 deterministic repeated use action `30`, with invalid fraction
    `1.00`;
  - stochastic rollouts were still random-like, with invalid fraction around
    `0.72-0.77`.

Decision:

- `event_v3_navigation_breadcrumbs` is not behavior-valid.
- Do not scale reward-only training from this point.
- The next intervention is action masking: expose a simulator-derived
  per-agent mask of actions that can currently succeed, train with
  `--use-action-mask`, and apply the same mask during checkpoint behavior
  rollouts.
- Promotion still requires deterministic behavior to beat no-op/random/sweep
  baselines on task events while dropping the invalid fraction materially below
  random.

### Reward-Debug Continuation: Action-Masked `event_v3_navigation_breadcrumbs`

Action masking was implemented and pushed at
`7635f01b78ef0a54aac695669a844a9228baef7e`.

What changed:

- The Tribal Village Nim runtime now exposes a per-agent action-validity mask.
- The canonical reward-geometry trainer can run with `--use-action-mask`.
- PPO samples, evaluates, and updates log-probabilities under the same mask.
- Checkpoint behavior rollouts reload `use_action_mask` from the checkpoint and
  apply the mask during deterministic or stochastic replay.

Local and sandbox smoke status:

- Local native masked smoke passed with `event_v3_navigation_breadcrumbs`.
- Sandbox native CUDA smokes passed on `relh-sandbox-1` and `relh-sandbox-2`.
- Short checkpoint rollout smoke showed the replay path can load masked
  checkpoints on CPU and apply the mask.
- The first CUDA replay smoke failed only because it requested `--device cuda`
  without a GPU-reserved Sky task; rerunning with `--device cpu` fixed it.

Masked Stage-1 training:

| Seed | Sandbox | Path | Agent steps | Status |
| --- | --- | --- | ---: | --- |
| `0` | `relh-sandbox-1` | `/workspace/tribal_event_mask_runs/stage1/seed0` | `1,000,008` | complete |
| `1` | `relh-sandbox-1` | `/workspace/tribal_event_mask_runs/stage1/seed1` | `1,000,008` | complete |
| `2` | `relh-sandbox-2` | `/workspace/tribal_event_mask_runs/stage1/seed2` | `1,000,008` | complete |

Masked Stage-1 final eval summary:

| Seed | Eval raw return | Eval role shaping | Eval total return | EffRank/n | JS diversity | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `-10.68` | `53.29` | `42.61` | `0.157` | `0.174` | `0.464` |
| `1` | `-9.93` | `45.73` | `35.80` | `0.131` | `0.155` | `0.331` |
| `2` | `-10.25` | `77.70` | `67.45` | `0.195` | `0.207` | `0.355` |

Masked Stage-1 behavior gate:

- Behavior artifacts:
  - `relh-sandbox-1:/workspace/tribal_event_mask_runs/behavior_gate_stage1_7635f01b`
  - `relh-sandbox-2:/workspace/tribal_event_mask_runs/behavior_gate_stage1_7635f01b`
- Rollout shape: `3` episodes x `240` steps, with JSONL replays and periodic
  rendered/snapshot frames.
- Baselines on `relh-sandbox-1`:
  - no-op: `0` task events, `1` unique joint action, invalid fraction `0.00`;
  - move-sweep: `0` task events, invalid fraction `0.07`;
  - random: `49.3` mean task events, invalid fraction `0.74`;
  - use-sweep: `3.0` mean task events, invalid fraction `1.00`.
- Deterministic checkpoints:
  - seed 0: `280.0` mean task events, invalid fraction `0.02`,
    `58.3` unique joint actions;
  - seed 1: `268.0` mean task events, invalid fraction `0.01`,
    `46.0` unique joint actions;
  - seed 2: `410.0` mean task events, invalid fraction `0.01`,
    `77.0` unique joint actions.
- Stochastic checkpoints:
  - seed 0: `265.7` mean task events, invalid fraction `0.04`;
  - seed 1: `281.0` mean task events, invalid fraction `0.03`;
  - seed 2: `157.7` mean task events, invalid fraction `0.08`.

Interpretation:

- Training did use shaped rewards to discover behavior, and action masking
  removed the dominant invalid-action local optimum from the earlier reward-only
  attempts.
- This is the first behavior-valid ramp by the task-event gate: deterministic
  checkpoints beat no-op, move-sweep, use-sweep, and random on task events while
  keeping invalid action fractions near zero.
- The policies are still not paper-ready. Raw environment return remains
  negative, no heart deposits were observed, and deterministic policies still
  spend a high fraction of steps on no-op (`0.64-0.77`).
- The main remaining reward-debug target is higher-level objective completion:
  turn resource pickup, crafting, handoff, and tumor-kill behavior into deposit
  or score progress without reintroducing invalid-action collapse.

Next ramp:

- Launch a `10M`-agent-step masked Stage-2 single-condition ramp at
  `shared_frac=0.0`, seeds `0,1,2`.
- Keep the same reward design and mask first; do not retune coefficients until
  the longer run shows whether deposits/score emerge naturally.
- Run the same behavior gate after Stage 2.
- Promote to a reward-mixing pilot only if Stage 2 keeps low invalid fractions,
  keeps high task-event counts, and improves either deposits, raw score, or
  another clear high-level objective metric.
- If Stage 2 plateaus at resource/handoff behavior with no deposits or score,
  add a targeted heart/deposit curriculum or stronger pre-deposit breadcrumb
  rather than scaling directly to billion-step representation experiments.

## Candidate Commands

These commands should be updated after the implementation lands, but this is
the intended shape.

Run the environment/version audit:

```bash
uv run python v3_experiments/audit_tribal_versions.py \
  --metta-root . \
  --coworld-root /Users/relh/Code/coworld-tribal-village \
  --output v3_experiments/behavioral_reward_results/tribal_version_audit.json
```

Run baseline rollouts:

```bash
uv run python v3_experiments/run_tribal_behavior_rollouts.py \
  --policy no_op \
  --episodes 5 \
  --steps 120 \
  --output-dir v3_experiments/behavioral_reward_results/baselines/no_op

uv run python v3_experiments/run_tribal_behavior_rollouts.py \
  --policy random \
  --episodes 5 \
  --steps 120 \
  --output-dir v3_experiments/behavioral_reward_results/baselines/random
```

Run a short event-reward training gate:

```bash
uv run python v3_experiments/train_canonical_reward_geometry.py \
  --reward-design event_v1 \
  --shared-frac 0.0 \
  --seed 0 \
  --total-agent-steps 1000000 \
  --eval-trials 10 \
  --output v3_experiments/behavioral_reward_results/event_v1_alpha0_seed0.json
```

Generate replays for a checkpoint:

```bash
uv run python v3_experiments/run_tribal_behavior_rollouts.py \
  --policy checkpoint \
  --checkpoint-path /path/to/checkpoint.pt \
  --episodes 5 \
  --steps 240 \
  --save-replays \
  --snapshot-every 20 \
  --output-dir v3_experiments/behavioral_reward_results/replays/event_v1_alpha0_seed0
```

Validate outputs:

```bash
uv run python v3_experiments/validate_tribal_behavior_outputs.py \
  --results-dir v3_experiments/behavioral_reward_results
```

Summarize behavior-gate outputs:

```bash
uv run python v3_experiments/summarize_tribal_behavior_outputs.py \
  --results-dir v3_experiments/behavioral_reward_results \
  --output v3_experiments/behavioral_reward_results/behavior_summary.json
```

## Implementation Checklist

- [x] Write the Tribal version audit script.
- [x] Add or expose initial action-level environment event counters.
- [x] Add richer resource, crafting, tumor, lifecycle, and inventory event counters.
- [x] Add inventory and world-state snapshot introspection.
- [x] Add rollout metrics JSON output.
- [x] Add behavior summary/red-flag output.
- [x] Add no-op, random, and simple scripted baseline policies.
- [x] Add replay generation to the rollout harness.
- [x] Implement event-based reward components.
- [x] Add reward component logging.
- [x] Add output validation.
- [x] Run tiny native no-op/random/scripted baseline smoke.
- [ ] Run full no-op/random/scripted baseline gate.
- [ ] Run short `shared_frac=0.0` training gate.
- [ ] Inspect replays and compare behavior metrics.
- [ ] Iterate on rewards until behavior passes.
- [ ] Run reward-mixing pilot.
- [ ] Run canonical 5-seed sweep only after pilot success.
- [ ] Update the paper from canonical outputs only.

## Open Questions

- Should the final paper target remain the 12-agent reconstructed Metta package,
  or should we migrate to the richer Coworld Tribal Village version?
- If we migrate, do we add a controlled 12-agent compatibility mode to Coworld,
  or do we embrace the 48-agent, 8-team game as a new experiment?
- Should the revised three roles be gatherer, crafter/logistics, guardian, or
  should we match the Coworld repo's richer six-agent team roles?
- What raw task score should be the primary behavioral success metric?
- How much task reward should remain shared before applying the experimental
  `shared_frac` mixing?
- Do we want a separate-encoder ablation after behavior passes, or only the
  shared-encoder sweep?

## Non-Negotiables

- Do not update the Overleaf paper from temporary or failed-reward outputs.
- Do not compare the 56-action trained package directly against the 64-action
  Coworld version as if they were the same environment.
- Do not treat role-probe accuracy as meaningful unless the corresponding
  policy passes the behavior gate.
- Do not launch another expensive full sweep before no-op/random/scripted
  baselines and replay inspection are in place.
- Keep every final result tied to a git SHA, exact command, seed list,
  checkpoint path, W&B run, raw JSON artifact, and replay artifact.
