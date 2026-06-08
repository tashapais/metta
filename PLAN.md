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
- [ ] Implement event-based reward components.
- [ ] Add reward component logging.
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
