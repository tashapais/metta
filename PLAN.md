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

## Long-Term Goal

Produce one provenance-clean, behavior-validated Tribal Village representation
result, or determine that the current reconstructed environment cannot support
the paper's MAPPO role-learning claim.

This goal has three hard gates:

1. Behavior first: identify a training protocol where `shared_frac=0.0` reliably
   learns meaningful Tribal Village work across seeds. The policy must beat
   no-op/random on raw task progress and pass replay/behavior validation before
   representation metrics are interpreted.
2. Provenance next: every candidate result must be tied to one git SHA, exact
   command shape, sandbox/job IDs, checkpoint paths, behavior gate root, and
   validation output. Failed and pilot runs stay in this plan/notes, not in the
   paper.
3. Representation last: only after behavior is stable do we run the
   `shared_frac in {0.0, 0.8, 1.0}` sweep and ask whether shared reward mixing
   reduces role-separable geometry, action diversity, and behavior diversity.

Current uncertainty:

- The strict-v10 `shared_frac=0.0` condition is behavior-valid at `2M` under a
  constrained affordance/curriculum mask.
- The 10M strict-v10 condition degraded, so longer training alone is not the
  right answer.
- Action-surface transfer did not yet create useful `put`/handoff behavior.
- A from-scratch strict-v10 reward-mixing pilot did not pass the behavior gate:
  `shared_frac=0.8` was seed-fragile and `shared_frac=1.0` collapsed.
- Warm-starting from behavior-valid alpha0 checkpoints showed that
  `shared_frac=1.0` actively destroys learned chain behavior, while
  `shared_frac=0.8` can preserve stochastic chain behavior but remains
  deterministic-evaluation fragile.
- Shorter/lower-lr alpha0.8 fine-tuning improved deterministic behavior for
  some seeds but did not find a single all-seed deterministic pass condition.
- Lower-alpha thresholding found that `shared_frac=0.6` is the first plausible
  mixed-reward candidate, but one deterministic seed remains weak enough that it
  needs a focused shorter-budget check before promotion.
- The focused alpha0.6 `100k` short-budget check improved the weak seed1
  deterministic behavior from `3` to `34` heart deposits, but still missed the
  all-seed preservation gate, so alpha0.6 is budget-sensitive rather than
  canonical.
- Linear alpha0-to-alpha0.6 annealed transfer passed the 3-seed deterministic
  behavior gate with heart deposits `63,40,63`, making it the first
  behavior-valid mixed-reward candidate in this reconstruction.
- Stage-12 expanded that candidate beyond seeds `0,1,2`, but it exposed a
  source-selection confound: seed `3` and replacement seed `5` were usable at a
  uniform `2M` alpha0 source budget, while contiguous seed `4` only became
  deterministic-source-valid after a `4M` rescue. Therefore Stage 12 is useful
  promotion evidence, but it is not yet a paper-eligible canonical result.
- Stage-13 removed the seed-selection confound with a fixed six-seed, uniform
  `2M` source rerun. It produced meaningful promoted behavior for all six
  seeds, but the strict canonical gate did not pass: source seed `2` was
  deterministic-weak, and promoted seed `0` missed the 75% source-preservation
  threshold by `3` heart deposits (`33` observed vs `36` required).
- Stage-14 tested whether a lower final reward-sharing fraction would recover
  preservation. It did not: alpha0-to-alpha0.5 annealing worsened deterministic
  behavior for seeds `1` and `3`.
- Stage-15 reran behavior gates at higher sample count (`10` episodes x `240`
  steps) for the Stage-13 alpha0 source and alpha0-to-alpha0.6 promoted
  checkpoints. This resolved the 3-episode fragility in the positive direction:
  all six promoted deterministic policies beat no-op/random baselines, produced
  nonzero ore/battery/heart/use chain work, and passed the 75% source-heart
  preservation rule. The Stage-13 alpha0.6 stream is now behavior-valid enough
  to run representation/probe analysis, with source seed `2` still noted as the
  weakest acquisition seed.
- Stage-16 aggregated the canonical result JSONs for the behavior-valid
  Stage-13 alpha0 source and alpha0-to-alpha0.6 promoted checkpoints. This is
  the first behavior-backed geometry readout. It does not recover the original
  fixed-role probe claim: role-probe accuracy is near the 1/3 chance baseline
  for alpha0 (`0.348 +/- 0.022`) and alpha0.6 (`0.309 +/- 0.021`). The
  defensible signal is currently behavioral recovery plus geometry/action
  changes, not clean fixed-role separability.
- Stage-17 audited behavior-derived roles from the Stage-15 per-agent simulator
  counters. The recovered behavior is chain competence without specialization:
  all deterministic checkpoint agents complete all chain stages, every
  agent-rollout is ore-dominant by event count, and fixed `agent_id % 3` groups
  do not separate by ore/battery/heart stage. This supports revising the paper
  away from a learned-role claim rather than searching for a behavior-derived
  role label to rescue it.
- Stage-18 updated the Overleaf paper from the behavior-backed canonical
  evidence stream. The paper now states the negative result: reward redesign
  recovers meaningful chain behavior, but the reconstructed setup does not
  support the original MAPPO fixed-role specialization claim.

Immediate operating plan:

- Phase A: completed. The strict-v10 budget ladder identified `2M` alpha0 as
  the cleanest behavior source.
- Phase B: completed for broad mask relaxation and `put`-only relaxation; neither
  is a promotion candidate.
- Phase C: completed the from-scratch `shared_frac in {0.8,1.0}` pilot; it
  failed the behavior gate and should not update the paper.
- Phase D: run one warm-start reward-mixing diagnostic from the behavior-valid
  strict alpha0 2M checkpoints to distinguish acquisition failure from
  reward-sharing-induced behavior destruction. Completed: alpha1.0 destroys the
  chain; alpha0.8 is stochastic-valid but deterministic-fragile.
- Phase E: run a short/low-lr alpha0.8 stabilization diagnostic to see whether
  deterministic behavior can be preserved while changing reward mixing.
  Completed: stochastic behavior remains strong, deterministic behavior is still
  seed/budget fragile.
- Phase F: run a lower-alpha threshold diagnostic to test whether reward sharing
  below `0.8` can preserve deterministic behavior. Completed: alpha0.6 is a
  partial pass with a weak seed1 deterministic rollout.
- Phase G: run a focused alpha0.6 shorter-budget check to see whether the weak
  seed1 behavior is a budget/overtraining issue. Completed: `100k` is much
  better than `250k` for seed1 but still misses the strict gate.
- Phase H: run an alpha0-to-alpha0.6 annealed-transfer diagnostic to test
  whether gradual reward sharing preserves the chain better than fixed alpha.
  Completed: annealing passed the 3-seed deterministic behavior gate.
- Phase I: promote annealed alpha0.6 into a canonical follow-up: expand seeds,
  preserve behavior gates, and run representation/probe analysis from the
  behavior-valid checkpoints. Completed as a diagnostic, not as a paper stream:
  the seed4 source-budget confound requires a cleaner rerun.
- Phase J: pre-register and run a Stage-13 canonical rerun under one SHA with a
  fixed source seed/budget rule before looking at promoted results. Completed:
  the primary Stage-13 stream is behavior-positive but misses the strict
  canonical gate.
- Phase K: run one conservative alpha0-to-alpha0.5 annealed follow-up on the
  same Stage-13 source checkpoints to test whether the seed0 preservation miss
  is final-alpha sensitive. Completed: alpha0.5 worsened deterministic behavior
  and is not a promotion candidate.
- Phase L: run a higher-N behavior audit for Stage-13 alpha0 sources and
  alpha0.6 promoted checkpoints to measure whether the seed0 preservation miss
  and seed2 source weakness persist beyond the 3-episode gate. Completed:
  higher-N behavior passes for the Stage-13 alpha0.6 stream.
- Phase M: run representation/probe analysis for the behavior-valid Stage-13
  alpha0 source and alpha0-to-alpha0.6 promoted checkpoints, preserving exact
  source/promotion provenance and separating behavior-valid alpha0.6 from failed
  alpha0.5, alpha0.8, and alpha1.0 diagnostics. Completed for the first
  behavior-backed Stage-13 readout.
- Phase N: decide the paper-safe representation claim. Current evidence supports
  sensible chain behavior and nontrivial action/geometry metrics, but not the
  original fixed `agent_id % 3` MAPPO role-probe claim. Completed Stage-17
  behavior-derived role audit: the behavior itself is not role-specialized.
- Phase O: update the paper only after the Stage-16 interpretation and any
  necessary follow-up analyses are stable. Completed in Overleaf commit
  `eef90a1`.

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

Masked Stage-2 outcome:

- Stage-2 ran at `shared_frac=0.0` for `10,000,008` agent steps with
  `event_v3_navigation_breadcrumbs`, `--use-action-mask`, and offline W&B.
- Branch/worktree commit: `5af510995562d229d34815b2bce0b4dbf59d0692`.
- Run directories:
  - `/workspace/tribal_event_mask_runs/stage2_10m/seed0`
  - `/workspace/tribal_event_mask_runs/stage2_10m/seed1`
  - `/workspace/tribal_event_mask_runs/stage2_10m/seed2`
- Result JSON validation passed with `--allow-smoke` for all three seeds.

Stage-2 final eval summary:

| Seed | Eval raw return | Eval role shaping | Eval total return | EffRank/n | JS diversity | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `-10.35` | `66.13` | `55.78` | `0.144` | `0.196` | `0.342` |
| `1` | `-10.85` | `30.16` | `19.31` | `0.134` | `0.351` | `0.377` |
| `2` | `-9.92` | `23.01` | `13.08` | `0.177` | `0.089` | `0.431` |

Stage-2 behavior gate:

- Behavior artifacts:
  - `relh-sandbox-1:/workspace/tribal_event_mask_runs/behavior_gate_stage2_10m_5af510995`
  - `relh-sandbox-2:/workspace/tribal_event_mask_runs/behavior_gate_stage2_10m_5af510995`
- Rollout shape: `3` episodes x `240` steps, with JSONL replays and periodic
  rendered/snapshot frames.
- Baselines on `relh-sandbox-1`:
  - no-op: `0` task events;
  - move-sweep: `0` task events, invalid fraction `0.06`;
  - random: `28.3` mean task events, invalid fraction `0.73`;
  - use-sweep: `21.0` mean task events, invalid fraction `0.99`.
- Deterministic checkpoints:
  - seed 0: `364.7` mean task events, invalid fraction `0.03`,
    `56.0` unique joint actions;
  - seed 1: `149.7` mean task events, invalid fraction `0.17`,
    `79.7` unique joint actions;
  - seed 2: `219.0` mean task events, invalid fraction `0.01`,
    `71.3` unique joint actions.
- Stochastic checkpoints:
  - seed 0: `235.3` mean task events, invalid fraction `0.04`;
  - seed 1: `263.7` mean task events, invalid fraction `0.15`;
  - seed 2: `295.0` mean task events, invalid fraction `0.02`.
- Detailed counters showed:
  - resource pickup, crafting, handoff, and combat are present;
  - seed 0 deterministic crafted only `1` battery across the gate;
  - seed 1 and seed 2 deterministic crafted `0` batteries;
  - all Stage-2 checkpoint gates produced `0` heart deposits.

Stage-2 decision:

- Do not promote this reward to reward-mixing or representation sweeps yet.
- The mask plus `event_v3_navigation_breadcrumbs` is meaningfully training
  low-invalid task-event behavior, but it is not yet achieving the core
  high-level objective.
- The next reward-debug phase needs more explicit breadcrumbs for the heart
  chain: ore pickup, converter use to craft battery, and battery use at the
  home assembler to deposit a heart.

### Reward-Debug Continuation: `event_v4_heart_chain_breadcrumbs`

Reason:

- `event_v3_navigation_breadcrumbs` over-rewards easy loops relative to the
  actual score chain. In Stage 2, policies learned resource pickup, armor/bread
  handoff, and tumor kills, but not battery production or heart deposits.
- The simulator contract for the score chain is:
  - `use` a mine to collect ore;
  - `use` a converter while carrying ore to craft a battery;
  - `use` the home assembler while carrying a battery to deposit a heart.

Design:

- Keep action masking and capped movement.
- Reduce easy handoff shaping:
  - `put_armor` and `put_bread` become small breadcrumbs, not dominant reward.
- Strongly prioritize the heart chain:
  - `resource_ore` is larger than other resource pickups;
  - `craft_battery` is much larger than other crafting events;
  - `deposit_heart` dominates all other shaped rewards.
- Keep small defense rewards so agents do not ignore tumors completely.
- Keep penalties for no-op and invalid actions so the mask cannot become a
  reason to park.

Implementation status:

- `event_v4_heart_chain_breadcrumbs` was added locally after the Stage-2
  analysis.
- Focused unit tests and a mock training smoke passed.
- A tiny native Tribal Village smoke passed:
  - output: `/tmp/event_v4_heart_chain_native_smoke.json`;
  - checkpoint: `/tmp/event_v4_heart_chain_native_smoke.pt`;
  - shaped return was positive on the smoke path.

Next ramp:

- Commit and push `event_v4_heart_chain_breadcrumbs`.
- Update the sandbox worktrees to the new commit.
- Run a short Stage-1 v4 masked ramp before scaling:
  - `shared_frac=0.0`;
  - seeds `0,1,2`;
  - `1,000,008` agent steps;
  - same behavior gate as Stage 1 and Stage 2.
- Promotion criteria are stricter than the Stage-1 v3 mask gate:
  - invalid fraction stays below random/use-sweep;
  - task events beat baselines;
  - at least one deterministic or stochastic checkpoint crafts batteries
    consistently;
  - promotion to a longer run requires nonzero heart deposits or a clear
    monotonic increase in battery production over Stage 2.

V4 Stage-1 outcome:

- Stage-1 ran at `shared_frac=0.0` for `1,000,008` agent steps with
  `event_v4_heart_chain_breadcrumbs`, `--use-action-mask`, and offline W&B.
- Branch/worktree commit: `6a54d233dd10f435d957e1882082933aceaf27bd`.
- Run directories:
  - `/workspace/tribal_event_mask_runs/stage1_v4_heart_chain_1m/seed0`
  - `/workspace/tribal_event_mask_runs/stage1_v4_heart_chain_1m/seed1`
  - `/workspace/tribal_event_mask_runs/stage1_v4_heart_chain_1m/seed2`
- Result JSON validation passed with `--allow-smoke` for all three seeds.

V4 Stage-1 final eval summary:

| Seed | Eval raw return | Eval role shaping | Eval total return | EffRank/n | JS diversity | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `-11.37` | `5.43` | `-5.95` | `0.147` | `0.316` | `0.377` |
| `1` | `-10.53` | `2.64` | `-7.89` | `0.102` | `0.312` | `0.386` |
| `2` | `-10.20` | `3.38` | `-6.81` | `0.107` | `0.224` | `0.392` |

V4 Stage-1 behavior gate:

- Behavior artifacts:
  - `relh-sandbox-1:/workspace/tribal_event_mask_runs/behavior_gate_stage1_v4_heart_chain_6a54d233d`
  - `relh-sandbox-2:/workspace/tribal_event_mask_runs/behavior_gate_stage1_v4_heart_chain_6a54d233d`
- Rollout shape: `3` episodes x `240` steps, with JSONL replays and periodic
  rendered/inventory/world snapshots.
- Baselines on `relh-sandbox-1`:
  - no-op: `0` task events;
  - move-sweep: `0` task events, invalid fraction `0.12`;
  - random: `35.3` mean task events, invalid fraction `0.73`;
  - use-sweep: `3.7` mean task events, invalid fraction `1.00`.
- Deterministic checkpoints:
  - seed 0: `229.7` mean task events, invalid fraction `0.08`,
    `0` battery crafts, `0` deposits;
  - seed 1: `195.3` mean task events, invalid fraction `0.07`,
    `0` battery crafts, `0` deposits;
  - seed 2: `135.0` mean task events, invalid fraction `0.12`,
    `0` battery crafts, `0` deposits.
- Stochastic checkpoints:
  - seed 0: `156.3` mean task events, invalid fraction `0.01`,
    `7` battery crafts, `0` deposits;
  - seed 1: `100.0` mean task events, invalid fraction `0.02`,
    `6` battery crafts, `0` deposits;
  - seed 2: `128.3` mean task events, invalid fraction `0.02`,
    `0` battery crafts, `0` deposits.

V4 Stage-1 decision:

- Do not promote v4 to a longer representation run.
- V4 did improve part of the heart chain under stochastic sampling: seeds `0`
  and `1` crafted batteries during the behavior gate.
- V4 still failed the actual objective because every checkpoint produced `0`
  heart deposits.
- The failure is now more specific than the earlier v3 failure: policies can
  discover ore and sometimes converter use, but they are not reliably returning
  battery carriers to the home assembler and using the assembler when ready.

### Reward-Debug Continuation: `event_v5_navigation_chain_breadcrumbs`

Reason:

- The simulator scoring contract requires `use` on an adjacent home assembler
  while carrying a battery and while the assembler cooldown is ready.
- V4 rewarded the terminal events heavily, but it did not provide a dense
  breadcrumb for the navigation leg from battery carrier back to home assembler.
- The built-in Tribal Village scripted AI handles this explicitly: after an
  agent crafts a battery, it navigates toward `agent.homeassembler` and uses the
  assembler when adjacent.

Design:

- Add a small simulator introspection hook,
  `tribal_village_get_navigation_snapshot`, with one row per agent:
  - agent position;
  - home assembler position;
  - nearest converter position;
  - nearest mine position;
  - distances to home assembler, nearest converter, and nearest mine;
  - ore and battery inventory counts.
- Keep action masking and terminal event rewards.
- Add inventory-conditioned progress rewards:
  - empty-handed agents get a small capped reward for reducing distance to the
    nearest mine;
  - ore carriers get a stronger reward for reducing distance to the nearest
    converter;
  - battery carriers get the strongest dense reward for reducing distance to
    the home assembler;
  - arriving adjacent to the home assembler with a battery gets a one-step
    breadcrumb, but repeated waiting does not pay.
- Continue saving behavior-gate replay snapshots; saved frames now include the
  optional navigation snapshot so we can inspect whether battery carriers are
  near the assembler before deciding on another coefficient change.

Implementation status:

- `event_v5_navigation_chain_breadcrumbs` has been implemented locally.
- Focused unit tests pass, including navigation-progress reward math.
- A tiny native Tribal Village smoke passed and verified the navigation
  snapshot shape `(12, 13)`.

Next ramp:

- Commit and push `event_v5_navigation_chain_breadcrumbs`.
- Update the sandbox worktrees to the new commit.
- Run a short Stage-1 v5 masked ramp before scaling:
  - `shared_frac=0.0`;
  - seeds `0,1,2`;
  - `1,000,008` agent steps;
  - same behavior gate as v4, with navigation snapshots in saved replay frames.
- Promotion criteria:
  - invalid fraction remains below random/use-sweep;
  - stochastic and deterministic gates produce battery crafts;
  - at least one seed produces nonzero heart deposits, or the replay snapshots
    show battery carriers repeatedly reaching a ready home assembler but failing
    only on the final `use` action.
- If v5 still produces batteries but no deposits, the next intervention should
  be either a short scripted/imitation warm start for the home-assembler leg or
  a curriculum map with shorter mine -> converter -> assembler distances.

V5 Stage-1 outcome:

- Stage-1 ran at `shared_frac=0.0` for `1,000,008` agent steps with
  `event_v5_navigation_chain_breadcrumbs`, `--use-action-mask`, and offline
  W&B.
- Branch/worktree commit: `44e889d63d6474f4286315aacd8a3afc4cfac3f0`.
- Run directories:
  - `/workspace/tribal_event_mask_runs/stage1_v5_navigation_chain_1m/seed0`
  - `/workspace/tribal_event_mask_runs/stage1_v5_navigation_chain_1m/seed1`
  - `/workspace/tribal_event_mask_runs/stage1_v5_navigation_chain_1m/seed2`
- Result JSON validation passed with `--allow-smoke` for all three seeds.

V5 Stage-1 final eval summary:

| Seed | Eval raw return | Eval role shaping | Eval total return | EffRank/n | JS diversity | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `-10.66` | `2.07` | `-8.58` | `0.138` | `0.295` | `0.377` |
| `1` | `-10.23` | `2.30` | `-7.93` | `0.128` | `0.378` | `0.388` |
| `2` | `-10.52` | `1.72` | `-8.80` | `0.146` | `0.420` | `0.321` |

V5 Stage-1 behavior gate:

- Behavior artifacts:
  - `relh-sandbox-1:/workspace/tribal_event_mask_runs/behavior_gate_stage1_v5_navigation_chain_44e889d63`
  - `relh-sandbox-2:/workspace/tribal_event_mask_runs/behavior_gate_stage1_v5_navigation_chain_44e889d63`
- Rollout shape: `3` episodes x `240` steps, with JSONL replays and periodic
  rendered/inventory/world/navigation snapshots.
- Baselines on `relh-sandbox-1`:
  - no-op: `0` task events;
  - move-sweep: `0` task events, invalid fraction `0.07`;
  - random: `43.3` mean task events, invalid fraction `0.73`;
  - use-sweep: `6.7` mean task events, invalid fraction `1.00`.
- Deterministic checkpoints:
  - seed 0: `117.0` mean task events, invalid fraction `0.28`,
    `0` battery crafts, `0` deposits;
  - seed 1: `33.0` mean task events, invalid fraction `0.20`,
    `0` battery crafts, `0` deposits;
  - seed 2: `89.7` mean task events, invalid fraction `0.39`,
    `0` battery crafts, `0` deposits.
- Stochastic checkpoints:
  - seed 0: `172.0` mean task events, invalid fraction `0.02`,
    `5` battery crafts, `0` deposits;
  - seed 1: `137.3` mean task events, invalid fraction `0.03`,
    `0` battery crafts, `0` deposits;
  - seed 2: `82.3` mean task events, invalid fraction `0.14`,
    `3` battery crafts, `1` deposit.
- Replay evidence for the deposit:
  - seed 2 stochastic, episode `1`, step `10`: agent `10` crafted a battery
    via `use`, receiving raw reward `0.79` after the step cost;
  - seed 2 stochastic, episode `1`, step `57`: agent `10` used the assembler
    and received raw reward `0.99`, confirming a heart deposit.

V5 Stage-1 decision:

- V5 is the first reward-debug setup to produce a nonzero heart deposit.
- Do not promote to shared-fraction representation sweeps yet. The behavior is
  still weak: deposits occurred only once, only under stochastic sampling, and
  deterministic rollouts still did not craft batteries or deposit hearts.
- Promote v5 to a `10M`-agent-step Stage-2 debug run at `shared_frac=0.0` for
  seeds `0,1,2`.
- Stage-2 promotion criteria:
  - nonzero deposits in deterministic or repeatable stochastic gates;
  - more than isolated one-off battery crafts;
  - invalid fraction remains below random/use-sweep;
  - raw return improves relative to no-op/move-sweep, not just shaped return.
- If Stage 2 does not increase deposits, switch to either:
  - a short scripted/imitation warm start for the ore -> converter -> assembler
    chain; or
  - a curriculum/debug map with shorter mine -> converter -> assembler
    distances before returning to the full canonical map.

V5 Stage-2 outcome:

- Stage-2 ran at `shared_frac=0.0` for `10,000,008` agent steps with
  `event_v5_navigation_chain_breadcrumbs`, `--use-action-mask`, and offline
  W&B.
- Training/checkpoint commit: `7ecf4453b649f26deb75219bfcdcc5405ff2df3b`.
- Result root:
  `/workspace/tribal_event_mask_runs/stage2_v5_navigation_chain_10m`.
- Result JSON validation passed with `--allow-smoke` for all three seeds.

V5 Stage-2 final eval summary:

| Seed | Eval raw return | Eval role shaping | Eval total return | EffRank/n | JS diversity | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `-13.53` | `12.32` | `-1.21` | `0.578` | `0.328` | `0.370` |
| `1` | `-13.19` | `8.70` | `-4.48` | `0.374` | `0.243` | `0.405` |
| `2` | `-12.57` | `6.40` | `-6.17` | `0.329` | `0.304` | `0.449` |

V5 Stage-2 behavior gate:

- Behavior-gate commit: `0351e1196ea2b6da9e6b824064ebe746dc47f636`.
- Behavior artifacts:
  - `relh-sandbox-1:/workspace/tribal_event_mask_runs/behavior_gate_stage2_v5_navigation_chain_0351e1196`
  - `relh-sandbox-2:/workspace/tribal_event_mask_runs/behavior_gate_stage2_v5_navigation_chain_0351e1196`
- Rollout shape: `3` episodes x `240` steps, with JSONL replays and periodic
  rendered/inventory/world/navigation snapshots.
- New diagnostic baseline:
  - `chain_oracle` reads the navigation snapshot and action mask, then follows
    the ore -> battery -> home-assembler chain;
  - local smoke produced `49` heart deposits over `3` episodes;
  - sandbox behavior gate produced `51` heart deposits, `70` battery crafts,
    and `73` ore pickups, proving the canonical simulator can produce the
    desired behavior under the current contract.
- Baselines:
  - no-op: `0` task events, `0` deposits;
  - move-sweep: `0` task events, `0` deposits;
  - random: `20.3` mean task events, `3` crafts, `0` deposits;
  - use-sweep: `14.7` mean task events, `6` armor crafts, `0` deposits.
- Stage-2 checkpoints:
  - seed 0 deterministic: `168.0` mean task events, `15` ore pickups,
    `0` battery crafts, `0` deposits;
  - seed 0 stochastic: `178.3` mean task events, `25` ore pickups,
    `0` battery crafts, `0` deposits;
  - seed 1 deterministic: `165.7` mean task events, `35` ore pickups,
    `0` battery crafts, `0` deposits;
  - seed 1 stochastic: `187.0` mean task events, `54` ore pickups,
    `2` battery crafts, `0` deposits;
  - seed 2 deterministic: `112.3` mean task events, `14` ore pickups,
    `0` battery crafts, `0` deposits;
  - seed 2 stochastic: `176.3` mean task events, `53` ore pickups,
    `0` battery crafts, `0` deposits.

V5 Stage-2 decision:

- V5 does not pass the behavioral validity gate.
- The failure is not a simulator impossibility: the chain oracle deposits
  hearts repeatedly under the same map, action space, and rollout length.
- The learned checkpoints discovered task activity, but it was mostly off-chain:
  water/wheat/wood collection, armor/spear/bread/lantern crafting, and tumor
  combat. The heart chain remained weak: almost no battery production and zero
  deposits across all Stage-2 checkpoint gates.
- Do not run reward-mixing or representation sweeps from v5.
- The next reward-debug step should strip off-chain shaped rewards and add
  explicit mask-valid breadcrumbs for the exact ore -> battery -> home-assembler
  chain.

### Reward-Debug Continuation: `event_v6_oracle_chain_breadcrumbs`

Purpose:

- Convert the v5 failure into a narrower debug target: learn the same chain
  that the diagnostic oracle can execute.
- Keep this as a behavior-discovery reward, not as a final paper reward.
  Once chain completion is reliable, we can decide whether to reintroduce
  role-specific terms for the representation experiment.

Design:

- Remove off-chain shaped task rewards:
  - no shaped reward for water, wheat, wood;
  - no shaped reward for spear, lantern, armor, bread;
  - no shaped reward for tumor/spawner combat.
- Keep only heart-chain event rewards:
  - `resource_ore`: `1.00`;
  - `craft_battery`: `8.00`;
  - `deposit_heart`: `40.00`.
- Keep inventory-conditioned navigation progress:
  - empty-handed agents toward nearest mine: `0.08` per positive distance step;
  - ore carriers toward nearest converter: `0.80` per positive distance step;
  - battery carriers toward home assembler: `1.50` per positive distance step;
  - first arrival adjacent to the home assembler while carrying a battery:
    `0.50`.
- Add mask-valid oracle-action breadcrumbs:
  - `move_toward_chain_target`: `0.05` when the chosen move matches the best
    valid move toward the current chain target;
  - `use_chain_target`: `1.00` when adjacent to the current chain target and
    the chosen `use` action is allowed by the pre-step action mask.
- Keep small anti-collapse terms:
  - `action_noop`: `-0.010`;
  - `action_invalid`: `-0.020`.

Implementation status:

- `event_v6_oracle_chain_breadcrumbs` has been implemented locally.
- The trainer now passes chosen actions and pre-step action masks into reward
  computation so the oracle-use breadcrumb cannot pay for cooldown-invalid use
  spam.
- Focused tests pass, including v6 reward math and trainer metadata.
- A tiny native Tribal Village smoke passed:
  - output: `/tmp/event_v6_oracle_chain_native_smoke.json`;
  - checkpoint: `/tmp/event_v6_oracle_chain_native_smoke.pt`;
  - validation: `validate_canonical_reward_geometry_results.py --allow-smoke`.

V6 Stage-1 outcome:

- Stage-1 ran at `shared_frac=0.0` for `1,000,008` agent steps with
  `--use-action-mask`, seeds `0,1,2`, and the v6 chain-only reward.
- Final eval summary:
  - seed `0`: raw `-11.3213`, shaping `11.9934`, total `0.6721`,
    effrank/agent `0.1068`, JS action diversity `0.3461`, role probe `0.4139`;
  - seed `1`: raw `-11.1030`, shaping `24.1995`, total `13.0965`,
    effrank/agent `0.1089`, JS action diversity `0.3402`, role probe `0.3033`;
  - seed `2`: raw `-10.8423`, shaping `22.1815`, total `11.3392`,
    effrank/agent `0.1412`, JS action diversity `0.3299`, role probe `0.3527`.
- Behavior gate summary:
  - diagnostic `chain_oracle`: `27` heart deposits, `49` battery crafts,
    `58` ore pickups across the same rollout horizon;
  - seed `0` deterministic/stochastic checkpoints: `0` battery crafts and
    `0` heart deposits;
  - seed `1` deterministic checkpoint: `1` battery craft and `0` deposits;
  - seed `1` stochastic checkpoint: `0` battery crafts and `0` deposits;
  - seed `2` deterministic checkpoint: `1` battery craft and `0` deposits;
  - seed `2` stochastic checkpoint: `0` battery crafts and `0` deposits.
- Decision: v6 still fails the meaningful-behavior gate. It proves reward
  terms can generate shaped return and sometimes battery crafts, but the policy
  still does not complete deposits. The crucial clue is that `chain_oracle`
  uses privileged global target positions from the navigation snapshot, while
  the learned policy receives only a local feed-forward observation. This makes
  the next debugging step an observability breadcrumb, not another coefficient
  tweak.

### Reward-Debug Continuation: `event_v7_chain_compass_breadcrumbs`

Purpose:

- Keep the v6 chain-only reward so we do not reintroduce off-chain incentives.
- Add a minimal observation breadcrumb that makes the currently shaped chain
  target visible to the policy.
- Treat this as a debug/curriculum intervention. It is not the final paper
  reward until it passes the behavior gate and we understand how much of the
  result comes from the observation breadcrumb.

Design:

- Reuse v6 reward coefficients:
  - `resource_ore`: `1.00`;
  - `craft_battery`: `8.00`;
  - `deposit_heart`: `40.00`;
  - inventory-conditioned progress and mask-valid oracle-action bonuses
    unchanged from v6.
- Append five uint8 observation planes to the canonical 21-channel local
  observation:
  - `chain_target_dx_sign`: west/aligned/east target direction;
  - `chain_target_dy_sign`: north/aligned/south target direction;
  - `chain_inventory_stage`: empty -> mine, ore -> converter,
    battery -> home assembler;
  - `chain_target_closeness`: clipped Chebyshev closeness to current target;
  - `chain_target_adjacent`: whether the current target is adjacent and a
    `use` action is plausibly relevant.
- Resulting checkpoint observation contract is `[26, 11, 11]` instead of
  `[21, 11, 11]`.
- Behavior rollout must pass `--chain-compass-observation` for v7 checkpoints;
  the rollout loader now checks `obs_shape` before loading model weights.

Implementation status:

- `event_v7_chain_compass_breadcrumbs` has been implemented locally.
- Focused validation passed:
  - `uv run ruff check ...`;
  - `uv run ruff format --check ...`;
  - `uv run pytest tests/v3_experiments/test_canonical_reward_geometry.py
    tests/v3_experiments/test_tribal_behavior_tools.py -q`;
  - result: `41 passed`.
- Tiny native Tribal Village smoke passed:
  - output: `/tmp/event_v7_chain_compass_native_smoke.json`;
  - checkpoint: `/tmp/event_v7_chain_compass_native_smoke.pt`;
  - recorded `obs_shape: [26, 11, 11]`;
  - recorded `chain_compass_observation: true`.

Next ramp:

- Commit and push `event_v7_chain_compass_breadcrumbs`.
- Update sandbox worktrees to the v7 commit.
- Run a short Stage-1 v7 masked ramp:
  - `shared_frac=0.0`;
  - seeds `0,1,2`;
  - `1,000,008` agent steps;
  - `--use-action-mask`;
  - `--reward-design event_v7_chain_compass_breadcrumbs`.
- Run the same behavior gate as v6, but evaluate v7 checkpoints with
  `--chain-compass-observation`.
- Promotion criteria:
  - repeated battery crafts across seeds;
  - nonzero heart deposits in deterministic or repeatable stochastic rollouts;
  - trained checkpoints close part of the gap to `chain_oracle`, not merely
    random/use-sweep;
  - off-chain events stay secondary to ore, battery, and deposit chain events.
- If v7 still does not produce deposits, stop coefficient tweaking and move to
  an explicit curriculum or warm start:
  - shorter mine -> converter -> assembler distances; or
  - a short imitation/behavior-cloning warm start from the chain oracle before
    MAPPO fine-tuning.

V7 Stage-1 outcome:

- Stage-1 ran at `shared_frac=0.0` for `1,000,008` agent steps with
  `--use-action-mask`, `--chain-compass-observation`, seeds `0,1,2`, and the
  v7 chain-compass reward.
- Result root:
  `/workspace/tribal_event_mask_runs/stage1_v7_chain_compass_1m`.
- Result validation:
  - `validate_canonical_reward_geometry_results.py --allow-smoke` passed for
    all three result JSONs.
- Final eval summary:
  - seed `0`: raw `-9.1662`, shaping `164.2705`, total `155.1043`,
    effrank/agent `0.1864`, JS action diversity `0.2868`, role probe `0.3491`;
  - seed `1`: raw `0.1002`, shaping `371.7712`, total `371.8713`,
    effrank/agent `0.1844`, JS action diversity `0.4025`, role probe `0.3868`;
  - seed `2`: raw `1.2093`, shaping `367.6329`, total `368.8422`,
    effrank/agent `0.2003`, JS action diversity `0.3297`, role probe `0.2980`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage1_v7_chain_compass_8f1ac1751`.
- Behavior-output validation passed:
  - sandbox 1: `9` rollout files;
  - sandbox 2: `2` rollout files.
- Behavior gate summary:
  - diagnostic `chain_oracle`: `58` heart deposits, `83` battery crafts,
    `87` ore pickups;
  - `random`: `0` deposits and mostly invalid actions;
  - `use_sweep`: `0` deposits and mostly invalid actions;
  - seed `0` deterministic checkpoint: `9` heart deposits, `29` battery crafts,
    `48` ore pickups;
  - seed `0` stochastic checkpoint: `19` heart deposits, `32` battery crafts,
    `69` ore pickups;
  - seed `1` deterministic checkpoint: `35` heart deposits, `63` battery crafts,
    `67` ore pickups;
  - seed `1` stochastic checkpoint: `25` heart deposits, `31` battery crafts,
    `58` ore pickups;
  - seed `2` deterministic checkpoint: `30` heart deposits, `40` battery crafts,
    `70` ore pickups;
  - seed `2` stochastic checkpoint: `18` heart deposits, `21` battery crafts,
    `43` ore pickups.
- Decision: v7 breaks the previous training failure. The policies are no
  longer merely accumulating shaped return or collecting off-chain resources;
  every seed has deterministic and stochastic checkpoint rollouts with nonzero
  heart deposits.
- Caveat: v7 is still a debug/curriculum intervention. Learned policies
  continue to collect water/wheat/wood and craft armor/bread/lantern/spear in
  addition to the heart chain, and raw environment reward remains negative for
  most behavior-gate rollouts. Treat v7 as evidence that meaningful chain
  behavior is trainable with an observation breadcrumb, not as the final paper
  condition.
- Next research step:
  - archive v7 replays for qualitative inspection;
  - run a longer v7 ramp if behavior remains stable under replay review;
  - only then rerun representation analyses, clearly labeling v7 as the
    chain-compass/curriculum condition rather than the original MAPPO-only
    reward-shaping condition.

### Reward-Debug Continuation: `event_v8_clean_chain_compass_breadcrumbs`

Purpose:

- Preserve the v7 result that made the heart chain trainable.
- Test whether the remaining off-chain activity is incidental exploration or a
  stable local optimum.
- Keep v8 as a debug/curriculum condition, not a final paper condition.

Design:

- Reuse the v7 observation intervention:
  - canonical local observation plus five chain-compass planes;
  - checkpoint observation contract remains `[26, 11, 11]`.
- Reuse v7/v6 chain rewards:
  - `resource_ore`: `1.00`;
  - `craft_battery`: `8.00`;
  - `deposit_heart`: `40.00`;
  - inventory-conditioned progress and mask-valid oracle-action bonuses
    unchanged.
- Add small negative coefficients for successful off-chain events seen in v7:
  - water/wheat/wood pickups: `-0.10`;
  - armor/bread/lantern/spear crafts: `-1.00`;
  - armor/bread handoffs: `-0.50`;
  - tumor/spawner/agent kills and lantern plants: `-0.25`.
- The penalties are intentionally much smaller than chain rewards, so a clean
  deposit trajectory should remain strongly preferred over avoiding all action.

V8 promotion criteria:

- Do not regress the v7 deposit gate:
  - every seed should still have nonzero deposits in deterministic or
    stochastic behavior rollouts;
  - ideally deterministic deposits stay near the v7 range of `9-35` hearts over
    the 3-episode behavior gate.
- Off-chain events should decrease relative to v7:
  - fewer water/wheat/wood pickups;
  - fewer non-battery crafts;
  - less combat.
- If deposits collapse, revert to v7 for representation work and treat off-chain
  cleanup as a separate curriculum problem.

V8 Stage-1 outcome:

- Implementation commit: `73b7910af`.
- Remote worktrees were staged on both sandboxes at commit `73b7910af`.
- Tiny native Tribal Village smokes passed on both sandboxes:
  - sandbox 1 output:
    `/workspace/tribal_event_mask_runs/smoke_v8_sandbox1.json`;
  - sandbox 2 output:
    `/workspace/tribal_event_mask_runs/smoke_v8_sandbox2.json`;
  - both recorded `reward_design:
    event_v8_clean_chain_compass_breadcrumbs`;
  - both recorded `obs_shape: [26, 11, 11]`;
  - both recorded `chain_compass_observation: true`.
- Stage-1 ran at `shared_frac=0.0` for `1,000,008` agent steps with
  `--use-action-mask`, `--chain-compass-observation`, seeds `0,1,2`, and the
  v8 clean-chain reward.
- Result root:
  `/workspace/tribal_event_mask_runs/stage1_v8_clean_chain_compass_1m`.
- Result validation:
  - `validate_canonical_reward_geometry_results.py --allow-smoke` passed for
    all three result JSONs.
- Final eval summary:
  - seed `0`: raw `-7.5573`, shaping `124.2987`, total `116.7414`,
    effrank/agent `0.1674`, JS action diversity `0.3160`, role probe `0.2466`;
  - seed `1`: raw `-11.9144`, shaping `21.6363`, total `9.7218`,
    effrank/agent `0.2616`, JS action diversity `0.3653`, role probe `0.3406`;
  - seed `2`: raw `1.0826`, shaping `313.8023`, total `314.8849`,
    effrank/agent `0.1306`, JS action diversity `0.3031`, role probe `0.3727`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage1_v8_clean_chain_compass_73b7910af`.
- Behavior-output validation passed:
  - sandbox 1: `9` rollout files;
  - sandbox 2: `2` rollout files.
- Behavior gate summary:
  - diagnostic `chain_oracle`: `52` heart deposits, `77` battery crafts,
    `82` ore pickups;
  - seed `0` deterministic checkpoint: `21` heart deposits, `50` battery
    crafts, `51` ore pickups, but also `20` water pickups, `32` wheat pickups,
    `151` wood pickups, `8` armor crafts, `8` lantern crafts, `8` spear crafts,
    and `7` lantern plants;
  - seed `0` stochastic checkpoint: `33` heart deposits, `50` battery crafts,
    `79` ore pickups, but also `64` wheat pickups, `134` wood pickups,
    `14` armor crafts, `6` bread crafts, `12` lantern crafts, `18` spear
    crafts, and `12` lantern plants;
  - seed `1` deterministic checkpoint: `0` heart deposits, `0` battery crafts,
    `6` ore pickups, plus `55` water pickups, `55` wheat pickups, `127` wood
    pickups, and `2` tumor kills;
  - seed `1` stochastic checkpoint: `0` heart deposits, `4` battery crafts,
    `33` ore pickups, plus `130` water pickups, `165` wheat pickups, `142` wood
    pickups, `6` armor crafts, `4` bread crafts, `6` spear crafts, `13` tumor
    kills, and `6` lantern plants;
  - seed `2` deterministic checkpoint: `18` heart deposits, `32` battery
    crafts, `78` ore pickups, plus `69` water pickups, `68` wheat pickups,
    `151` wood pickups, `11` armor crafts, `15` spear crafts, `19` tumor
    kills, `1` spawner kill, and `25` armor handoffs;
  - seed `2` stochastic checkpoint: `24` heart deposits, `61` battery crafts,
    `91` ore pickups, plus `55` water pickups, `62` wheat pickups, `170` wood
    pickups, `16` armor crafts, `6` bread crafts, `1` lantern craft, `18` spear
    crafts, `12` tumor kills, `1` agent kill, `1` lantern plant, `20` armor
    handoffs, and `19` bread handoffs.
- Decision: v8 does not pass the promotion criteria. It preserves the v7 chain
  behavior for seeds `0` and `2`, but seed `1` loses heart deposits entirely,
  and the small off-chain penalties do not reliably reduce the unwanted
  water/wheat/wood, non-battery craft, combat, or handoff behaviors. In some
  rollouts the off-chain behavior is worse than v7.
- Interpretation: reward-side cleanup alone is not enough here. The small
  penalties make the objective noisier without isolating the heart chain. Keep
  v7 as the stronger representation-debug signal, and treat v8 as evidence
  that off-chain cleanup needs either curriculum structure or affordance-level
  control rather than another small coefficient tweak.
- Next iteration if we continue past v8:
  - do not increase the v8 penalty magnitudes;
  - instead test a v9 curriculum/affordance condition that makes the intended
    chain easier to discover without training on additional negative rewards;
  - keep the v7 chain-compass observation and v6 chain rewards unchanged while
    varying only the curriculum/affordance intervention;
  - require v9 to match v7 deposits while reducing off-chain resource and
    non-battery craft counts before rerunning representation analyses.

### Reward-Shaping Correction Before V9

The v8 result is consistent with a known reward-shaping failure mode: adding
independent negative terms can make exploration brittle, alter the effective
task, and create local optima or avoidance behavior. Before continuing, we
should treat the v8 penalties as a diagnostic mistake, not as a template to
scale up.

Primary references read before planning v9:

- Ng, Harada, and Russell, "Policy Invariance Under Reward Transformations:
  Theory and Application to Reward Shaping"
  (`https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf`):
  potential-based shaping is the safe default because it adds progress
  information as a potential difference rather than independent rewards and
  penalties.
- Devlin and Kudenko, "Dynamic Potential-Based Reward Shaping"
  (`https://www.ifaamas.org/Proceedings/aamas2012/papers/2C_3.pdf`):
  the potential-based view extends to multi-agent settings and dynamic
  potentials only when the invariance assumptions are handled explicitly.
- Toro Icarte et al., "Reward Machines: Exploiting Reward Function Structure
  in Reinforcement Learning" (`https://arxiv.org/abs/2010.03950`):
  sequential tasks should expose reward structure to the learner instead of
  hiding a long chain behind black-box scalar events.
- Andrychowicz et al., "Hindsight Experience Replay"
  (`https://arxiv.org/abs/1707.01495`) and Nair et al., "Overcoming
  Exploration in Reinforcement Learning with Demonstrations"
  (`https://arxiv.org/abs/1709.10089`): sparse long-horizon tasks often need
  relabeling, demonstrations, or curriculum to solve exploration, not more
  manual reward terms.
- Trott et al., "Keeping Your Distance: Solving Sparse Reward Tasks Using
  Self-Balancing Shaped Rewards" (`https://arxiv.org/abs/1911.01417`):
  naive dense distance shaping can trap learning in local optima; successful
  shaping should decay toward the original sparse objective or otherwise avoid
  becoming the task itself.

Practical rules for v9:

- No new negative rewards for off-chain behavior during PPO training.
- Keep the terminal/success reward definition simple and positive:
  ore -> battery -> heart deposit.
- If dense shaping is needed, express it as potential-based progress:
  `F(s, s') = gamma * Phi(s') - Phi(s)`, where `Phi` is monotonic over the
  chain state:
  - empty and closer to ore target;
  - ore held and closer to converter;
  - battery held and closer to home assembler;
  - heart deposited.
- Prefer curriculum over punishment:
  - start with a cleaned or constrained chain-only village;
  - gradually reintroduce distractor resources and alternate recipes only after
    the chain behavior is stable;
  - keep the evaluation world unchanged so promotion still measures robustness.
- Prefer affordance guidance over penalty guidance:
  - expose the current chain target, required inventory stage, and valid chain
    action as observation features;
  - optionally mask or route only impossible/irrelevant off-chain affordances in
    the curriculum phase, but do not attach negative reward to successful
    distractor actions.
- Keep `chain_oracle` as a diagnostic teacher:
  - compare learned checkpoint rollouts to oracle event counts;
  - optionally add a short behavioral-cloning or warm-start phase from oracle
    trajectories if pure PPO still struggles.
- Promotion criteria for the next run:
  - match or exceed v7 deposits on all three seeds;
  - reduce off-chain behavior because the curriculum/environment makes it less
    available, not because the agent is being punished for exploring it;
  - raw return should not degrade relative to v7;
  - the final evaluation must run in the full Tribal Village with distractors
    restored.

### Reward-Debug Continuation: `event_v9_potential_chain_compass_breadcrumbs`

Source log:

- Keep `v3_experiments/REWARD_SHAPING_NOTES.md` as the compact reference list
  and reward-design work log for later paper updates.
- Treat v8 as a diagnostic failure of explicit negative reward coefficients,
  not as a reason to add stronger penalties.

V9 design:

- Preserve the successful v7 intervention:
  - canonical Tribal Village observations plus the five chain-compass planes;
  - action masking;
  - positive chain events only: `resource_ore`, `craft_battery`, and
    `deposit_heart`;
  - positive mask-valid oracle-action breadcrumbs for moving toward or using
    the current chain target.
- Remove explicit negative reward coefficients:
  - no noop penalty;
  - no invalid-action penalty;
  - no off-chain successful-event penalty.
- Replace the v6/v7 clipped distance-progress bonuses with potential-based
  chain progress:

```text
F(s, s') = gamma * Phi(s') - Phi(s)
```

- Define `Phi` over the inventory-conditioned heart-chain state:
  - empty agent -> nearest mine;
  - ore carrier -> nearest converter;
  - battery carrier -> home assembler.
- Keep `Phi` modest so the terminal chain events remain dominant. A heart
  deposit should still be strongly positive even when the potential resets to
  the next ore-search cycle.

Important interpretation:

- The potential-difference term may be signed because that is how
  potential-based shaping works. It is not a new independent punishment for
  water, wheat, wood, armor, bread, combat, invalid actions, or no-op actions.
- If v9 reduces off-chain behavior, the explanation should be that the chain
  path became easier to discover and maintain, not that exploration was
  punished.

V9 Stage-1 ramp:

- Reward design: `event_v9_potential_chain_compass_breadcrumbs`.
- Shared fraction: `0.0`.
- Seeds: `0,1,2`.
- Budget: `1,000,000` agent steps per seed.
- Use `--use-action-mask`.
- Chain-compass observation is auto-enabled by the reward design and should
  appear in result JSON as `chain_compass_observation: true` and
  `obs_shape: [26, 11, 11]`.
- Result root:
  `/workspace/tribal_event_mask_runs/stage1_v9_potential_chain_compass_1m`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage1_v9_potential_chain_compass_<sha>`.

V9 promotion criteria:

- Match v7's core signal: every seed should produce nonzero heart deposits in
  deterministic or stochastic checkpoint rollouts.
- Preferably match or exceed the v7 deterministic deposit range of `9-35`
  hearts over the 3-episode behavior gate.
- Reduce off-chain water/wheat/wood collection and non-battery crafts relative
  to v7 without using off-chain penalties.
- Raw return should not degrade relative to v7.
- If v9 does not match v7 deposits, do not promote it; return to v7 for
  representation debugging and pursue curriculum, affordance masking, or
  short oracle warm starts as the next intervention.

V9 Stage-1 outcome:

- Implementation commit: `d4708a0b8a`.
- Remote worktrees were staged on both sandboxes at commit `d4708a0b8a` from the
  `tasha` remote. `origin` on those worktrees still points at
  `Metta-AI/metta`, so future fetches for this branch should use `tasha`.
- Tiny native Tribal Village smokes passed on both sandboxes:
  - sandbox 1 output:
    `/workspace/tribal_event_mask_runs/smoke_v9_sandbox1.json`;
  - sandbox 2 output:
    `/workspace/tribal_event_mask_runs/smoke_v9_sandbox2.json`;
  - both recorded `reward_design:
    event_v9_potential_chain_compass_breadcrumbs`;
  - both recorded `chain_compass_observation: true`.
- Stage-1 ran at `shared_frac=0.0` for `1,000,008` agent steps with
  `--use-action-mask`, seeds `0,1,2`, offline W&B, and the v9 potential-chain
  reward.
- Result root:
  `/workspace/tribal_event_mask_runs/stage1_v9_potential_chain_compass_1m`.
- Result validation:
  - `validate_canonical_reward_geometry_results.py --allow-smoke` passed for
    all three result JSONs.
- Final eval summary:
  - seed `0`: raw `-10.3439`, shaping `-75.4350`, total `-85.7789`,
    effrank/agent `0.2062`, JS action diversity `0.2997`, role probe `0.2709`;
  - seed `1`: raw `1.3358`, shaping `137.9324`, total `139.2683`,
    effrank/agent `0.2765`, JS action diversity `0.4978`, role probe `0.3436`;
  - seed `2`: raw `-9.4900`, shaping `-55.8264`, total `-65.3164`,
    effrank/agent `0.1906`, JS action diversity `0.3677`, role probe `0.3919`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage1_v9_potential_chain_compass_d4708a0b8a`.
- Behavior-output validation passed:
  - sandbox 1: `9` rollout files;
  - sandbox 2: `2` rollout files.
- Behavior gate summary:
  - diagnostic `chain_oracle`: `25` heart deposits, `49` crafts, `53`
    resources, raw reward `-5.767`;
  - `random`: `0` deposits, `111` resources, `5` crafts, raw reward `-35.467`,
    mostly invalid;
  - `use_sweep`: `0` deposits, `31` resources, `14` crafts, raw reward
    `-30.400`, mostly invalid;
  - seed `0` deterministic checkpoint: `0` deposits, `335` resources,
    `8` crafts, `67` combat events, raw reward `-41.883`;
  - seed `0` stochastic checkpoint: `2` deposits, `408` resources, `37` crafts,
    `43` combat events, raw reward `-33.763`;
  - seed `1` deterministic checkpoint: `48` deposits, `347` resources,
    `145` crafts, `17` combat events, raw reward `7.967`;
  - seed `1` stochastic checkpoint: `49` deposits, `326` resources,
    `129` crafts, `0` combat events, raw reward `6.167`;
  - seed `2` deterministic checkpoint: `37` deposits, `182` resources,
    `82` crafts, `2` combat events, raw reward `-2.367`;
  - seed `2` stochastic checkpoint: `30` deposits, `506` resources,
    `125` crafts, `22` combat events, raw reward `-13.900`.
- Decision: v9 is a real training signal but not a clean promotion condition.
  It restores nonzero heart deposits for every seed under at least one rollout
  mode and strongly improves seeds `1` and `2`, but seed `0` remains weak and
  the policies still do large amounts of off-chain resource collection,
  non-battery crafting, handoffs, and combat.
- Next iteration:
  - do not add explicit negative rewards;
  - keep v9's potential-chain framing as the preferred reward-shaping baseline;
  - add a curriculum or affordance intervention that removes or delays
    distractors while training, then evaluate in the full world;
  - consider a short `chain_oracle` warm start or behavior-cloning phase before
    PPO if pure PPO remains seed-fragile;
  - do not rerun representation analyses until the behavior gate is cleaner and
    stable across all seeds.

### Reward-Debug Continuation: `event_v10_chain_affordance_compass_breadcrumbs`

Purpose:

- Continue the v9 cleanup path without adding negative rewards.
- Test whether the remaining off-chain behavior is mostly an affordance problem:
  PPO can currently press successful `use`, `put`, `attack`, and `plant`
  actions on many distractor objects while learning the heart chain.

Design:

- Keep the v9 reward design:
  - positive chain events: `resource_ore`, `craft_battery`, `deposit_heart`;
  - bounded potential-based chain progress;
  - positive chain-oracle action breadcrumbs;
  - no noop, invalid-action, or off-chain successful-event penalties.
- Keep the v7/v9 chain-compass observation.
- Add a chain-affordance action mask during this debug/curriculum condition:
  - movement actions remain available when valid;
  - `use` is available only for the current chain target when the target is
    adjacent;
  - off-chain `use`, `put`, `attack`, `plant`, and `swap` affordances are not
    available to the policy in v10;
  - if navigation is missing, the mask falls back to the normal environment
    action mask.
- Checkpoint behavior rollouts use the effective
  `chain_affordance_action_mask`. New checkpoints save the effective flag in
  their config; legacy v10 checkpoints also infer it from `reward_design`. This
  keeps evaluation consistent with the v10 curriculum interface.

Interpretation:

- V10 is a cleanup/curriculum condition, not a final unconstrained Tribal
  Village paper condition.
- If v10 succeeds, it tells us that chain behavior can be made clean by
  removing distractor affordances. The next research step would be to anneal or
  relax the mask, or to transfer from the v10 checkpoint into the full action
  surface.
- If v10 fails, the next intervention should be a short `chain_oracle` warm
  start or a chain-only start-state curriculum.

V10 Stage-1 ramp:

- Reward design: `event_v10_chain_affordance_compass_breadcrumbs`.
- Shared fraction: `0.0`.
- Seeds: `0,1,2`.
- Budget: `1,000,000` agent steps per seed.
- Use `--use-action-mask`; the reward design auto-enables
  `chain_affordance_action_mask`.
- Result root:
  `/workspace/tribal_event_mask_runs/stage1_v10_chain_affordance_compass_1m`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage1_v10_chain_affordance_compass_<sha>`.

V10 promotion criteria:

- Every seed should have deterministic or stochastic heart deposits, ideally
  near the v9 seed-1/seed-2 range.
- Seed 0 should improve materially over v9's `0` deterministic and `2`
  stochastic deposits.
- Off-chain resource pickups, non-battery crafts, handoffs, and combat should
  drop sharply relative to v9 because the off-chain affordances are unavailable.
- Raw reward should not collapse relative to v9.
- Do not promote v10 directly to paper representation analysis unless we also
  decide that constrained affordance training is the experimental condition we
  want to claim.

V10 first ramp outcome, commit `6b2442b2b4`:

- Result validation passed for seeds `0`, `1`, and `2`; behavior-output
  validation passed for 11 rollout files split across `relh-sandbox-1` and
  `relh-sandbox-2`.
- Training eval shaped returns improved, but raw env return stayed negative:

| Seed | Eval raw return | Eval shaped/individual return | `D_act_JS` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `-1.483` | `173.838` | `0.474` | `0.245` | `0.345` |
| `1` | `-1.819` | `144.540` | `0.463` | `0.215` | `0.333` |
| `2` | `-7.994` | `114.768` | `0.496` | `0.171` | `0.227` |

- Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Resources | Crafts | Deposits | Combat | Invalid frac. | Joint actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `8.000` | `74.7` | `91` | `81` | `52` | `0` | `0.2225` | `197.7` |
| `random` | `-28.800` | `42.7` | `115` | `6` | `0` | `0` | `0.7237` | `240.0` |
| `seed0_det` | `-30.467` | `112.0` | `106` | `5` | `0` | `0` | `0.0051` | `24.3` |
| `seed0_stoch` | `0.600` | `242.0` | `226` | `96` | `42` | `0` | `0.0277` | `235.0` |
| `seed1_det` | `-28.800` | `20.3` | `36` | `24` | `0` | `1` | `0.0035` | `8.0` |
| `seed1_stoch` | `-41.697` | `112.3` | `261` | `16` | `0` | `5` | `0.0198` | `186.3` |
| `seed2_det` | `-28.800` | `11.7` | `26` | `9` | `0` | `0` | `0.0021` | `6.0` |
| `seed2_stoch` | `-28.800` | `23.7` | `52` | `10` | `0` | `0` | `0.0007` | `44.7` |

- Interpretation: the first v10 ramp is not a clean promotion candidate. It
  produced one strong learned chain mode (`seed0_stoch`) and proved the oracle
  can complete the chain under the gate, but learned deposits were not robust
  across seeds/modes and deterministic policies still collapsed.
- Cleanup bug found after the gate: the per-agent chain-affordance mask fell
  back to the full native action mask when no chain move/use was available.
  That reopened noop and sometimes off-chain `put`, `attack`, and `plant`
  actions. The strict follow-up should make that fallback noop-only instead of
  full-mask before rerunning Stage 1.
- Next action: rerun the v10 Stage-1 ramp after the strict fallback fix, using a
  fresh result root that includes the new commit SHA.

V10 strict-mask rerun outcome, training commit `f9b78f1937`, loader fix commit
`76fbf21da6`:

- Result validation passed for seeds `0`, `1`, and `2`.
- Behavior-output validation passed for 11 corrected masked rollout files split
  across `relh-sandbox-1` and `relh-sandbox-2`.
- Stage-1 result root:
  `/workspace/tribal_event_mask_runs/stage1_v10_chain_affordance_compass_1m`.
- Corrected behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage1_v10_chain_affordance_compass_masked_76fbf21da6`.
- The strict fallback fix worked, but it exposed a second provenance bug:
  checkpoints saved the raw CLI `chain_affordance_action_mask=false` even though
  v10 auto-enabled the effective mask. The behavior loader now infers the mask
  for legacy v10 checkpoints and new checkpoints save the effective mask and
  chain-compass flags.

Training eval:

| Seed | Eval raw return | Eval shaped/individual return | `D_act_JS` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `-3.351` | `79.441` | `0.418` | `0.251` | `0.347` |
| `1` | `-11.275` | `108.642` | `0.502` | `0.285` | `0.361` |
| `2` | `-0.548` | `147.130` | `0.472` | `0.246` | `0.366` |

Corrected masked behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Resources | Battery crafts | Heart deposits | Invalid frac. | Joint actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `-3.433` | `59.0` | `76` | `62` | `39` | `0.1949` | `179.3` |
| `random` | `-45.103` | `33.0` | `87` | `0` | `0` | `0.7386` | `223.7` |
| `seed0_det` | `12.533` | `75.0` | `88` | `81` | `56` | `0.1890` | `221.0` |
| `seed0_stoch` | `-5.190` | `67.0` | `89` | `67` | `45` | `0.1311` | `201.3` |
| `seed1_det` | `6.300` | `71.3` | `87` | `74` | `53` | `0.1910` | `203.0` |
| `seed1_stoch` | `-3.600` | `81.3` | `102` | `82` | `60` | `0.1536` | `240.0` |
| `seed2_det` | `5.933` | `74.0` | `90` | `82` | `50` | `0.1569` | `209.7` |
| `seed2_stoch` | `-25.957` | `49.0` | `74` | `46` | `27` | `0.0454` | `176.3` |

Decision:

- V10 is the first clean behavior-valid ramp by the constrained-mask gate: all
  three seeds now execute the ore -> battery -> heart chain, deterministic
  rollouts for all seeds beat no-op and random on raw reward, and checkpoint
  attempts contain no `put`, `attack`, `plant`, or `swap`.
- This is not yet a final paper condition because it is a constrained
  affordance curriculum. It is strong evidence that the chain behavior is
  learnable when distractor affordances are removed.
- Next research step: run a short transfer/annealing ramp from these v10
  checkpoints into a less constrained action surface before launching a
  representation sweep. A reasonable sequence is strict mask -> allow native
  `use` after stable chain progress -> allow `put`/`attack`/`plant` only after
  behavior remains deposit-positive.

V10 Stage-2 strict-mask ramp plan:

- Purpose: test whether the constrained v10 behavior remains stable at a longer
  budget before changing the action surface.
- Reward design: `event_v10_chain_affordance_compass_breadcrumbs`.
- Shared fraction: `0.0`.
- Seeds: `0,1,2`.
- Budget: `10,000,008` agent steps per seed.
- Run root:
  `/workspace/tribal_event_mask_runs/stage2_v10_chain_affordance_compass_10m`.
- Run naming:
  `stage2_v10_chain_affordance_compass_alpha0_seed<seed>_10m_<sha>`.
- Sandbox split:
  - `relh-sandbox-1`: seeds `0` and `1`, one process per GPU slot.
  - `relh-sandbox-2`: seed `2`.
- Keep the strict v10 action mask and chain-compass observation. Do not start a
  representation sweep from this result unless the behavior gate remains
  deposit-positive and cleaner than v9/v10 Stage 1.

Stage-2 promotion criteria:

- Every deterministic checkpoint has heart deposits in the 3-episode behavior
  gate.
- Deterministic raw reward remains better than no-op and random.
- Checkpoint action attempts stay limited to `move`, `noop`, and chain-target
  `use` under the strict mask.
- Invalid-action fraction does not increase materially from the Stage-1 strict
  masked gate.
- If Stage 2 passes, the next run should be an annealing/transfer experiment,
  not a final paper sweep: start from the Stage-2 checkpoint and relax only one
  affordance family at a time.

V10 Stage-2 strict-mask outcome, commit `129dbac38b`:

- Training jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `134`: seeds `0` and `1`.
  - `relh-sandbox-2` job `133`: seed `2`.
- Result JSON validation passed for all three seeds.
- Behavior-output validation passed for 11 rollout files split across
  `relh-sandbox-1` and `relh-sandbox-2`.
- Result root:
  `/workspace/tribal_event_mask_runs/stage2_v10_chain_affordance_compass_10m`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage2_v10_chain_affordance_compass_129dbac38b`.

Training eval:

| Seed | Eval raw return | Eval shaped/individual return | `D_act_JS` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `-10.949` | `-67.389` | `0.440` | `0.307` | `0.323` |
| `1` | `-10.627` | `-50.329` | `0.462` | `0.446` | `0.345` |
| `2` | `-4.907` | `33.563` | `0.391` | `0.362` | `0.283` |

Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Resources | Battery crafts | Heart deposits | Invalid frac. | Joint actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `-19.210` | `54.0` | `71` | `56` | `35` | `0.1167` | `167.0` |
| `random` | `-28.733` | `12.7` | `37` | `0` | `0` | `0.7514` | `240.0` |
| `seed0_det` | `-44.093` | `4.0` | `12` | `0` | `0` | `0.0749` | `75.3` |
| `seed0_stoch` | `-35.227` | `11.0` | `30` | `3` | `0` | `0.0941` | `141.7` |
| `seed1_det` | `-8.183` | `59.3` | `79` | `62` | `37` | `0.1703` | `176.0` |
| `seed1_stoch` | `2.433` | `57.0` | `72` | `61` | `38` | `0.1069` | `240.0` |
| `seed2_det` | `-11.567` | `34.3` | `47` | `44` | `12` | `0.0586` | `134.3` |
| `seed2_stoch` | `-25.063` | `32.0` | `52` | `32` | `12` | `0.0181` | `173.0` |

Decision:

- The strict action mask continued to work mechanically: checkpoint rollouts had
  zero `put`, `attack`, `plant`, and `swap` attempts.
- Longer strict-v10 training did not improve the result. It weakened the clean
  1M behavior, especially seed `0`, which lost battery/heart-chain completion.
- Do not promote the 10M strict-v10 checkpoints to a representation sweep.
- The next ramp should be a budget/transfer diagnostic, not another longer
  strict-mask run:
  - use the 1M strict-v10 checkpoints as the known-good behavior reference;
  - run a short budget ladder or early-stopping comparison before 10M;
  - then test annealing from the clean checkpoint into a less constrained action
    surface one affordance family at a time.

V10 budget-ladder diagnostic plan:

- Purpose: test whether strict-v10 behavior degrades because 10M overtrains past
  a shorter stable policy.
- Reward design: `event_v10_chain_affordance_compass_breadcrumbs`.
- Shared fraction: `0.0`.
- Seeds: `0,1,2`.
- Budgets: `2,000,004` and `4,000,008` agent steps per seed.
- Run root:
  `/workspace/tribal_event_mask_runs/stage3_v10_budget_ladder`.
- Run naming:
  `stage3_v10_budget_ladder_alpha0_seed<seed>_<budget>_<sha>`.
- Sandbox split:
  - `relh-sandbox-1`: seeds `0` and `1`, each running the 2M then 4M budget on
    a dedicated GPU.
  - `relh-sandbox-2`: seed `2`, running the 2M then 4M budget.
- Behavior gates:
  - `/workspace/tribal_event_mask_runs/behavior_gate_stage3_v10_budget_ladder_2m_<sha>`.
  - `/workspace/tribal_event_mask_runs/behavior_gate_stage3_v10_budget_ladder_4m_<sha>`.

Budget-ladder decision rule:

- If 2M or 4M retains all-seed deterministic deposits while 10M does not, treat
  that budget as the candidate transfer source.
- If both budgets degrade, prefer the 1M strict checkpoint as the transfer
  source and do not spend more on strict-mask-only PPO.
- If a budget passes behavior but has lower representation diversity than 1M,
  prioritize behavior stability first; representation sweeps remain blocked
  until an annealed action-surface condition passes.

V10 Stage-3 budget-ladder outcome, commit `69e6627267`:

- Training jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `135`: seeds `0` and `1`, each running `2M` then `4M`.
  - `relh-sandbox-2` job `134`: seed `2`, running `2M` then `4M`.
- Result JSON validation passed for all six training outputs.
- Behavior-output validation passed for both replay gates:
  - 2M gate: 11 rollout files across baselines plus checkpoint seeds `0,1,2`.
  - 4M gate: 11 rollout files across baselines plus checkpoint seeds `0,1,2`.
- Training root:
  `/workspace/tribal_event_mask_runs/stage3_v10_budget_ladder`.
- Behavior gate roots:
  - `/workspace/tribal_event_mask_runs/behavior_gate_stage3_v10_budget_ladder_2m_69e6627267`.
  - `/workspace/tribal_event_mask_runs/behavior_gate_stage3_v10_budget_ladder_4m_69e6627267`.

2M training eval:

| Seed | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `-7.348` | `132.379` | `139.727` | `0.499` | `0.295` | `0.333` |
| `1` | `-0.962` | `122.953` | `123.915` | `0.525` | `0.431` | `0.343` |
| `2` | `-0.538` | `154.728` | `155.266` | `0.416` | `0.240` | `0.327` |

2M behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Mean unique actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `-4.903` | `55.3` | `72` | `56` | `36` | `943` | `16.7` |
| `no_op` | `-30.467` | `0.0` | `0` | `0` | `0` | `0` | `1.0` |
| `random` | `-39.730` | `51.7` | `5` | `0` | `0` | `6129` | `56.0` |
| `move_sweep` | `-32.557` | `0.0` | `0` | `0` | `0` | `578` | `8.0` |
| `use_sweep` | `-28.800` | `12.3` | `0` | `0` | `0` | `8603` | `8.0` |
| `seed0_det` | `3.967` | `78.7` | `91` | `84` | `57` | `2227` | `16.3` |
| `seed1_det` | `-7.200` | `72.3` | `88` | `75` | `51` | `1182` | `16.7` |
| `seed2_det` | `-17.600` | `78.0` | `92` | `83` | `58` | `1518` | `17.0` |
| `seed0_stoch` | `-2.200` | `68.7` | `82` | `72` | `49` | `1408` | `16.3` |
| `seed1_stoch` | `-18.167` | `74.0` | `99` | `70` | `51` | `1242` | `16.7` |
| `seed2_stoch` | `-12.667` | `72.0` | `86` | `71` | `53` | `858` | `16.7` |

4M training eval:

| Seed | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `-0.183` | `155.654` | `155.837` | `0.465` | `0.208` | `0.356` |
| `1` | `-1.587` | `133.019` | `134.607` | `0.471` | `0.279` | `0.337` |
| `2` | `-11.210` | `-20.535` | `-9.325` | `0.463` | `0.251` | `0.296` |

4M behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Mean unique actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `-14.367` | `68.0` | `83` | `70` | `49` | `2238` | `16.7` |
| `no_op` | `-35.467` | `0.0` | `0` | `0` | `0` | `0` | `1.0` |
| `random` | `-28.733` | `41.3` | `3` | `0` | `0` | `6255` | `56.0` |
| `move_sweep` | `-34.997` | `0.0` | `0` | `0` | `0` | `584` | `8.0` |
| `use_sweep` | `-28.767` | `10.0` | `0` | `0` | `0` | `8610` | `8.0` |
| `seed0_det` | `5.533` | `69.0` | `78` | `74` | `51` | `1153` | `16.3` |
| `seed1_det` | `14.933` | `75.7` | `86` | `82` | `57` | `2649` | `17.0` |
| `seed2_det` | `-26.797` | `61.7` | `82` | `58` | `39` | `1691` | `15.7` |
| `seed0_stoch` | `-64.833` | `72.7` | `113` | `57` | `45` | `844` | `17.0` |
| `seed1_stoch` | `9.600` | `78.7` | `90` | `84` | `59` | `1516` | `17.0` |
| `seed2_stoch` | `-3.200` | `74.0` | `86` | `79` | `55` | `969` | `16.3` |

Decision:

- The strict-v10 action mask still learns the ore -> battery -> heart-deposit
  chain at 2M and 4M. All deterministic checkpoint seeds deposited hearts at
  both budgets, unlike the 10M seed `0` failure.
- The 2M budget is the cleaner candidate transfer source:
  - all deterministic seeds exceeded the chain-oracle deposit count from the
    same 2M gate;
  - all deterministic seeds had stronger task-event counts than the oracle and
    trivial baselines;
  - seed `2` remained clean at 2M but had negative training eval return and a
    weaker deterministic behavior gate at 4M.
- Do not interpret the role-probe or representation metrics as a paper result
  yet. Role-probe accuracy remains near chance, and these strict-mask policies
  are behavior-validation checkpoints, not final reward-mixing evidence.
- Next ramp:
  - use the 2M strict-v10 checkpoints as the transfer source;
  - run one affordance-relaxation experiment at a time from those checkpoints;
  - behavior-gate each relaxed condition before starting any
    `shared_frac in {0.0, 0.8, 1.0}` representation sweep.

V10 Stage-4 transfer/annealing plan:

- Purpose: test whether the behavior-valid 2M strict-v10 policies can survive
  the first action-surface relaxation.
- Source checkpoints:
  - seed `0`:
    `/workspace/tribal_event_mask_runs/stage3_v10_budget_ladder/stage3_v10_budget_ladder_alpha0_seed0_2m_69e6627267/final_model.pt`.
  - seed `1`:
    `/workspace/tribal_event_mask_runs/stage3_v10_budget_ladder/stage3_v10_budget_ladder_alpha0_seed1_2m_69e6627267/final_model.pt`.
  - seed `2`:
    `/workspace/tribal_event_mask_runs/stage3_v10_budget_ladder/stage3_v10_budget_ladder_alpha0_seed2_2m_69e6627267/final_model.pt`.
- First relaxation: keep `event_v10_chain_affordance_compass_breadcrumbs`,
  chain-compass observations, and the environment action mask, but disable the
  strict chain-affordance mask with
  `--disable-chain-affordance-action-mask`. This allows all environment-valid
  actions while continuing from a policy that already performs the
  ore -> battery -> heart chain.
- Run shape:
  - `--init-checkpoint-path <2m strict checkpoint>`.
  - `--shared-frac 0.0`.
  - `--total-agent-steps 1,000,008`.
  - `--eval-trials 10`, `--eval-steps 1000`.
  - `--wandb-mode offline`.
- Run root:
  `/workspace/tribal_event_mask_runs/stage4_v10_transfer_relaxed_envmask`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage4_v10_transfer_relaxed_envmask_<sha>`.
- Pass criteria:
  - deterministic checkpoint rollouts for all seeds keep nonzero heart deposits;
  - at least two of three deterministic seeds stay within 25% of their 2M
    strict-v10 deposit counts;
  - relaxed rollouts do not become dominated by no-op/random invalid-action
    behavior;
  - new off-chain actions may appear, but they must not erase the learned
    ore -> battery -> heart chain.
- Decision rule:
  - If the relaxed-env-mask transfer passes, test the next relaxation as a
    narrower follow-up instead of jumping directly to reward mixing.
  - If it fails, return to the 2M strict source and either shorten the transfer
    budget, lower the learning rate, or relax only one verb family with a custom
    mask instead of the full environment-valid action set.

V10 Stage-4 relaxed-env-mask transfer outcome, commit `7e5adc4a50`:

- Training jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `136`: seeds `0` and `1`.
  - `relh-sandbox-2` job `135`: seed `2`.
- Result JSON validation passed for all three transfer outputs.
- Behavior-output validation passed for 11 rollout files split across
  `relh-sandbox-1` and `relh-sandbox-2`.
- Training root:
  `/workspace/tribal_event_mask_runs/stage4_v10_transfer_relaxed_envmask`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage4_v10_transfer_relaxed_envmask_7e5adc4a50`.

Training eval:

| Seed | Init checkpoint | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | EffRank/n | Role probe | Strict mask |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `stage3...seed0_2m_69e6627267` | `-3.129` | `90.139` | `93.268` | `0.432` | `0.299` | `0.359` | `false` |
| `1` | `stage3...seed1_2m_69e6627267` | `-3.200` | `118.559` | `121.759` | `0.491` | `0.261` | `0.287` | `false` |
| `2` | `stage3...seed2_2m_69e6627267` | `-5.917` | `-18.254` | `-12.338` | `0.423` | `0.296` | `0.362` | `false` |

Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Off-chain attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `-9.830` | `58.7` | `75` | `57` | `41` | `630` | `0` |
| `no_op` | `-30.467` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-28.800` | `29.7` | `0` | `0` | `0` | `6306` | `4944` |
| `move_sweep` | `-34.877` | `0.0` | `0` | `0` | `0` | `552` | `0` |
| `use_sweep` | `-37.133` | `4.0` | `0` | `0` | `0` | `8628` | `0` |
| `seed0_det` | `19.567` | `83.0` | `95` | `92` | `62` | `717` | `0` |
| `seed1_det` | `5.700` | `62.0` | `75` | `65` | `44` | `1442` | `0` |
| `seed2_det` | `-11.233` | `37.7` | `45` | `44` | `18` | `1639` | `0` |
| `seed0_stoch` | `-0.500` | `58.7` | `73` | `62` | `38` | `789` | `0` |
| `seed1_stoch` | `-16.400` | `84.7` | `96` | `87` | `63` | `959` | `0` |
| `seed2_stoch` | `-1.633` | `50.3` | `63` | `54` | `32` | `680` | `0` |

Decision:

- The relaxed-env-mask transfer passes the minimum written behavior gate: all
  deterministic seeds still make heart deposits, and seeds `0` and `1` remain
  within 25% of their 2M strict-v10 deposit counts.
- The result is not clean enough to promote to reward mixing:
  - seed `2` drops from `58` strict 2M deterministic deposits to `18`;
  - seed `2` also has negative eval shaped/individual return;
  - none of the checkpoint rollouts attempted `put`, `attack`, `plant`, or
    `swap`, despite those actions being available under the relaxed environment
    mask.
- Interpretation: the first relaxation did not produce richer off-chain
  behavior. The main risk is policy drift under continued PPO updates after the
  action surface expands.
- Next diagnostic before more training: evaluate the original 2M strict
  checkpoints under the relaxed environment mask with no additional training.
  If those replay gates remain strong, then the problem is transfer update
  drift; if they fail immediately, then the strict mask is still doing essential
  inference-time control.

V10 Stage-4 source-checkpoint relaxed-mask replay diagnostic, commit
`f22c6ed1a1`:

- No training was run. This gate evaluated the Stage-3 2M source checkpoints
  with `--disable-checkpoint-chain-affordance-action-mask`.
- Behavior-output validation passed for 11 rollout files.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage4_source2m_relaxed_eval_f22c6ed1a1`.

Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Put attempts | Other off-chain attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `-12.530` | `55.7` | `70` | `57` | `39` | `1048` | `0` | `0` |
| `random` | `-43.383` | `30.3` | `0` | `0` | `0` | `5913` | `1130` | `3467` |
| `seed0_det` | `-18.133` | `323.0` | `76` | `50` | `20` | `379` | `517` | `3` |
| `seed1_det` | `-22.767` | `91.7` | `58` | `12` | `5` | `255` | `7` | `1` |
| `seed2_det` | `-23.443` | `155.0` | `57` | `26` | `19` | `1497` | `102` | `7` |
| `seed0_stoch` | `-10.500` | `374.0` | `78` | `58` | `41` | `637` | `522` | `15` |
| `seed1_stoch` | `10.100` | `192.0` | `115` | `92` | `58` | `633` | `144` | `7` |
| `seed2_stoch` | `-10.600` | `180.7` | `88` | `79` | `42` | `1224` | `129` | `4` |

Decision:

- Removing the strict mask at inference time materially disrupts deterministic
  behavior before any transfer update. The source checkpoint policies start
  selecting many `put` actions, and deterministic deposits fall sharply for all
  seeds.
- The 1M relaxed-env-mask transfer therefore did repair some inference-time
  mask-removal damage for seeds `0` and `1`, but did not recover seed `2`.
- Next ramp should not use the full environment-valid action surface. It should
  allow exactly one off-chain verb family while preserving the strict
  move/current-chain-use curriculum.

V10 Stage-5 one-family relaxation plan:

- First one-family condition: `put` only.
- Purpose: test whether handoff-style actions can be exposed without the broad
  relaxed-mask instability seen in Stage 4.
- Command change from Stage 4:
  - remove `--disable-chain-affordance-action-mask`;
  - add `--chain-affordance-extra-verbs put`;
  - keep `--init-checkpoint-path <2m strict checkpoint>`.
- Initial budget: `1,000,008` agent steps for seeds `0,1,2`.
- Run root:
  `/workspace/tribal_event_mask_runs/stage5_v10_transfer_put_only`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage5_v10_transfer_put_only_<sha>`.
- Pass criteria:
  - deterministic deposits remain nonzero for all seeds;
  - deterministic seed `2` improves over the Stage-4 relaxed-env-mask transfer
    deposit count of `18`;
  - `put` attempts may occur, but task-chain progress must remain near the 2M
    strict source behavior rather than collapsing into handoff spam.

V10 Stage-5 put-only transfer outcome, commit `f575fdd22c`:

- Training jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `137`: seeds `0` and `1`.
  - `relh-sandbox-2` job `136`: seed `2`.
- Result JSON validation passed for all three transfer outputs.
- Behavior-output validation passed for 11 rollout files.
- Training root:
  `/workspace/tribal_event_mask_runs/stage5_v10_transfer_put_only`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage5_v10_transfer_put_only_f575fdd22c`.

Training eval:

| Seed | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | EffRank/n | Role probe | Extra verbs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `-7.876` | `-1.570` | `6.306` | `0.351` | `0.150` | `0.370` | `put` |
| `1` | `-1.988` | `149.559` | `151.547` | `0.450` | `0.304` | `0.327` | `put` |
| `2` | `-6.580` | `26.519` | `33.099` | `0.421` | `0.307` | `0.350` | `put` |

Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Put attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `-23.900` | `53.7` | `75` | `44` | `37` | `2008` | `0` |
| `seed0_det` | `-29.300` | `10.7` | `19` | `7` | `6` | `1220` | `0` |
| `seed1_det` | `-18.267` | `57.0` | `70` | `57` | `39` | `537` | `0` |
| `seed2_det` | `-49.157` | `70.3` | `100` | `68` | `41` | `1011` | `0` |
| `seed0_stoch` | `17.867` | `82.0` | `92` | `86` | `62` | `1188` | `0` |
| `seed1_stoch` | `-26.467` | `73.7` | `96` | `68` | `53` | `768` | `0` |
| `seed2_stoch` | `-51.790` | `56.0` | `75` | `45` | `42` | `622` | `0` |

Decision:

- Put-only relaxation is not a promotion candidate. It improves deterministic
  seed `2` over Stage 4, but deterministic seed `0` collapses from `57` strict
  2M deposits to `6`.
- The agents do not actually use `put` in the behavior gate, so this does not
  create meaningful handoff behavior. It is another training-dynamics
  perturbation around the chain policy.
- Current stable behavior condition remains strict-v10 at 2M. Do not continue
  action-surface relaxation as the next mainline route.

V10 Stage-6 strict-mask reward-mixing pilot plan:

- Purpose: test the paper's reward-mixing claim only on the behavior-validated
  strict-v10 condition.
- Baseline: Stage-3 strict-v10 `shared_frac=0.0`, `2M` budget.
- New arms:
  - `shared_frac=0.8`, seeds `0,1,2`, `2M` agent steps.
  - `shared_frac=1.0`, seeds `0,1,2`, `2M` agent steps.
- Keep:
  - `event_v10_chain_affordance_compass_breadcrumbs`;
  - `--use-action-mask`;
  - strict chain-affordance mask;
  - chain-compass observations;
  - no checkpoint warm start.
- Run root:
  `/workspace/tribal_event_mask_runs/stage6_v10_strict_reward_mixing_pilot`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage6_v10_strict_reward_mixing_pilot_<sha>`.
- Pass criteria before interpreting representation:
  - every deterministic checkpoint arm has nonzero heart deposits;
  - each shared-reward arm has at least two of three deterministic seeds above
    the chain-oracle deposit count from the same gate;
  - no arm collapses to no-op/random behavior.
- Interpretation rule:
  - If behavior fails for shared-reward arms, paper claims about reward mixing
    remain blocked.
  - If behavior passes, compare `D_act_JS`, EffRank/n, role probe, action
    diversity, and replay behavior against the Stage-3 `shared_frac=0.0`
    baseline as a pilot, not yet as final paper evidence.

V10 Stage-6 strict-mask reward-mixing pilot outcome, commit `450cfb15a3`:

- Training jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `138`: `shared_frac=0.8` seeds `0,1` and
    `shared_frac=1.0` seeds `0,1`.
  - `relh-sandbox-2` job `137`: `shared_frac=0.8` seed `2` and
    `shared_frac=1.0` seed `2`.
- Behavior-gate jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `139`: eight deterministic/stochastic checkpoint
    rollouts for seeds `0,1`.
  - `relh-sandbox-2` job `138`: five baselines plus four
    deterministic/stochastic checkpoint rollouts for seed `2`.
- Result JSON validation passed for all six training outputs.
- Behavior-output validation passed for all Stage-6 rollout files.
- Training root:
  `/workspace/tribal_event_mask_runs/stage6_v10_strict_reward_mixing_pilot`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage6_v10_strict_reward_mixing_pilot_450cfb15a3`.
- Remote launch scripts:
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage6_v10_strict_reward_mixing_pilot_sandbox1_450cfb15a3.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage6_v10_strict_reward_mixing_pilot_sandbox2_450cfb15a3.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage6_behavior_gate_sandbox1_450cfb15a3.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage6_behavior_gate_sandbox2_450cfb15a3.sh`

Training eval:

| Shared frac | Seed | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | `D_act_KL` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.8` | `0` | `-2.999` | `129.604` | `132.603` | `0.524` | `13.840` | `0.275` | `0.324` |
| `0.8` | `1` | `-3.028` | `151.705` | `154.733` | `0.488` | `12.572` | `0.254` | `0.323` |
| `0.8` | `2` | `-11.597` | `-56.380` | `-44.783` | `0.502` | `15.447` | `0.222` | `0.383` |
| `1.0` | `0` | `-11.476` | `-31.779` | `-20.302` | `0.576` | `20.614` | `0.112` | `0.351` |
| `1.0` | `1` | `-11.433` | `-33.136` | `-21.704` | `0.559` | `20.418` | `0.143` | `0.352` |
| `1.0` | `2` | `-12.299` | `-39.858` | `-27.559` | `0.498` | `17.019` | `0.125` | `0.331` |

Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-32.133` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-35.467` | `39.3` | `0` | `0` | `0` | `6307` | `111` |
| `move_sweep` | `-28.800` | `0.0` | `0` | `0` | `0` | `626` | `0` |
| `use_sweep` | `-33.800` | `18.7` | `0` | `0` | `0` | `8584` | `56` |
| `chain_oracle` | `-15.867` | `69.7` | `86` | `74` | `46` | `1482` | `209` |
| `alpha0.8_seed0_det` | `13.800` | `79.0` | `90` | `86` | `60` | `2098` | `237` |
| `alpha0.8_seed0_stoch` | `7.533` | `75.3` | `84` | `77` | `59` | `1908` | `226` |
| `alpha0.8_seed1_det` | `-15.257` | `38.7` | `53` | `38` | `22` | `1043` | `116` |
| `alpha0.8_seed1_stoch` | `-23.497` | `45.3` | `65` | `40` | `27` | `623` | `136` |
| `alpha0.8_seed2_det` | `-41.030` | `7.7` | `23` | `0` | `0` | `1577` | `23` |
| `alpha0.8_seed2_stoch` | `-27.467` | `9.3` | `24` | `2` | `0` | `2672` | `28` |
| `alpha1.0_seed0_det` | `-34.303` | `3.7` | `8` | `1` | `0` | `2981` | `11` |
| `alpha1.0_seed0_stoch` | `-28.733` | `1.0` | `2` | `0` | `0` | `3887` | `3` |
| `alpha1.0_seed1_det` | `-28.667` | `1.3` | `4` | `0` | `0` | `3702` | `4` |
| `alpha1.0_seed1_stoch` | `-35.367` | `1.3` | `3` | `0` | `0` | `3504` | `4` |
| `alpha1.0_seed2_det` | `-34.197` | `3.7` | `8` | `0` | `0` | `912` | `11` |
| `alpha1.0_seed2_stoch` | `-39.867` | `6.0` | `18` | `0` | `0` | `2245` | `18` |

Decision:

- Stage 6 does not pass the reward-mixing behavior gate.
- `shared_frac=0.8` is seed-fragile:
  - seed `0` beats the chain-oracle baseline on deposits and raw reward;
  - seed `1` has nonzero deposits but negative raw reward;
  - seed `2` fails the deposit criterion entirely.
- `shared_frac=1.0` collapses under both deterministic and stochastic gates:
  every seed has zero heart deposits and very low task-event counts.
- Representation/probe metrics from Stage 6 are not paper evidence because the
  behavior precondition failed.
- The immediate next diagnostic is a warm-start reward-mixing test from the
  behavior-valid strict-v10 alpha0 2M checkpoints. This distinguishes whether
  high sharing prevents chain-behavior acquisition from scratch, or destroys
  the learned behavior after acquisition.

V10 Stage-7 warm-start reward-mixing diagnostic plan:

- Purpose: separate acquisition failure from reward-sharing-induced behavior
  destruction.
- Source checkpoints:
  - seed `0`:
    `/workspace/tribal_event_mask_runs/stage3_v10_budget_ladder/stage3_v10_budget_ladder_alpha0_seed0_2m_69e6627267/final_model.pt`
  - seed `1`:
    `/workspace/tribal_event_mask_runs/stage3_v10_budget_ladder/stage3_v10_budget_ladder_alpha0_seed1_2m_69e6627267/final_model.pt`
  - seed `2`:
    `/workspace/tribal_event_mask_runs/stage3_v10_budget_ladder/stage3_v10_budget_ladder_alpha0_seed2_2m_69e6627267/final_model.pt`
- Arms:
  - `shared_frac=0.8`, seeds `0,1,2`, `500,004` fine-tuning agent steps.
  - `shared_frac=1.0`, seeds `0,1,2`, `500,004` fine-tuning agent steps.
- Keep:
  - `event_v10_chain_affordance_compass_breadcrumbs`;
  - `--use-action-mask`;
  - strict chain-affordance mask;
  - chain-compass observations;
  - per-seed `--init-checkpoint-path` from the Stage-3 strict 2M source.
- Run root:
  `/workspace/tribal_event_mask_runs/stage7_v10_warmstart_reward_mixing`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage7_v10_warmstart_reward_mixing_<sha>`.
- Pass criteria:
  - every deterministic checkpoint arm has nonzero heart deposits;
  - at least two of three seeds per shared fraction stay within 25% of their
    Stage-3 strict 2M deterministic heart-deposit count;
  - raw behavior remains above no-op/random and does not collapse to the
    Stage-6 alpha1 pattern.
- Interpretation rule:
  - If warm-started alpha0.8 preserves behavior but alpha1.0 collapses, the
    likely claim is a behavioral threshold for reward sharing, not a clean
    monotone representation result.
  - If both preserve behavior, run a longer behavior-gated reward-mixing sweep.
  - If both collapse, high reward sharing is incompatible with the current
    reconstructed reward/task setup and the MAPPO reward-mixing paper claim
    remains unsupported.

V10 Stage-7 warm-start reward-mixing diagnostic outcome, commit `ef8dc7b8b`:

- Smoke jobs verified checkpoint loading and trainer flags on both sandboxes:
  - `relh-sandbox-1` job `140`: seed `0`, `shared_frac=0.8` smoke.
  - `relh-sandbox-2` job `139`: seed `2`, `shared_frac=1.0` smoke.
- Training jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `141`: seeds `0,1`, `shared_frac in {0.8,1.0}`.
  - `relh-sandbox-2` job `140`: seed `2`, `shared_frac in {0.8,1.0}`.
- Behavior-gate jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `142`: eight deterministic/stochastic checkpoint
    rollouts for seeds `0,1`.
  - `relh-sandbox-2` job `141`: five baselines plus four
    deterministic/stochastic checkpoint rollouts for seed `2`.
- Result JSON validation passed for all six training outputs.
- Behavior-output validation passed for all Stage-7 rollout files.
- Training root:
  `/workspace/tribal_event_mask_runs/stage7_v10_warmstart_reward_mixing`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage7_v10_warmstart_reward_mixing_ef8dc7b8b`.
- Remote launch scripts:
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage7_v10_warmstart_reward_mixing_sandbox1_ef8dc7b8b.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage7_v10_warmstart_reward_mixing_sandbox2_ef8dc7b8b.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage7_behavior_gate_sandbox1_ef8dc7b8b.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage7_behavior_gate_sandbox2_ef8dc7b8b.sh`

Training eval:

| Shared frac | Seed | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | `D_act_KL` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.8` | `0` | `-8.462` | `-3.660` | `4.802` | `0.396` | `7.349` | `0.186` | `0.356` |
| `0.8` | `1` | `1.386` | `204.829` | `203.443` | `0.500` | `11.890` | `0.283` | `0.319` |
| `0.8` | `2` | `-9.976` | `1.249` | `11.226` | `0.374` | `8.522` | `0.246` | `0.381` |
| `1.0` | `0` | `-12.496` | `-37.334` | `-24.838` | `0.366` | `11.726` | `0.164` | `0.349` |
| `1.0` | `1` | `-9.577` | `-32.929` | `-23.352` | `0.530` | `17.484` | `0.133` | `0.336` |
| `1.0` | `2` | `-11.446` | `-49.376` | `-37.930` | `0.431` | `13.171` | `0.169` | `0.335` |

Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-30.467` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-35.467` | `26.3` | `0` | `0` | `0` | `6330` | `77` |
| `move_sweep` | `-38.397` | `0.0` | `0` | `0` | `0` | `492` | `0` |
| `use_sweep` | `-35.467` | `1.7` | `0` | `0` | `0` | `8635` | `5` |
| `chain_oracle` | `3.000` | `70.7` | `84` | `80` | `48` | `1958` | `212` |
| `alpha0.8_seed0_det` | `-38.140` | `19.0` | `34` | `21` | `0` | `908` | `57` |
| `alpha0.8_seed0_stoch` | `-17.923` | `61.7` | `77` | `64` | `43` | `741` | `185` |
| `alpha0.8_seed1_det` | `-0.967` | `66.0` | `81` | `68` | `46` | `1713` | `198` |
| `alpha0.8_seed1_stoch` | `12.467` | `80.0` | `92` | `87` | `60` | `1474` | `240` |
| `alpha0.8_seed2_det` | `-30.453` | `29.3` | `39` | `28` | `19` | `492` | `88` |
| `alpha0.8_seed2_stoch` | `-9.783` | `69.7` | `86` | `74` | `47` | `727` | `209` |
| `alpha1.0_seed0_det` | `-31.933` | `2.3` | `6` | `0` | `0` | `870` | `7` |
| `alpha1.0_seed0_stoch` | `-34.120` | `0.3` | `1` | `0` | `0` | `1017` | `1` |
| `alpha1.0_seed1_det` | `-35.467` | `0.0` | `0` | `0` | `0` | `1793` | `0` |
| `alpha1.0_seed1_stoch` | `-37.753` | `1.0` | `2` | `0` | `0` | `3289` | `3` |
| `alpha1.0_seed2_det` | `-34.370` | `0.7` | `2` | `0` | `0` | `1613` | `2` |
| `alpha1.0_seed2_stoch` | `-33.433` | `2.7` | `6` | `0` | `0` | `604` | `8` |

Decision:

- Stage 7 does not pass the deterministic warm-start reward-mixing gate.
- `shared_frac=1.0` is now a stronger negative result: it destroys the learned
  chain even when initialized from behavior-valid alpha0 checkpoints.
- `shared_frac=0.8` preserves a stochastic version of the chain across all
  seeds, with stochastic heart deposits `43,60,47`. It does not preserve robust
  deterministic behavior:
  - seed `0` deterministic deposits drop from Stage-3 strict 2M `57` to `0`;
  - seed `1` deterministic deposits stay near threshold at `46`;
  - seed `2` deterministic deposits drop from Stage-3 strict 2M `58` to `19`.
- The next question is whether the deterministic fragility is caused by too much
  fine-tuning under mixed reward, too large a learning rate, or an unavoidable
  alpha0.8 policy-distribution effect.
- Do not update the paper from Stage 7. It is mechanistic evidence that
  alpha1.0 is destructive and alpha0.8 may be recoverable only under
  stochastic execution.

V10 Stage-8 short/low-lr alpha0.8 stabilization diagnostic plan:

- Purpose: test whether alpha0.8 can preserve deterministic chain behavior if
  the reward-mixing transfer is gentler.
- Source checkpoints: the same Stage-3 strict-v10 alpha0 2M per-seed
  checkpoints used in Stage 7.
- Arms:
  - `shared_frac=0.8`, seeds `0,1,2`, `100,008` fine-tuning agent steps,
    learning rate `1e-4`.
  - `shared_frac=0.8`, seeds `0,1,2`, `250,008` fine-tuning agent steps,
    learning rate `1e-4`.
- Keep:
  - `event_v10_chain_affordance_compass_breadcrumbs`;
  - `--use-action-mask`;
  - strict chain-affordance mask;
  - chain-compass observations;
  - per-seed `--init-checkpoint-path` from the Stage-3 strict 2M source.
- Run root:
  `/workspace/tribal_event_mask_runs/stage8_v10_alpha0p8_low_lr_stabilization`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage8_v10_alpha0p8_low_lr_stabilization_<sha>`.
- Pass criteria:
  - every deterministic checkpoint arm has nonzero heart deposits;
  - at least two of three seeds per budget stay within 25% of their Stage-3
    strict 2M deterministic heart-deposit count;
  - stochastic rollouts remain nonzero and do not lose the chain;
  - raw behavior stays above no-op/random and near the chain-oracle baseline.
- Interpretation rule:
  - If 100k or 250k preserves deterministic behavior, use that budget as the
    alpha0.8 transfer candidate and run one follow-up representation/probe
    check against the alpha0 source.
  - If both fail deterministically but pass stochastically, the paper can only
    discuss alpha0.8 behavior under stochastic execution, not deterministic
    competence.
  - If both fail, high reward sharing remains unsupported for the reconstructed
    Tribal Village setup.

V10 Stage-8 short/low-lr alpha0.8 stabilization diagnostic outcome, commit
`c17b7dfeb`:

- Training jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `143`: seeds `0,1`, budgets `100k` and `250k`.
  - `relh-sandbox-2` job `142`: seed `2`, budgets `100k` and `250k`.
- Behavior-gate jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `144`: eight deterministic/stochastic checkpoint
    rollouts for seeds `0,1`.
  - `relh-sandbox-2` job `143`: five baselines plus four
    deterministic/stochastic checkpoint rollouts for seed `2`.
- Result JSON validation passed for all six training outputs.
- Behavior-output validation passed for all Stage-8 rollout files.
- Training root:
  `/workspace/tribal_event_mask_runs/stage8_v10_alpha0p8_low_lr_stabilization`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage8_v10_alpha0p8_low_lr_stabilization_c17b7dfeb`.
- Remote launch scripts:
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage8_v10_alpha0p8_low_lr_stabilization_sandbox1_c17b7dfeb.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage8_v10_alpha0p8_low_lr_stabilization_sandbox2_c17b7dfeb.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage8_behavior_gate_sandbox1_c17b7dfeb.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage8_behavior_gate_sandbox2_c17b7dfeb.sh`

Training eval:

| Budget | Seed | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | `D_act_KL` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `100k` | `0` | `-0.226` | `183.194` | `183.421` | `0.507` | `13.054` | `0.283` | `0.334` |
| `100k` | `1` | `-2.542` | `131.114` | `133.656` | `0.480` | `12.459` | `0.328` | `0.303` |
| `100k` | `2` | `-4.451` | `81.685` | `86.137` | `0.456` | `11.781` | `0.208` | `0.363` |
| `250k` | `0` | `-7.709` | `221.178` | `228.887` | `0.517` | `12.339` | `0.274` | `0.323` |
| `250k` | `1` | `-3.927` | `95.184` | `99.110` | `0.547` | `15.605` | `0.340` | `0.322` |
| `250k` | `2` | `-4.415` | `124.258` | `128.673` | `0.511` | `15.091` | `0.234` | `0.327` |

Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-30.467` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-40.210` | `37.3` | `3` | `0` | `0` | `5505` | `105` |
| `move_sweep` | `-45.073` | `0.0` | `0` | `0` | `0` | `603` | `0` |
| `use_sweep` | `-37.133` | `5.0` | `0` | `0` | `0` | `8625` | `15` |
| `chain_oracle` | `-17.633` | `55.0` | `69` | `52` | `40` | `1480` | `165` |
| `100k_seed0_det` | `-7.597` | `58.0` | `76` | `60` | `37` | `769` | `174` |
| `100k_seed0_stoch` | `10.533` | `70.3` | `84` | `72` | `52` | `1220` | `211` |
| `100k_seed1_det` | `-31.303` | `65.0` | `84` | `60` | `47` | `1588` | `195` |
| `100k_seed1_stoch` | `-11.570` | `65.0` | `78` | `69` | `44` | `1503` | `195` |
| `100k_seed2_det` | `-17.997` | `46.3` | `62` | `52` | `22` | `769` | `139` |
| `100k_seed2_stoch` | `-15.133` | `81.3` | `98` | `84` | `59` | `1092` | `244` |
| `250k_seed0_det` | `8.367` | `71.3` | `83` | `69` | `58` | `1869` | `214` |
| `250k_seed0_stoch` | `4.633` | `78.0` | `89` | `83` | `60` | `2475` | `234` |
| `250k_seed1_det` | `-14.963` | `57.0` | `73` | `61` | `36` | `1809` | `171` |
| `250k_seed1_stoch` | `12.200` | `71.7` | `84` | `77` | `53` | `1254` | `215` |
| `250k_seed2_det` | `-31.220` | `35.0` | `60` | `25` | `19` | `685` | `105` |
| `250k_seed2_stoch` | `-31.833` | `70.3` | `93` | `67` | `48` | `768` | `211` |

Decision:

- Stage 8 does not pass the deterministic alpha0.8 stabilization gate.
- Lower learning rate and shorter budgets improved alpha0.8 relative to Stage 7
  but did not find a single all-seed deterministic pass condition:
  - `100k` deterministic deposits are `37,47,22`;
  - `250k` deterministic deposits are `58,36,19`.
- Stochastic behavior remains much stronger, with all six stochastic rollouts
  depositing hearts and four of six above the Stage-8 chain-oracle deposit count
  of `40`.
- The persistent failure mode is deterministic seed fragility, especially seed
  `2`. This suggests alpha0.8 preserves a useful stochastic policy distribution
  but does not reliably preserve the greedy action sequence.
- Do not update the paper from Stage 8. Treat it as evidence that alpha0.8 is a
  partial/stochastic behavior condition, not a canonical deterministic one.

V10 Stage-9 lower-alpha threshold diagnostic plan:

- Purpose: find whether reward sharing below `0.8` can preserve deterministic
  behavior while still perturbing representation geometry.
- Source checkpoints: the same Stage-3 strict-v10 alpha0 2M per-seed
  checkpoints.
- Arms:
  - `shared_frac=0.4`, seeds `0,1,2`, `250,008` fine-tuning agent steps,
    learning rate `1e-4`.
  - `shared_frac=0.6`, seeds `0,1,2`, `250,008` fine-tuning agent steps,
    learning rate `1e-4`.
- Keep:
  - `event_v10_chain_affordance_compass_breadcrumbs`;
  - `--use-action-mask`;
  - strict chain-affordance mask;
  - chain-compass observations;
  - per-seed `--init-checkpoint-path` from the Stage-3 strict 2M source.
- Run root:
  `/workspace/tribal_event_mask_runs/stage9_v10_lower_alpha_threshold`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage9_v10_lower_alpha_threshold_<sha>`.
- Pass criteria:
  - every deterministic checkpoint arm has nonzero heart deposits;
  - at least two of three seeds per alpha stay within 25% of their Stage-3
    strict 2M deterministic heart-deposit count;
  - stochastic rollouts remain nonzero;
  - raw behavior stays above no-op/random and near the chain-oracle baseline.
- Interpretation rule:
  - If alpha0.4 or alpha0.6 passes, that alpha becomes the candidate
    behavior-valid mixed-reward condition for a representation/probe follow-up.
  - If both fail deterministically but pass stochastically, deterministic
    MAPPO reward-mixing claims remain unsupported in this reconstruction.
  - If both fail both modes, stop reward-mixing ramps and write the negative
    finding clearly: only individual reward reliably supports the reconstructed
    Tribal Village behavior.

V10 Stage-9 lower-alpha threshold diagnostic outcome, commit `632fac02e`:

- Training jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `145`: seeds `0,1`, alpha `0.4` and `0.6`.
  - `relh-sandbox-2` job `144`: seed `2`, alpha `0.4` and `0.6`.
- Behavior-gate jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `146`: eight deterministic/stochastic checkpoint
    rollouts for seeds `0,1`.
  - `relh-sandbox-2` job `145`: five baselines plus four
    deterministic/stochastic checkpoint rollouts for seed `2`.
- Result JSON validation passed for all six training outputs.
- Behavior-output validation passed for all Stage-9 rollout files.
- Training root:
  `/workspace/tribal_event_mask_runs/stage9_v10_lower_alpha_threshold`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage9_v10_lower_alpha_threshold_632fac02e`.
- Remote launch scripts:
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage9_v10_lower_alpha_threshold_sandbox1_632fac02e.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage9_v10_lower_alpha_threshold_sandbox2_632fac02e.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage9_behavior_gate_sandbox1_632fac02e.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage9_behavior_gate_sandbox2_632fac02e.sh`

Training eval:

| Alpha | Seed | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | `D_act_KL` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.4` | `0` | `-6.743` | `173.729` | `180.472` | `0.523` | `14.086` | `0.253` | `0.353` |
| `0.4` | `1` | `-7.251` | `138.303` | `145.553` | `0.513` | `13.379` | `0.316` | `0.338` |
| `0.4` | `2` | `-3.102` | `123.841` | `126.942` | `0.539` | `15.627` | `0.262` | `0.350` |
| `0.6` | `0` | `-5.591` | `93.988` | `99.579` | `0.484` | `12.468` | `0.228` | `0.297` |
| `0.6` | `1` | `-3.725` | `174.042` | `177.767` | `0.548` | `15.502` | `0.324` | `0.351` |
| `0.6` | `2` | `-3.920` | `137.545` | `141.465` | `0.517` | `14.328` | `0.244` | `0.330` |

Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-32.133` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-28.800` | `34.7` | `1` | `0` | `0` | `5575` | `104` |
| `move_sweep` | `-37.133` | `0.0` | `0` | `0` | `0` | `1173` | `0` |
| `use_sweep` | `-30.467` | `8.3` | `0` | `0` | `0` | `8611` | `25` |
| `chain_oracle` | `-8.933` | `50.3` | `65` | `48` | `38` | `1648` | `151` |
| `alpha0.4_seed0_det` | `-11.367` | `62.7` | `79` | `63` | `44` | `2022` | `188` |
| `alpha0.4_seed0_stoch` | `-34.517` | `55.0` | `82` | `50` | `32` | `906` | `165` |
| `alpha0.4_seed1_det` | `-7.430` | `53.7` | `69` | `55` | `36` | `1350` | `161` |
| `alpha0.4_seed1_stoch` | `-26.497` | `63.7` | `81` | `62` | `44` | `989` | `191` |
| `alpha0.4_seed2_det` | `-22.070` | `49.3` | `65` | `50` | `30` | `1163` | `148` |
| `alpha0.4_seed2_stoch` | `3.930` | `67.0` | `81` | `71` | `46` | `1928` | `201` |
| `alpha0.6_seed0_det` | `1.933` | `68.0` | `80` | `74` | `50` | `2184` | `204` |
| `alpha0.6_seed0_stoch` | `5.900` | `79.3` | `95` | `82` | `59` | `1768` | `238` |
| `alpha0.6_seed1_det` | `-22.203` | `22.7` | `35` | `28` | `3` | `2203` | `68` |
| `alpha0.6_seed1_stoch` | `-23.067` | `77.7` | `94` | `81` | `53` | `1520` | `233` |
| `alpha0.6_seed2_det` | `-25.833` | `59.7` | `75` | `58` | `45` | `2849` | `179` |
| `alpha0.6_seed2_stoch` | `1.767` | `78.0` | `91` | `82` | `57` | `1709` | `234` |

Decision:

- Stage 9 found the first plausible mixed-reward threshold candidate, but did
  not yet find a canonical behavior-valid condition.
- `shared_frac=0.6` is the best candidate:
  - deterministic heart deposits were `50,3,45`;
  - stochastic heart deposits were `59,53,57`;
  - two of three deterministic seeds stayed within 25% of their Stage-3 strict
    2M source deposits, but seed `1` nearly collapsed under greedy execution.
- `shared_frac=0.4` is weaker:
  - deterministic heart deposits were `44,36,30`;
  - stochastic heart deposits were `32,44,46`;
  - all deterministic seeds remained nonzero, but only seed `0` clearly met
    the Stage-3 25% preservation threshold.
- The main mechanism uncertainty is budget/overtraining versus inherent
  deterministic fragility. Alpha0.6 has strong stochastic behavior for every
  seed, which suggests the useful chain policy may still exist in the
  distribution even when the greedy policy fails.
- Do not update the paper from Stage 9. Treat it as a threshold-finding
  diagnostic and run one shorter-budget alpha0.6 check before promotion.

V10 Stage-10 alpha0.6 short-budget diagnostic plan:

- Purpose: test whether the Stage-9 alpha0.6 seed1 deterministic collapse is a
  `250k` fine-tuning budget/overtraining issue.
- Source checkpoints: the same Stage-3 strict-v10 alpha0 2M per-seed
  checkpoints.
- Arm:
  - `shared_frac=0.6`, seeds `0,1,2`, `100,008` fine-tuning agent steps,
    learning rate `1e-4`.
- Keep:
  - `event_v10_chain_affordance_compass_breadcrumbs`;
  - `--use-action-mask`;
  - strict chain-affordance mask;
  - chain-compass observations;
  - per-seed `--init-checkpoint-path` from the Stage-3 strict 2M source.
- Run root:
  `/workspace/tribal_event_mask_runs/stage10_v10_alpha0p6_short_budget`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage10_v10_alpha0p6_short_budget_<sha>`.
- Pass criteria:
  - every deterministic checkpoint has nonzero heart deposits;
  - all three deterministic seeds stay within 25% of their Stage-3 strict 2M
    deterministic heart-deposit count;
  - stochastic rollouts remain nonzero and do not lose the chain;
  - raw behavior stays above no-op/random and near the chain-oracle baseline.
- Interpretation rule:
  - If alpha0.6 at `100k` passes deterministically across all seeds, promote it
    as the first behavior-valid mixed-reward candidate and run a
    representation/probe follow-up against the alpha0 source.
  - If only seed1 improves while seed0/2 degrade, alpha0.6 is budget-fragile
    and should not be promoted without an even tighter schedule or annealing.
  - If seed1 remains weak, reward sharing is still not reproducibly
    deterministic-valid in the reconstructed setup, and the next discriminating
    experiment should be an annealed transfer rather than another fixed-alpha
    short run.

V10 Stage-10 alpha0.6 short-budget diagnostic outcome, commit `a16d0d6d2`:

- Training jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `147`: seeds `0,1`.
  - `relh-sandbox-2` job `146`: seed `2`.
- Behavior-gate jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `148`: deterministic/stochastic checkpoint rollouts
    for seeds `0,1`.
  - `relh-sandbox-2` job `147`: five baselines plus deterministic/stochastic
    checkpoint rollouts for seed `2`.
- Result JSON and checkpoint files were present and nonempty for all three
  training runs.
- Behavior-output validation passed for all Stage-10 rollout files.
- Training root:
  `/workspace/tribal_event_mask_runs/stage10_v10_alpha0p6_short_budget`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage10_v10_alpha0p6_short_budget_a16d0d6d2`.
- Remote launch scripts:
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage10_v10_alpha0p6_short_budget_sandbox1_a16d0d6d2.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage10_v10_alpha0p6_short_budget_sandbox2_a16d0d6d2.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage10_behavior_gate_sandbox1_a16d0d6d2.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage10_behavior_gate_sandbox2_a16d0d6d2.sh`
- Offline W&B run IDs:
  - seed `0`: `dujrobc5`;
  - seed `1`: `wyexhp9n`;
  - seed `2`: `or44hm32`.

Training eval:

| Seed | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | `D_act_KL` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `-6.437` | `133.935` | `140.372` | `0.530` | `14.976` | `0.237` | `0.356` |
| `1` | `-7.789` | `91.684` | `99.473` | `0.522` | `14.531` | `0.281` | `0.338` |
| `2` | `-5.606` | `92.734` | `98.340` | `0.520` | `14.902` | `0.224` | `0.315` |

Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-28.800` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-28.800` | `29.0` | `0` | `0` | `0` | `6359` | `80` |
| `move_sweep` | `-35.397` | `0.0` | `0` | `0` | `0` | `694` | `0` |
| `use_sweep` | `-28.800` | `19.7` | `0` | `0` | `0` | `8581` | `59` |
| `chain_oracle` | `-64.000` | `85.3` | `114` | `85` | `55` | `1211` | `256` |
| `alpha0.6_100k_seed0_det` | `16.800` | `78.7` | `90` | `86` | `59` | `2287` | `236` |
| `alpha0.6_100k_seed0_stoch` | `14.800` | `75.7` | `86` | `79` | `59` | `1247` | `227` |
| `alpha0.6_100k_seed1_det` | `-8.530` | `54.3` | `64` | `60` | `34` | `1807` | `163` |
| `alpha0.6_100k_seed1_stoch` | `-19.940` | `72.3` | `90` | `71` | `49` | `1577` | `217` |
| `alpha0.6_100k_seed2_det` | `1.177` | `59.3` | `74` | `61` | `42` | `1634` | `178` |
| `alpha0.6_100k_seed2_stoch` | `-2.067` | `81.0` | `94` | `86` | `62` | `1842` | `243` |

Decision:

- Stage 10 is a near miss, not a promotion candidate.
- Shortening alpha0.6 fine-tuning from `250k` to `100k` improved deterministic
  seed1 heart deposits from `3` to `34`, and seed0 improved from `50` to `59`.
- The strict preservation thresholds from Stage-3 source behavior remain:
  - seed0: `57 * 0.75 = 42.75`;
  - seed1: `51 * 0.75 = 38.25`;
  - seed2: `58 * 0.75 = 43.5`.
- Stage-10 deterministic deposits were `59,34,42`, so seed1 and seed2 still
  miss the all-seed pass gate.
- Stochastic deposits were `59,49,62`, again showing that useful behavior
  remains in the policy distribution even when greedy execution is fragile.
- The mechanism read is budget sensitivity plus deterministic fragility, not
  simple acquisition failure. The next discriminating experiment should anneal
  reward sharing from alpha0 to alpha0.6 instead of applying fixed alpha0.6
  immediately.
- Do not update the paper from Stage 10.

V10 Stage-11 alpha0-to-alpha0.6 annealed-transfer diagnostic plan:

- Purpose: test whether gradual reward-sharing introduction preserves
  deterministic chain behavior better than fixed alpha0.6.
- Code support:
  - add `--shared-frac-start <float>` to
    `v3_experiments/train_canonical_reward_geometry.py`;
  - keep `--shared-frac` as the final/evaluation alpha;
  - record `shared_frac_start` and `shared_frac_schedule` in result JSON and
    checkpoint metadata;
  - log the current training `shared_frac` in `train_metrics`.
- Source checkpoints: the same Stage-3 strict-v10 alpha0 2M per-seed
  checkpoints.
- Arm:
  - linear alpha schedule from `0.0` to `0.6`, seeds `0,1,2`, `250,008`
    fine-tuning agent steps, learning rate `1e-4`.
- Keep:
  - `event_v10_chain_affordance_compass_breadcrumbs`;
  - `--use-action-mask`;
  - strict chain-affordance mask;
  - chain-compass observations;
  - per-seed `--init-checkpoint-path` from the Stage-3 strict 2M source.
- Run root:
  `/workspace/tribal_event_mask_runs/stage11_v10_alpha0_to_alpha0p6_anneal`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage11_v10_alpha0_to_alpha0p6_anneal_<sha>`.
- Pass criteria:
  - every deterministic checkpoint has nonzero heart deposits;
  - all three deterministic seeds stay within 25% of their Stage-3 strict 2M
    deterministic heart-deposit count;
  - stochastic rollouts remain nonzero and do not lose the chain;
  - raw behavior stays above no-op/random and near the chain-oracle baseline.
- Interpretation rule:
  - If annealing passes where fixed alpha0.6 failed, promote annealing as the
    behavior-preserving transfer protocol and run representation/probe follow-up
    at the final alpha.
  - If annealing improves but still misses one seed, run one smaller final-alpha
    anneal, likely `0.5`, before giving up on mixed rewards.
  - If annealing fails similarly to fixed alpha0.6, fixed reward mixing is not
    behavior-valid enough for deterministic MAPPO role-learning claims in this
    reconstruction.

V10 Stage-11 alpha0-to-alpha0.6 annealed-transfer diagnostic outcome, commit
`284d19ddb`:

- Training jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `149`: seeds `0,1`.
  - `relh-sandbox-2` job `148`: seed `2`.
- Behavior-gate jobs completed successfully on 2026-06-09:
  - `relh-sandbox-1` job `150`: deterministic/stochastic checkpoint rollouts
    for seeds `0,1`.
  - `relh-sandbox-2` job `149`: five baselines plus deterministic/stochastic
    checkpoint rollouts for seed `2`.
- Result JSON and checkpoint files were present and nonempty for all three
  training runs.
- Result metadata confirmed `shared_frac=0.6`, `shared_frac_start=0.0`, and
  `shared_frac_schedule=linear` for all three seeds.
- Behavior-output validation passed for all Stage-11 rollout files.
- Training root:
  `/workspace/tribal_event_mask_runs/stage11_v10_alpha0_to_alpha0p6_anneal`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage11_v10_alpha0_to_alpha0p6_anneal_284d19ddb`.
- Remote launch scripts:
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage11_v10_alpha0_to_alpha0p6_anneal_sandbox1_284d19ddb.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage11_v10_alpha0_to_alpha0p6_anneal_sandbox2_284d19ddb.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage11_behavior_gate_sandbox1_284d19ddb.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage11_behavior_gate_sandbox2_284d19ddb.sh`
- Offline W&B run IDs:
  - seed `0`: `nt50m4xe`;
  - seed `1`: `c89gvm59`;
  - seed `2`: `8n4sgyoa`.

Training eval:

| Seed | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | `D_act_KL` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `-6.194` | `146.546` | `152.741` | `0.529` | `14.218` | `0.275` | `0.330` |
| `1` | `-0.148` | `120.430` | `120.579` | `0.503` | `13.975` | `0.232` | `0.354` |
| `2` | `-0.869` | `134.325` | `135.194` | `0.516` | `14.677` | `0.290` | `0.388` |

Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-28.800` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-37.100` | `43.7` | `1` | `0` | `0` | `6277` | `124` |
| `move_sweep` | `-33.800` | `0.0` | `0` | `0` | `0` | `543` | `0` |
| `use_sweep` | `-35.467` | `11.7` | `0` | `0` | `0` | `8605` | `35` |
| `chain_oracle` | `-12.830` | `61.0` | `81` | `61` | `41` | `1261` | `183` |
| `anneal0to0.6_seed0_det` | `-4.467` | `90.7` | `108` | `99` | `63` | `1910` | `272` |
| `anneal0to0.6_seed0_stoch` | `-33.587` | `66.0` | `81` | `69` | `43` | `1339` | `198` |
| `anneal0to0.6_seed1_det` | `-6.367` | `58.3` | `69` | `63` | `40` | `630` | `175` |
| `anneal0to0.6_seed1_stoch` | `-5.423` | `50.3` | `66` | `48` | `34` | `400` | `151` |
| `anneal0to0.6_seed2_det` | `18.967` | `81.3` | `91` | `89` | `63` | `2621` | `244` |
| `anneal0to0.6_seed2_stoch` | `-5.600` | `75.7` | `90` | `77` | `54` | `1260` | `227` |

Decision:

- Stage 11 passes the deterministic mixed-reward behavior gate.
- Deterministic heart deposits were `63,40,63`, above the Stage-3 source
  preservation thresholds:
  - seed0: threshold `42.75`, observed `63`;
  - seed1: threshold `38.25`, observed `40`;
  - seed2: threshold `43.5`, observed `63`.
- Deterministic rollouts also beat no-op/random on raw reward and had nonzero
  ore pickup, battery crafting, heart deposit, and use-success counts.
- Stochastic rollouts remained nonzero with heart deposits `43,34,54`.
- This result supports the mechanism hypothesis that abrupt fixed reward mixing
  destabilizes greedy behavior, while gradual reward-sharing introduction can
  preserve the learned chain.
- Stage 11 is the first behavior-valid mixed-reward candidate. It is still not
  paper evidence by itself: the next step is a canonical promotion sweep with
  exact seed expansion, behavior gates, and representation/probe analysis.

V10 Stage-12 canonical annealed alpha0.6 promotion plan:

- Purpose: convert the Stage-11 3-seed behavior-valid candidate into a
  paper-eligible canonical result stream.
- Required source behavior:
  - keep existing strict-v10 alpha0 2M seeds `0,1,2`;
  - add strict-v10 alpha0 2M seeds `3,4` if we want a 5-seed canonical table;
  - run the same behavior gate for any new alpha0 source checkpoint.
- Required mixed-reward condition:
  - linear alpha schedule from `0.0` to `0.6`;
  - final/evaluation `shared_frac=0.6`;
  - warm-start from the matching strict-v10 alpha0 2M seed checkpoint;
  - run seeds `0..4` after source seeds `3,4` exist.
- Required validation:
  - deterministic and stochastic behavior gates for every final checkpoint;
  - replay artifacts saved for every gate;
  - representation/probe summary over the behavior-valid checkpoints only;
  - explicit comparison against the alpha0 source representation metrics.
- Paper rule:
  - do not update Overleaf until Stage-12 artifacts are complete and the
    behavior gate passes on the canonical seed set.

V10 Stage-12 annealed alpha0.6 promotion diagnostic outcome, commit
`5738657e7`:

- Current sandbox status on 2026-06-09: `relh-sandbox-1` and
  `relh-sandbox-2` queues were empty under
  `uv run sky queue <cluster> --all-users --skip-finished`; all GPUs reported
  `0 MiB` memory and `0%` utilization. Recent queue history showed the Stage-12
  jobs succeeded, but the durable provenance anchors are the exact script paths,
  result roots, checkpoint paths, and behavior-output JSON listed below.
- Remote worktree:
  `/workspace/tribal_event_mask_7635f01b`.
- Stage-12 alpha0 source root:
  `/workspace/tribal_event_mask_runs/stage12_v10_alpha0_sources`.
- Stage-12 corrected source behavior root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage12_v10_alpha0_sources_chain_compass_5738657e7`.
- Stage-12 annealed promotion root:
  `/workspace/tribal_event_mask_runs/stage12_v10_alpha0_to_alpha0p6_anneal_promotion`.
- Stage-12 annealed promotion behavior root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage12_v10_alpha0_to_alpha0p6_anneal_promotion_5738657e7`.

Source-run scripts:

- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_v10_alpha0_source_seed3_sandbox1_5738657e7.sh`
- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_v10_alpha0_source_seed4_sandbox2_5738657e7.sh`
- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_v10_alpha0_source_seed5_sandbox1_5738657e7.sh`
- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_v10_alpha0_source_seed4_4m_sandbox2_5738657e7.sh`

Source behavior-gate scripts:

- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_source_behavior_seed3_sandbox1_5738657e7.sh`
- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_source_behavior_seed4_sandbox2_5738657e7.sh`
- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_source_behavior_seed5_sandbox1_5738657e7.sh`
- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_source_behavior_seed4_4m_sandbox2_5738657e7.sh`

Promotion scripts:

- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_anneal_promotion_sandbox1_5738657e7.sh`
- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_anneal_promotion_sandbox2_5738657e7.sh`
- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_anneal_promotion_behavior_sandbox1_5738657e7.sh`
- `/workspace/tribal_event_mask_runs/scripts/launch_stage12_anneal_promotion_behavior_sandbox2_5738657e7.sh`

Important correction:

- The first source behavior attempt omitted `--chain-compass-observation`, so
  checkpoints with `obs_shape=(26,11,11)` were evaluated against an environment
  with the smaller non-compass observation. The failed root
  `/workspace/tribal_event_mask_runs/behavior_gate_stage12_v10_alpha0_sources_5738657e7`
  is invalid and should not be used. The corrected root is the
  `_chain_compass_5738657e7` behavior gate listed above.

Stage-12 source training eval:

| Source | Steps | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | `D_act_KL` | EffRank/n | Role probe | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `seed3_alpha0_2m` | `2,000,004` | `2.470` | `161.881` | `159.411` | `0.446` | `11.304` | `0.472` | `0.344` | `nzv3ap3w` |
| `seed4_alpha0_2m` | `2,000,004` | `-3.595` | `41.403` | `44.998` | `0.488` | `13.664` | `0.277` | `0.365` | `4eym7llu` |
| `seed5_alpha0_2m` | `2,000,004` | `-1.561` | `217.636` | `219.197` | `0.441` | `11.257` | `0.262` | `0.353` | `5ij0a14b` |
| `seed4_alpha0_4m` | `4,000,008` | `-7.261` | `133.010` | `140.271` | `0.483` | `12.925` | `0.263` | `0.363` | `4vp3xz53` |

Corrected Stage-12 source behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-32.133` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-36.810` | `39.3` | `3` | `0` | `0` | `5977` | `113` |
| `move_sweep` | `-28.800` | `0.0` | `0` | `0` | `0` | `905` | `0` |
| `use_sweep` | `-30.467` | `7.0` | `0` | `0` | `0` | `8619` | `21` |
| `chain_oracle` | `-11.963` | `56.3` | `73` | `58` | `37` | `1239` | `169` |
| `seed3_alpha0_2m_det` | `-4.600` | `83.3` | `104` | `84` | `60` | `1212` | `250` |
| `seed3_alpha0_2m_stoch` | `-43.763` | `64.7` | `87` | `62` | `42` | `790` | `194` |
| `seed4_alpha0_2m_det` | `-29.067` | `37.3` | `52` | `45` | `13` | `1422` | `112` |
| `seed4_alpha0_2m_stoch` | `-36.467` | `85.0` | `108` | `84` | `59` | `1092` | `255` |
| `seed5_alpha0_2m_det` | `-18.650` | `52.0` | `74` | `41` | `36` | `621` | `156` |
| `seed5_alpha0_2m_stoch` | `-18.797` | `44.7` | `64` | `43` | `27` | `359` | `134` |
| `seed4_alpha0_4m_det` | `1.833` | `56.0` | `69` | `60` | `37` | `1493` | `168` |
| `seed4_alpha0_4m_stoch` | `-4.530` | `55.0` | `72` | `55` | `36` | `933` | `165` |

Stage-12 promoted annealed alpha0-to-alpha0.6 training eval:

| Promotion | Source | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | `D_act_KL` | EffRank/n | Role probe | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `seed3_anneal0to0.6` | `seed3_alpha0_2m` | `-1.255` | `87.752` | `89.007` | `0.516` | `14.288` | `0.352` | `0.281` | `jlv4x6np` |
| `seed5_anneal0to0.6` | `seed5_alpha0_2m` | `-1.072` | `200.297` | `201.369` | `0.482` | `11.963` | `0.298` | `0.324` | `x9wbdh6o` |
| `seed4_anneal0to0.6` | `seed4_alpha0_4m` | `-9.534` | `62.692` | `72.226` | `0.512` | `14.150` | `0.261` | `0.350` | `ddzij58q` |

Stage-12 promoted behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-28.800` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-37.843` | `51.3` | `1` | `0` | `0` | `6042` | `150` |
| `move_sweep` | `-34.317` | `0.0` | `0` | `0` | `0` | `779` | `0` |
| `use_sweep` | `-32.133` | `6.0` | `0` | `0` | `0` | `8622` | `18` |
| `chain_oracle` | `-32.757` | `75.3` | `96` | `74` | `54` | `1232` | `226` |
| `seed3_anneal0to0.6_det` | `-9.433` | `69.0` | `85` | `72` | `47` | `1838` | `207` |
| `seed3_anneal0to0.6_stoch` | `-5.673` | `60.3` | `74` | `68` | `39` | `1092` | `181` |
| `seed5_anneal0to0.6_det` | `-1.600` | `48.3` | `56` | `50` | `36` | `1308` | `145` |
| `seed5_anneal0to0.6_stoch` | `-4.790` | `52.0` | `65` | `50` | `35` | `595` | `156` |
| `seed4_anneal0to0.6_det_from_4m` | `-7.333` | `46.0` | `58` | `47` | `31` | `1874` | `138` |
| `seed4_anneal0to0.6_stoch_from_4m` | `6.167` | `63.7` | `77` | `64` | `46` | `808` | `191` |

Stage-12 decision:

- Stage 12 strengthens the evidence that alpha0-to-alpha0.6 annealing preserves
  meaningful chain behavior after source acquisition: promoted deterministic
  heart deposits for additional seeds were `47`, `36`, and `31`, all nonzero
  and all above the corresponding 75% source-preservation thresholds when using
  the source checkpoints that were promoted.
- It is not a canonical paper stream. Two confounds remain:
  - contiguous seed `4` failed the deterministic source gate at the uniform `2M`
    source budget (`13` deterministic heart deposits), even though its
    stochastic rollout was strong (`59` heart deposits);
  - the promoted seed4 checkpoint came from a `4M` source rescue, which changes
    the source-acquisition budget and makes it incomparable with the uniform
    `2M` seeds.
- The uniform-2M replacement seed set `{0,1,2,3,5}` looks promising, but using
  seed5 after observing seed4 failure would be selection-biased unless we
  pre-register the rule and rerun under one SHA.
- The contiguous set `{0,1,2,3,4}` also looks promising only if all seeds use a
  common `4M` source budget, or if seed4 is explicitly reported as a source
  acquisition failure rather than silently rescued.
- Do not update Overleaf from Stage 12. Treat it as a promotion diagnostic that
  justifies a Stage-13 canonical rerun.

V10 Stage-13 pre-registered canonical rerun plan:

- Purpose: remove the Stage-12 source-selection and source-budget confounds.
- SHA rule: run all Stage-13 source, promotion, behavior, replay, and
  representation artifacts under one committed SHA after this PLAN update.
- Primary source protocol:
  - train strict-v10 alpha0 source checkpoints for seeds `0,1,2,3,4,5`;
  - use the same `2,000,004` agent-step source budget for every seed;
  - do not replace or drop a seed after seeing its behavior result;
  - run deterministic and stochastic source behavior gates for all six seeds.
- Primary mixed-reward protocol:
  - run linear alpha0-to-alpha0.6 annealed promotion for every Stage-13 source
    checkpoint, including any source-weak seed;
  - use `250,008` promotion steps and `learning_rate=1e-4`;
  - evaluate with deterministic and stochastic behavior gates.
- Reporting rule:
  - report source acquisition success across all six seeds;
  - report reward-mixing preservation conditionally on all seeds and separately
    on source-valid seeds;
  - if seed4 or any other seed is source-weak at `2M`, do not hide it. Treat
    the failure as part of the behavior-first evidence.
- Optional rescue protocol:
  - only after the primary `2M` protocol is complete, run a separate all-seed
    `4M` source-budget rescue arm if we need to know whether the source
    acquisition failure is budget-driven;
  - never mix `2M` and `4M` source checkpoints in the same canonical table.
- Promotion gate:
  - deterministic raw reward beats no-op/random;
  - ore pickup, battery crafting, heart deposits, and successful `use` actions
    are nonzero;
  - promoted deterministic heart deposits are at least `75%` of that seed's
    source deterministic heart deposits when the source deterministic behavior
    is nonzero.
- Paper rule:
  - if the primary Stage-13 seed set passes source acquisition and mixed-reward
    preservation, run representation/probe analysis and then update Overleaf;
  - if source acquisition is seed-fragile, the paper should describe the
    reconstructed setup as behavior-fragile rather than claim canonical MAPPO
    role learning.

V10 Stage-13 pre-registered canonical rerun outcome, commit `2c1c32d93`:

- Stage-13 source, promotion, and behavior-gate jobs completed successfully on
  2026-06-09.
- Remote worktree:
  `/workspace/tribal_event_mask_7635f01b`.
- Stage-13 source root:
  `/workspace/tribal_event_mask_runs/stage13_v10_alpha0_sources_2m`.
- Stage-13 source behavior root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage13_v10_alpha0_sources_2m_2c1c32d935`.
- Stage-13 promotion root:
  `/workspace/tribal_event_mask_runs/stage13_v10_alpha0_to_alpha0p6_anneal_promotion`.
- Stage-13 promotion behavior root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage13_v10_alpha0_to_alpha0p6_anneal_promotion_2c1c32d935`.
- Source training jobs:
  - `relh-sandbox-1` job `158`: seeds `0,1,2`.
  - `relh-sandbox-2` job `157`: seeds `3,4,5`.
- Source behavior jobs:
  - `relh-sandbox-1` job `159`: seeds `0,1,2`.
  - `relh-sandbox-2` job `158`: baselines plus seeds `3,4,5`.
- Promotion training jobs:
  - `relh-sandbox-1` job `160`: seeds `0,1,2`.
  - `relh-sandbox-2` job `159`: seeds `3,4,5`.
- Promotion behavior jobs:
  - `relh-sandbox-1` job `161`: seeds `0,1,2`.
  - `relh-sandbox-2` job `160`: baselines plus seeds `3,4,5`.
- Remote launch scripts:
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage13_v10_alpha0_sources_sandbox1_2c1c32d935.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage13_v10_alpha0_sources_sandbox2_2c1c32d935.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage13_source_behavior_sandbox1_2c1c32d935.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage13_source_behavior_sandbox2_2c1c32d935.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage13_anneal_promotion_sandbox1_2c1c32d935.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage13_anneal_promotion_sandbox2_2c1c32d935.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage13_anneal_promotion_behavior_sandbox1_2c1c32d935.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage13_anneal_promotion_behavior_sandbox2_2c1c32d935.sh`
- Validation:
  - source behavior validation passed for `6` checkpoint rollout files on
    sandbox 1 and `11` rollout files on sandbox 2;
  - promoted behavior validation passed for `6` checkpoint rollout files on
    sandbox 1 and `11` rollout files on sandbox 2;
  - after the final behavior gate, both sandbox queues were empty under
    `uv run sky queue <cluster> --all-users --skip-finished`.

Stage-13 source training eval:

| Source | Steps | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | `D_act_KL` | EffRank/n | Role probe | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `seed0_alpha0_2m` | `2,000,004` | `-4.835` | `96.127` | `100.961` | `0.451` | `12.323` | `0.302` | `0.383` | `zzd5s35r` |
| `seed1_alpha0_2m` | `2,000,004` | `-7.132` | `234.434` | `241.566` | `0.467` | `11.805` | `0.306` | `0.334` | `t1qe4j0o` |
| `seed2_alpha0_2m` | `2,000,004` | `-8.815` | `7.886` | `16.701` | `0.256` | `5.744` | `0.147` | `0.351` | `8h806fxf` |
| `seed3_alpha0_2m` | `2,000,004` | `-1.069` | `232.975` | `234.044` | `0.447` | `10.648` | `0.391` | `0.350` | `10qkn71k` |
| `seed4_alpha0_2m` | `2,000,004` | `-0.451` | `196.263` | `196.714` | `0.476` | `11.451` | `0.518` | `0.352` | `zqgmwl9e` |
| `seed5_alpha0_2m` | `2,000,004` | `-4.137` | `114.597` | `118.734` | `0.519` | `14.063` | `0.306` | `0.317` | `yflwy21l` |

Stage-13 source behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-30.467` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-35.433` | `47.7` | `1` | `0` | `0` | `6257` | `137` |
| `move_sweep` | `-30.467` | `0.0` | `0` | `0` | `0` | `611` | `0` |
| `use_sweep` | `-28.800` | `9.7` | `0` | `0` | `0` | `8611` | `29` |
| `chain_oracle` | `9.200` | `75.7` | `86` | `83` | `54` | `1979` | `227` |
| `seed0_alpha0_2m_det` | `6.133` | `67.0` | `76` | `74` | `48` | `1248` | `201` |
| `seed0_alpha0_2m_stoch` | `-9.523` | `60.7` | `77` | `62` | `40` | `916` | `182` |
| `seed1_alpha0_2m_det` | `-6.833` | `73.3` | `87` | `74` | `53` | `1894` | `220` |
| `seed1_alpha0_2m_stoch` | `-5.933` | `72.3` | `92` | `73` | `51` | `1019` | `217` |
| `seed2_alpha0_2m_det` | `-36.660` | `21.0` | `34` | `16` | `10` | `1090` | `63` |
| `seed2_alpha0_2m_stoch` | `-36.697` | `41.3` | `59` | `37` | `26` | `324` | `124` |
| `seed3_alpha0_2m_det` | `5.400` | `64.7` | `72` | `68` | `51` | `2726` | `194` |
| `seed3_alpha0_2m_stoch` | `13.567` | `74.7` | `85` | `77` | `57` | `663` | `224` |
| `seed4_alpha0_2m_det` | `11.633` | `70.7` | `81` | `74` | `54` | `2087` | `212` |
| `seed4_alpha0_2m_stoch` | `1.167` | `73.0` | `89` | `75` | `51` | `1030` | `219` |
| `seed5_alpha0_2m_det` | `16.800` | `79.3` | `90` | `86` | `59` | `1731` | `238` |
| `seed5_alpha0_2m_stoch` | `12.133` | `80.3` | `92` | `87` | `59` | `1291` | `241` |

Stage-13 promoted alpha0-to-alpha0.6 training eval:

| Promotion | Source | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | `D_act_KL` | EffRank/n | Role probe | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `seed0_anneal0to0.6` | `seed0_alpha0_2m` | `-0.201` | `136.827` | `137.028` | `0.531` | `14.137` | `0.279` | `0.315` | `fyomzprl` |
| `seed1_anneal0to0.6` | `seed1_alpha0_2m` | `-5.614` | `123.095` | `128.709` | `0.525` | `14.804` | `0.276` | `0.297` | `r9cb1p3k` |
| `seed2_anneal0to0.6` | `seed2_alpha0_2m` | `-0.481` | `124.664` | `125.145` | `0.421` | `10.190` | `0.207` | `0.303` | `3nr014eh` |
| `seed3_anneal0to0.6` | `seed3_alpha0_2m` | `-0.389` | `175.293` | `175.682` | `0.490` | `13.368` | `0.281` | `0.281` | `6uco67fs` |
| `seed4_anneal0to0.6` | `seed4_alpha0_2m` | `-4.042` | `134.814` | `138.856` | `0.513` | `13.961` | `0.285` | `0.341` | `7w6thihp` |
| `seed5_anneal0to0.6` | `seed5_alpha0_2m` | `1.826` | `196.609` | `194.783` | `0.493` | `12.053` | `0.252` | `0.320` | `ji3zx0si` |

Stage-13 promoted behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-28.800` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-33.733` | `63.7` | `5` | `0` | `0` | `6234` | `186` |
| `move_sweep` | `-36.717` | `0.0` | `0` | `0` | `0` | `1160` | `0` |
| `use_sweep` | `-38.700` | `10.7` | `15` | `0` | `0` | `8608` | `32` |
| `chain_oracle` | `-19.137` | `28.0` | `37` | `28` | `17` | `2357` | `84` |
| `seed0_anneal0to0.6_det` | `-10.333` | `48.0` | `64` | `45` | `33` | `2531` | `144` |
| `seed0_anneal0to0.6_stoch` | `-26.723` | `59.3` | `77` | `55` | `44` | `1466` | `178` |
| `seed1_anneal0to0.6_det` | `10.833` | `68.3` | `75` | `73` | `53` | `1744` | `205` |
| `seed1_anneal0to0.6_stoch` | `-70.337` | `71.7` | `99` | `60` | `52` | `1128` | `215` |
| `seed2_anneal0to0.6_det` | `-15.133` | `38.3` | `52` | `41` | `18` | `1442` | `115` |
| `seed2_anneal0to0.6_stoch` | `-9.897` | `46.0` | `57` | `49` | `29` | `545` | `138` |
| `seed3_anneal0to0.6_det` | `-1.333` | `57.0` | `64` | `60` | `43` | `1735` | `171` |
| `seed3_anneal0to0.6_stoch` | `-21.030` | `55.7` | `67` | `57` | `40` | `840` | `167` |
| `seed4_anneal0to0.6_det` | `-16.067` | `72.7` | `96` | `67` | `50` | `1220` | `218` |
| `seed4_anneal0to0.6_stoch` | `-28.723` | `58.0` | `79` | `54` | `37` | `957` | `174` |
| `seed5_anneal0to0.6_det` | `-8.000` | `75.0` | `82` | `79` | `61` | `1143` | `225` |
| `seed5_anneal0to0.6_stoch` | `-53.520` | `62.3` | `95` | `49` | `41` | `1079` | `187` |

Stage-13 decision:

- Source acquisition:
  - deterministic source behavior was valid for seeds `0,1,3,4,5`;
  - seed `2` was deterministic-weak at the fixed `2M` source budget
    (`-36.660` raw reward, `10` hearts), although it retained nonzero stochastic
    behavior (`26` hearts).
- Promoted behavior:
  - all six deterministic promoted policies beat no-op/random on raw reward and
    produced nonzero ore pickups, battery crafts, heart deposits, and use
    successes;
  - heart deposits were `33,53,18,43,50,61` for seeds `0..5`.
- Strict preservation gate:
  - seed0 source hearts `48`, threshold `36`, promoted `33`: fail by `3`;
  - seed1 source hearts `53`, threshold `39.75`, promoted `53`: pass;
  - seed2 source hearts `10`, threshold `7.5`, promoted `18`: pass, but the
    source itself was deterministic-weak;
  - seed3 source hearts `51`, threshold `38.25`, promoted `43`: pass;
  - seed4 source hearts `54`, threshold `40.5`, promoted `50`: pass;
  - seed5 source hearts `59`, threshold `44.25`, promoted `61`: pass.
- The primary Stage-13 stream is therefore behavior-positive but not strictly
  canonical. It proves we now have meaningful chain behavior under a fixed
  six-seed, uniform-source protocol, but it does not yet justify updating the
  paper as a clean MAPPO role-learning result because source acquisition is
  seed-fragile and alpha0.6 misses preservation on seed0.
- Next discriminating experiment:
  - run a conservative Stage-14 alpha0-to-alpha0.5 annealed promotion from the
    same Stage-13 source checkpoints;
  - use the same six seeds and behavior gates;
  - the specific question is whether a lower final shared-reward fraction keeps
    the meaningful all-seed behavior while recovering seed0 preservation.

V10 Stage-14 alpha0-to-alpha0.5 annealed-transfer outcome, commit
`436a07de7`:

- Purpose: test whether the Stage-13 seed0 preservation miss is caused by final
  shared-reward fraction being too high.
- Source checkpoints: the Stage-13 strict-v10 alpha0 `2M` checkpoints, still
  using the fixed six-seed source set.
- Stage-14 promotion root:
  `/workspace/tribal_event_mask_runs/stage14_v10_alpha0_to_alpha0p5_anneal_promotion`.
- Stage-14 behavior root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage14_v10_alpha0_to_alpha0p5_anneal_promotion_436a07de71`.
- Stage-14 launch scripts:
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage14_alpha0_to_alpha0p5_anneal_sandbox1_436a07de71.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage14_alpha0_to_alpha0p5_anneal_sandbox2_436a07de71.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage14_alpha0_to_alpha0p5_behavior_sandbox1_436a07de71.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage14_alpha0_to_alpha0p5_behavior_sandbox2_436a07de71.sh`
- Jobs:
  - promotion: `relh-sandbox-1` job `162`, `relh-sandbox-2` job `161`;
  - behavior: `relh-sandbox-1` job `163`, `relh-sandbox-2` job `162`.
- Validation:
  - behavior validation passed for `6` checkpoint rollout files on sandbox 1
    and `11` rollout files on sandbox 2;
  - after the final behavior gate, both sandbox queues were empty.

Stage-14 training eval:

| Promotion | Source | Eval raw return | Eval shaped/individual return | Eval role-shaping return | `D_act_JS` | `D_act_KL` | EffRank/n | Role probe | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `seed0_anneal0to0.5` | `seed0_alpha0_2m` | `-8.205` | `21.595` | `29.800` | `0.459` | `12.405` | `0.227` | `0.297` | `ixgrr5z5` |
| `seed1_anneal0to0.5` | `seed1_alpha0_2m` | `1.893` | `187.654` | `185.761` | `0.431` | `10.343` | `0.271` | `0.359` | `ibts2eus` |
| `seed2_anneal0to0.5` | `seed2_alpha0_2m` | `-9.897` | `3.105` | `13.001` | `0.295` | `7.825` | `0.164` | `0.322` | `tvdv4xxg` |
| `seed3_anneal0to0.5` | `seed3_alpha0_2m` | `0.835` | `229.083` | `228.248` | `0.423` | `10.334` | `0.297` | `0.311` | `yemag6nn` |
| `seed4_anneal0to0.5` | `seed4_alpha0_2m` | `-7.322` | `153.077` | `160.399` | `0.518` | `14.496` | `0.271` | `0.308` | `6953ens5` |
| `seed5_anneal0to0.5` | `seed5_alpha0_2m` | `-6.229` | `121.220` | `127.449` | `0.493` | `12.931` | `0.297` | `0.363` | `fkhy83bq` |

Stage-14 behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-28.800` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-43.700` | `26.7` | `5` | `0` | `0` | `6354` | `76` |
| `move_sweep` | `-28.800` | `0.0` | `0` | `0` | `0` | `1380` | `0` |
| `use_sweep` | `-32.133` | `3.3` | `0` | `0` | `0` | `8630` | `10` |
| `chain_oracle` | `-24.733` | `52.7` | `62` | `55` | `37` | `2209` | `158` |
| `seed0_anneal0to0.5_det` | `-7.967` | `39.7` | `51` | `48` | `19` | `902` | `119` |
| `seed0_anneal0to0.5_stoch` | `-35.420` | `54.7` | `68` | `59` | `36` | `783` | `164` |
| `seed1_anneal0to0.5_det` | `-30.213` | `20.0` | `41` | `15` | `3` | `1110` | `60` |
| `seed1_anneal0to0.5_stoch` | `-31.497` | `36.3` | `58` | `29` | `21` | `530` | `109` |
| `seed2_anneal0to0.5_det` | `-26.790` | `26.0` | `31` | `27` | `17` | `1214` | `78` |
| `seed2_anneal0to0.5_stoch` | `1.900` | `65.7` | `79` | `64` | `48` | `432` | `197` |
| `seed3_anneal0to0.5_det` | `-44.950` | `50.0` | `67` | `51` | `31` | `1516` | `150` |
| `seed3_anneal0to0.5_stoch` | `16.300` | `79.7` | `91` | `84` | `59` | `1304` | `239` |
| `seed4_anneal0to0.5_det` | `4.100` | `77.3` | `87` | `80` | `61` | `2435` | `232` |
| `seed4_anneal0to0.5_stoch` | `-3.167` | `80.0` | `95` | `83` | `61` | `2052` | `240` |
| `seed5_anneal0to0.5_det` | `4.533` | `69.7` | `80` | `75` | `52` | `1700` | `209` |
| `seed5_anneal0to0.5_stoch` | `-10.567` | `83.3` | `101` | `87` | `60` | `1027` | `250` |

Stage-14 decision:

- Alpha0-to-alpha0.5 is not a promotion candidate.
- It did not fix seed0: deterministic heart deposits dropped from Stage-13
  alpha0.6's `33` to `19`, farther below the `36` preservation threshold.
- It created new deterministic failures:
  - seed1 dropped from `53` hearts at alpha0.6 to `3` hearts at alpha0.5;
  - seed3 retained nonzero chain work but raw reward fell to `-44.950`, below
    no-op/random.
- Stochastic behavior remained strong for several seeds, especially seeds
  `2,3,4,5`, but the paper gate is deterministic behavior first.
- Interpretation: lower final alpha is not the missing ingredient. The
  Stage-13 alpha0.6 candidate remains the stronger mixed-reward condition.

V10 Stage-15 higher-N behavior audit plan:

- Purpose: reduce behavior-gate noise before deciding whether Stage-13 alpha0.6
  is truly a strict-gate failure.
- Run higher-sample behavior audits, not new training:
  - Stage-13 alpha0 source checkpoints, seeds `0..5`;
  - Stage-13 alpha0-to-alpha0.6 promoted checkpoints, seeds `0..5`.
- Use the same environment, chain-compass observation, and action-mask metadata.
- Increase behavior sampling from `3` to `10` episodes at `240` steps.
- Primary questions:
  - Does source seed `2` remain deterministic-weak with more episodes?
  - Does promoted seed `0` remain below 75% source-heart preservation, or was
    the `33` vs `36` miss 3-episode variance?
  - Do all promoted deterministic policies still beat no-op/random on raw reward
    and maintain nonzero chain work?
- If Stage-15 shows Stage-13 alpha0.6 passes within higher-N uncertainty, run
  representation/probe analysis from Stage-13.
- If Stage-15 confirms the miss, the next training diagnostic should focus on
  source acquisition stability for seed2 and a targeted seed0 preservation
  schedule, not lower final alpha.

V10 Stage-15 higher-N behavior audit outcome, launch commit `89a6701f9`:

- Purpose: resolve whether the Stage-13 seed0 preservation miss and seed2
  source weakness were real failures or artifacts of the 3-episode behavior
  gate.
- This was an audit only, not new training.
- Source checkpoints: Stage-13 strict-v10 alpha0 `2M` checkpoints, seeds `0..5`.
- Promoted checkpoints: Stage-13 alpha0-to-alpha0.6 annealed checkpoints, seeds
  `0..5`.
- Source audit root:
  `/workspace/tribal_event_mask_runs/behavior_audit_stage15_v10_alpha0_sources_10ep_89a6701f94`.
- Alpha0.6 promotion audit root:
  `/workspace/tribal_event_mask_runs/behavior_audit_stage15_v10_alpha0p6_promotion_10ep_89a6701f94`.
- Launch scripts:
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage15_behavior_audit_sandbox1_89a6701f94.sh`
  - `/workspace/tribal_event_mask_runs/scripts/launch_stage15_behavior_audit_sandbox2_89a6701f94.sh`
- Jobs:
  - `relh-sandbox-1` job `164`: source and promoted checkpoint audits for
    seeds `0,1,2`.
  - `relh-sandbox-2` job `163`: source and promoted baselines plus checkpoint
    audits for seeds `3,4,5`.
- Validation:
  - sandbox 1 validated `6` source rollout files and `6` promoted rollout
    files;
  - sandbox 2 validated `11` source rollout files and `11` promoted rollout
    files;
  - after the audit, both sandbox queues were empty.

Stage-15 source baselines over 10 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-30.300` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-34.549` | `50.1` | `11` | `0` | `0` | `20678` | `483` |
| `move_sweep` | `-31.300` | `0.0` | `0` | `0` | `0` | `2393` | `0` |
| `use_sweep` | `-29.800` | `10.9` | `0` | `0` | `0` | `28704` | `109` |
| `chain_oracle` | `-1.717` | `66.1` | `255` | `234` | `169` | `4756` | `661` |

Stage-15 source checkpoint audit over 10 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed0_alpha0_2m_det` | `-6.030` | `61.90` | `247` | `220` | `142` | `7904` | `619` |
| `seed0_alpha0_2m_stoch` | `-11.294` | `64.20` | `270` | `218` | `146` | `3226` | `642` |
| `seed1_alpha0_2m_det` | `-7.083` | `67.10` | `270` | `235` | `157` | `4792` | `671` |
| `seed1_alpha0_2m_stoch` | `-20.321` | `64.00` | `280` | `212` | `141` | `2835` | `640` |
| `seed2_alpha0_2m_det` | `-22.668` | `21.40` | `93` | `66` | `45` | `2327` | `214` |
| `seed2_alpha0_2m_stoch` | `-27.850` | `46.80` | `219` | `141` | `94` | `856` | `468` |
| `seed3_alpha0_2m_det` | `-15.298` | `51.40` | `217` | `176` | `116` | `4146` | `514` |
| `seed3_alpha0_2m_stoch` | `11.460` | `80.10` | `308` | `286` | `198` | `4720` | `801` |
| `seed4_alpha0_2m_det` | `2.290` | `64.70` | `243` | `232` | `156` | `7211` | `647` |
| `seed4_alpha0_2m_stoch` | `-11.996` | `72.80` | `295` | `250` | `173` | `4368` | `728` |
| `seed5_alpha0_2m_det` | `-4.691` | `68.40` | `280` | `234` | `163` | `6100` | `684` |
| `seed5_alpha0_2m_stoch` | `-0.850` | `75.40` | `295` | `255` | `191` | `4853` | `754` |

Stage-15 alpha0.6 promotion baselines over 10 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | `-28.800` | `0.0` | `0` | `0` | `0` | `0` | `0` |
| `random` | `-35.142` | `38.20` | `5` | `0` | `0` | `19475` | `371` |
| `move_sweep` | `-37.327` | `0.0` | `0` | `0` | `0` | `3344` | `0` |
| `use_sweep` | `-28.800` | `7.3` | `0` | `0` | `0` | `28697` | `73` |
| `chain_oracle` | `-14.101` | `63.40` | `265` | `216` | `148` | `4211` | `634` |

Stage-15 alpha0.6 promoted checkpoint audit over 10 episodes x 240 steps:

| Rollout | Raw reward | Task events | Ore pickups | Battery crafts | Heart deposits | Invalid attempts | Use successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed0_anneal0to0.6_det` | `-0.757` | `68.80` | `271` | `246` | `166` | `5062` | `688` |
| `seed0_anneal0to0.6_stoch` | `9.820` | `83.30` | `316` | `297` | `207` | `5729` | `833` |
| `seed1_anneal0to0.6_det` | `-2.793` | `72.00` | `291` | `243` | `182` | `5459` | `720` |
| `seed1_anneal0to0.6_stoch` | `8.305` | `75.70` | `296` | `268` | `185` | `5312` | `757` |
| `seed2_anneal0to0.6_det` | `-18.938` | `41.40` | `178` | `150` | `72` | `3065` | `414` |
| `seed2_anneal0to0.6_stoch` | `-8.166` | `54.30` | `218` | `179` | `127` | `1536` | `543` |
| `seed3_anneal0to0.6_det` | `-20.150` | `57.00` | `241` | `183` | `131` | `7019` | `570` |
| `seed3_anneal0to0.6_stoch` | `-12.271` | `67.30` | `266` | `228` | `173` | `3294` | `673` |
| `seed4_anneal0to0.6_det` | `-13.585` | `65.10` | `278` | `211` | `149` | `4700` | `651` |
| `seed4_anneal0to0.6_stoch` | `-4.127` | `66.90` | `279` | `227` | `157` | `4416` | `669` |
| `seed5_anneal0to0.6_det` | `-0.370` | `64.50` | `257` | `222` | `156` | `4038` | `645` |
| `seed5_anneal0to0.6_stoch` | `5.840` | `79.00` | `308` | `277` | `194` | `3697` | `790` |

Stage-15 preservation check:

| Seed | Source deterministic hearts | 75% threshold | Promoted deterministic hearts | Decision |
| ---: | ---: | ---: | ---: | --- |
| `0` | `142` | `106.5` | `166` | pass |
| `1` | `157` | `117.75` | `182` | pass |
| `2` | `45` | `33.75` | `72` | pass |
| `3` | `116` | `87` | `131` | pass |
| `4` | `156` | `117` | `149` | pass |
| `5` | `163` | `122.25` | `156` | pass |

Stage-15 decision:

- The higher-N audit resolves the Stage-13 fragility in the positive direction:
  the Stage-13 alpha0-to-alpha0.6 annealed stream passes the six-seed
  deterministic behavior gate at `10` episodes.
- All six source deterministic checkpoints beat no-op/random baselines on raw
  reward and produced nonzero ore pickup, battery crafting, heart deposits, and
  successful `use` actions. Source seed `2` remains the weakest acquisition seed
  (`45` hearts vs `142..163` for the other deterministic source policies), but
  it is not behavior-empty.
- All six promoted deterministic checkpoints beat no-op/random/move/use
  baselines on raw reward and produced nonzero chain work.
- All six promoted deterministic checkpoints pass the 75% source-heart
  preservation rule under the higher-N audit.
- The next paper-eligible step is representation/probe analysis for the
  Stage-13 alpha0 source and alpha0-to-alpha0.6 promoted checkpoints. Failed
  alpha0.5, alpha0.8, and alpha1.0 diagnostics should remain in the provenance
  log as negative/diagnostic evidence, not as paper-positive conditions.

V10 Stage-16 behavior-backed representation readout:

- Purpose: aggregate the canonical-format result JSONs for the first
  behavior-valid Stage-13 stream.
- This was not a new training run. It copied and validated the existing
  Stage-13 result JSONs:
  - alpha0 source seeds `0..5`;
  - alpha0-to-alpha0.6 annealed promoted seeds `0..5`.
- Local copied result root:
  `/Users/relh/Code/tasha/canonical-reward-geometry-results/stage13_result_json`.
- Aggregated summary:
  `/Users/relh/Code/tasha/canonical-reward-geometry-results/stage13_result_json/stage13_alpha0_vs_alpha0p6_geometry_summary.json`.
- Repo provenance audit artifact:
  `/Users/relh/Code/tasha/canonical-reward-geometry-results/stage13_result_json/repo_reward_geometry_provenance_audit.json`.
- Validation command:

```bash
uv run python v3_experiments/validate_canonical_reward_geometry_results.py \
  --allow-smoke \
  --allow-missing-checkpoint \
  /Users/relh/Code/tasha/canonical-reward-geometry-results/stage13_result_json/alpha0_sources/*.json \
  /Users/relh/Code/tasha/canonical-reward-geometry-results/stage13_result_json/alpha0p6_promotion/*.json
```

- Validation result: `validated 12 canonical reward-geometry result file(s)`.
- The `--allow-smoke` flag is required because this stream intentionally uses
  `2M` alpha0 source training plus `250k` promotion, not the older 4M full-run
  validator threshold.

Stage-16 all-six representation summary:

| Condition | n | EffRank/n | Ordered `D_act_KL` | `D_act_JS` | Fixed-role probe | Raw env return | Individual return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-13 alpha0 source | `6` | `0.328 +/- 0.122` | `11.006 +/- 2.819` | `0.436 +/- 0.092` | `0.348 +/- 0.022` | `-4.406 +/- 3.283` | `147.047 +/- 89.949` |
| Stage-13 alpha0-to-alpha0.6 promoted | `6` | `0.264 +/- 0.030` | `13.086 +/- 1.694` | `0.496 +/- 0.040` | `0.309 +/- 0.021` | `-1.483 +/- 2.771` | `148.550 +/- 30.230` |

Stage-16 source-stronger subset summary, excluding weak source seed `2`:

| Condition | n | EffRank/n | Ordered `D_act_KL` | `D_act_JS` | Fixed-role probe | Raw env return | Individual return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-13 alpha0 source, seeds `0,1,3,4,5` | `5` | `0.365 +/- 0.094` | `12.058 +/- 1.276` | `0.472 +/- 0.029` | `0.347 +/- 0.024` | `-3.525 +/- 2.765` | `174.879 +/- 65.603` |
| Stage-13 alpha0-to-alpha0.6 promoted, seeds `0,1,3,4,5` | `5` | `0.275 +/- 0.013` | `13.665 +/- 1.036` | `0.511 +/- 0.019` | `0.311 +/- 0.023` | `-1.684 +/- 3.049` | `153.328 +/- 31.164` |

Stage-16 interpretation:

- Behavior is now meaningful: Stage 15 showed all six promoted deterministic
  policies perform nonzero chain work, beat baselines, and preserve at least 75%
  of source deterministic heart deposits.
- The original fixed-role probe claim is not recovered. Both alpha0 and alpha0.6
  fixed `agent_id % 3` role-probe accuracies are near the `1/3` chance baseline.
- Excluding weak source seed `2` does not change that conclusion; alpha0 remains
  near chance (`0.347 +/- 0.024`).
- Alpha0.6 has lower EffRank/n than alpha0 in this behavior-backed stream
  (`0.264` vs `0.328` all-six; `0.275` vs `0.365` excluding seed2), but action
  diversity metrics move upward (`D_act_JS` `0.496` vs `0.436` all-six). This is
  not the old monotonic-collapse story.
- The paper-safe claim should therefore be revised: after reward redesign, MAPPO
  can learn sensible Tribal Village chain behavior, but the behavior-backed
  fixed-role representation evidence is weak. We should not claim clean learned
  fixed roles unless a follow-up analysis finds a defensible behavior-derived
  role label or another pre-registered representation metric that aligns with
  the actual learned behavior.

V10 Stage-17 behavior-derived role audit:

- Purpose: test whether the recovered behavior contains role-like specialization
  even though the fixed `agent_id % 3` probe is near chance.
- Added reusable analyzer:
  `v3_experiments/analyze_tribal_behavior_roles.py`.
- Focused tests:
  `tests/v3_experiments/test_tribal_behavior_role_analysis.py`.
- Input artifacts: copied Stage-15 `rollout_metrics.json` files, using only
  deterministic checkpoint rollouts and excluding stochastic rollouts/baselines.
- Local input root:
  `/Users/relh/Code/tasha/canonical-reward-geometry-results/stage15_behavior_metrics`.
- Output artifact:
  `/Users/relh/Code/tasha/canonical-reward-geometry-results/stage15_behavior_metrics/stage15_deterministic_behavior_role_analysis.json`.
- Analyzer command:

```bash
find /Users/relh/Code/tasha/canonical-reward-geometry-results/stage15_behavior_metrics \
  -type f -name rollout_metrics.json \
  | grep -v baseline_ \
  | grep '_det/rollout_metrics.json' \
  | sort \
  | xargs uv run python v3_experiments/analyze_tribal_behavior_roles.py \
      --output /Users/relh/Code/tasha/canonical-reward-geometry-results/stage15_behavior_metrics/stage15_deterministic_behavior_role_analysis.json
```

Stage-17 behavior-role summary:

| Condition | Rollouts | Dominant stage counts | Mean agents with all stages | Mean specialized agents | Mean max fixed-role stage-share gap | Fixed-role separated rollouts |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Stage-13 alpha0 source | `6` | `ore: 72` | `12.0` | `0.167` | `0.0366` | `0` |
| Stage-13 alpha0-to-alpha0.6 promoted | `6` | `ore: 72` | `12.0` | `0.0` | `0.0368` | `0` |

Stage-17 interpretation:

- The recovered behavior is not a division of labor across gatherer,
  crafter/logistics, and depositor roles. Every deterministic checkpoint agent
  has nonzero ore pickup, battery craft, and heart deposit events across the
  10-episode audit.
- The reason every agent is "ore-dominant" is mechanical rather than semantic:
  the learned task is a serial ore-to-battery-to-heart chain, so successful
  agents necessarily produce at least as many ore events as downstream battery
  and heart events.
- The fixed `agent_id % 3` groups are behaviorally similar. The mean maximum
  stage-share gap across fixed role groups is only about `0.037`, far below the
  `0.15` audit threshold.
- This explains the Stage-16 near-chance fixed-role probe. The probe is not
  merely using the wrong labels for a hidden specialized behavior; the
  deterministic policies appear to implement a shared chain policy across all
  agents.
- Paper implication: the reconstructed setup now supports a claim about reward
  shaping recovering meaningful chain behavior, but it does not support the
  original MAPPO learned-role/specialization claim.

V10 Stage-18 paper update:

- Purpose: update the paper only after behavior validation, representation
  readout, and behavior-derived role audit were complete.
- Overleaf checkout:
  `/Users/relh/Code/overleaf/representation-collapse-paper`.
- Overleaf commit:
  `eef90a1` (`Revise Tribal Village result after behavior audit`).
- Files updated:
  - `samples/main.tex`;
  - `runs.md`.
- Paper changes:
  - replaces the old positive Tribal Village role-separable MAPPO claim with a
    behavior-first reconstruction narrative;
  - removes the old PCA teaser claim and old five-seed monotonic Tribal table
    from the paper body;
  - adds the behavior-backed Stage-16 representation table:
    alpha0 source probe `0.348 +/- 0.022`, alpha0-to-alpha0.6 probe
    `0.309 +/- 0.021`;
  - adds the Stage-17 behavior-derived role audit:
    `0/6` fixed-role separated rollouts for both alpha0 and alpha0-to-alpha0.6;
  - marks the old passive-shaping five-seed rerun as superseded provenance in
    `runs.md`.
- Validation:

```bash
cd /Users/relh/Code/overleaf/representation-collapse-paper/samples
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

- Validation result: build completed successfully and wrote `main.pdf`. Local
  build products were removed before commit, leaving only source/provenance
  changes.

Stage-19 specialization-forcing follow-up:

- Purpose: pursue the next research fork after the Stage-17 negative role audit:
  create a Tribal Village variant where division of labor is mechanically
  necessary, then rerun the same behavior-gated representation experiments only
  if the policies learn meaningful cross-role behavior.
- Old figure contract recovered from the original paper assets:
  - Tribal PCA plots compare individual vs shared/mixed reward conditions in
    side-by-side panels;
  - axes are encoder embedding PC1/PC2 with explained variance percentages;
  - point colors are fixed role labels and large outlined markers are
    per-agent embedding means;
  - panel subtitles report `EffRank/n`, ordered `D_act`, `D_JS`, and role-probe
    accuracy.
- New plotting tool:
  `v3_experiments/export_tribal_representation_artifacts.py`.
  It reloads canonical checkpoints, runs deterministic evaluation rollouts,
  saves `.npz` embedding/logit/action dumps, and regenerates old-style PCA
  figures plus a JSON summary. It accepts checkpoint path prefix rewrites so
  canonical JSONs can remain unchanged while artifacts are generated on a
  sandbox or local harvested directory.
- New simulator support:
  - `put` can now hand off ore and batteries, in addition to armor and bread;
  - action stats now log `put_ore` and `put_battery`;
  - behavior summaries count ore/battery handoffs as first-class task events.
- New reward/curriculum design:
  `event_v11_role_gated_chain_handoffs`.
  This is a diagnostic curriculum, not a negative-reward patch:
  - role `0` suppliers can mine ore and hand off ore;
  - role `1` crafters can convert ore into batteries and hand off batteries;
  - role `2` depositors can deposit batteries into the home assembler;
  - movement remains free;
  - dense guidance is potential-based and role-aware, with depositors pulled
    toward the home assembler rather than mines.
- Stage-19 launch gate:
  - local Python syntax check passed;
  - local Nim `environment.nim` check passed;
  - local mock trainer smoke passed;
  - local real Tribal backend smoke passed and rebuilt the widened stats buffer;
  - local representation-export smoke passed on the tiny v11 checkpoint.
- Initial sandbox ramp:
  - launch short `event_v11_role_gated_chain_handoffs` alpha0 diagnostics on
    `relh-sandbox-1/2`;
  - seeds should be split across boxes and kept short enough to inspect
    `put_ore`, `put_battery`, `craft_battery`, and `deposit_heart` before any
    long run;
  - if ore/battery handoffs stay at zero, do not scale. Iterate on navigation,
    role-aware observation breadcrumbs, or scripted/BC warm start first.
- Stage-19 short diagnostic result, commit `1b8b0d44`:
  - launched `event_v11_role_gated_chain_handoffs`, `shared_frac=0.0`,
    `1,000,008` agent steps, seeds `0,1,2,3`, split across
    `relh-sandbox-1/2`;
  - local artifact root:
    `v3_experiments/behavioral_reward_results/stage19_v11_role_gated_handoffs_1m_1b8b0d44d`;
  - old-style PCA diagnostic plot:
    `combined_all/tribal_pca_role_geometry.png`;
  - deterministic behavior gate used `5` episodes x `240` steps per seed with
    saved sandbox replays;
  - aggregate events across the four seeds: `1089` ore handoffs, `78` battery
    handoffs, `87` battery crafts, and `24` heart deposits;
  - per-seed heart deposits were `0,4,20,0`, so v11 fixed the handoff/crafting
    bottleneck but has not yet made depositor completion reliable;
  - per-seed representation readout:
    - seed `0`: EffRank/n `0.256`, D_act `8.872`, JS `0.307`, probe `0.300`;
    - seed `1`: EffRank/n `0.380`, D_act `5.992`, JS `0.227`, probe `0.358`;
    - seed `2`: EffRank/n `0.269`, D_act `10.594`, JS `0.374`, probe `0.459`;
    - seed `3`: EffRank/n `0.265`, D_act `7.847`, JS `0.292`, probe `0.400`;
  - interpretation: continue iterating v11 before launching reward-mixing
    representation sweeps. The next intervention should target reliable
    depositor battery-to-home completion without adding negative rewards.
- Stage-20 v12 depositor-reliability iteration:
  - diagnosis from the Stage-19 per-agent counters: depositors in failed seeds
    received batteries but often ended episodes still holding them, while
    generic `put_ore` and `put_battery` counters rewarded handoffs to any
    receiver role;
  - this was RL with shaped rewards, not scripted behavior. The policy still
    chose movement, timing, handoff direction, and final use actions. The strong
    curriculum was the role-gated action mask, which restricts each role's
    non-movement verbs to the intended chain affordance;
  - v12 keeps the positive-only shaping rule and adds no penalties;
  - new simulator counters distinguish correct-recipient handoffs:
    `put_ore_to_crafter`, `put_battery_to_depositor`,
    `receive_ore_from_supplier`, and `receive_battery_from_crafter`;
  - v12 reward changes:
    - supplier still gets ore pickup reward but gets the main handoff reward
      only when ore goes to a crafter;
    - crafter gets positive credit for receiving ore from a supplier, crafting
      batteries, and handing batteries to a depositor;
    - depositor gets positive credit for receiving a battery from a crafter and
      a larger `deposit_heart` reward;
    - depositor empty/home and battery/home potential scales are increased so
      battery-to-home completion is less sparse;
  - local validation passed:
    - Python syntax check;
    - Nim `environment.nim` check;
    - mock v12 trainer smoke;
    - real Tribal v12 trainer smoke, including rebuild of the 32-column stats
      shared library;
    - v12 checkpoint behavior rollout smoke confirmed the role-gated mask and
      new targeted-handoff metrics.

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
- [x] Add behavior-derived role analysis.
- [x] Add no-op, random, and simple scripted baseline policies.
- [x] Add replay generation to the rollout harness.
- [x] Implement event-based reward components.
- [x] Add reward component logging.
- [x] Add output validation.
- [x] Record reward-shaping source notes for the v9 correction.
- [x] Run tiny native no-op/random/scripted baseline smoke.
- [x] Run full no-op/random/scripted baseline gate.
- [x] Run short `shared_frac=0.0` training gate.
- [x] Compare behavior metrics.
- [ ] Inspect replays.
- [ ] Iterate on rewards until behavior passes.
- [x] Run first v10 chain-affordance cleanup ramp.
- [x] Rerun v10 after strict action-mask fallback fix.
- [x] Fix v10 checkpoint metadata so behavior rollouts use the effective mask.
- [x] Run v10 Stage-2 strict-mask 10M ramp.
- [x] Run v10 Stage-2 behavior gate.
- [x] Run from-scratch strict-v10 reward-mixing pilot.
- [x] Run Stage-6 reward-mixing behavior gate.
- [x] Run Stage-7 warm-start reward-mixing diagnostic.
- [x] Run Stage-7 warm-start behavior gate.
- [x] Run Stage-8 short/low-lr alpha0.8 stabilization diagnostic.
- [x] Run Stage-8 behavior gate.
- [x] Run Stage-9 lower-alpha threshold diagnostic.
- [x] Run Stage-9 behavior gate.
- [x] Run Stage-10 alpha0.6 short-budget diagnostic.
- [x] Run Stage-10 behavior gate.
- [x] Add and run Stage-11 alpha0-to-alpha0.6 annealed-transfer diagnostic.
- [x] Run Stage-11 behavior gate.
- [x] Run Stage-12 annealed alpha0.6 promotion diagnostic.
- [x] Document Stage-12 source-selection/source-budget confound.
- [x] Run Stage-13 pre-registered six-seed canonical source rerun.
- [x] Run Stage-13 source behavior gate.
- [x] Run Stage-13 annealed alpha0.6 promotion and behavior gates.
- [x] Document Stage-13 behavior-positive/strict-gate-fail outcome.
- [x] Run Stage-14 conservative alpha0-to-alpha0.5 annealed promotion.
- [x] Run Stage-14 behavior gates.
- [x] Document Stage-14 negative alpha0.5 outcome.
- [x] Run Stage-15 higher-N behavior audit for Stage-13 alpha0 and alpha0.6.
- [x] Run representation/probe analysis only after a strict behavior gate passes.
- [x] Run Stage-17 behavior-derived role audit after the fixed-role probe failed.
- [x] Resolve the canonical sweep decision after behavior-backed role audit.
- [x] Update the paper from canonical outputs only.
- [x] Recover the old PCA/representation figure contract from paper images.
- [x] Add representation artifact export and old-style PCA plotting.
- [x] Add v11 ore/battery handoff instrumentation.
- [x] Add v11 role-gated chain diagnostic reward/mask.
- [x] Smoke v11 locally on mock and real Tribal backends.
- [x] Launch Stage-19 short v11 sandbox diagnostics.
- [x] Inspect Stage-19 handoff/craft/deposit counters before scaling.
- [x] Rerun old-style representation plots for the Stage-19 alpha0 diagnostic.
- [x] Implement v12 depositor-reliability shaping and correct-recipient handoff counters.
- [ ] Launch Stage-20 short v12 sandbox diagnostics.
- [ ] Inspect Stage-20 correct-recipient handoffs and depositor completion before scaling.
- [ ] Rerun old-style representation plots only for behavior-valid v11/v12 reward-mixing streams.

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
