# Cogsguard Training Readiness

> **Status:** Draft **Author:** Richard **Created:** 2026-01-13

## Summary

Ensure the Metta RL training stack can train and evaluate Cogsguard using the new mission and station configs, while
keeping PPO and behavioral cloning workflows intact.

## Problem

Cogsguard changes land inside the existing `cogames.cogs_vs_clips` package and add new resources, vibes, collectives,
and handler logic. The current training and evaluation entrypoints assume `cogs_vs_clips`, and behavioral cloning
(supervised, scripted cloner, and kickstarter modes) relies on stable action and observation layouts. Without explicit
wiring, Cogsguard will be difficult to train and may break BC workflows.

## Solution

Define how Cogsguard is named and discovered, then update training and evaluation entrypoints so PPO and BC can target
Cogsguard missions without manual edits. Add a minimal smoke test to confirm Cogsguard env construction and a short
rollout.

## Goals

- [ ] Decide and document the naming strategy for Cogsguard (module path and suite name).
- [ ] Training entrypoints can target Cogsguard without code edits.
- [ ] PPO recipes run on Cogsguard.
- [ ] BC workflows run on Cogsguard:
  - supervised (teacher actions)
  - scripted cloner (sliced cloner)
  - kickstarting from pre-trained agents
- [ ] Evaluation and diagnose tooling can run Cogsguard missions.
- [ ] Add a minimal smoke test for Cogsguard env creation or short rollout.

## Non-Goals

- Redesigning Cogsguard gameplay, rewards, or map generation.
- Reworking training infrastructure beyond what Cogsguard requires.

## Spec Process Notes

- Status is for internal tracking; specs can be merged while still in Draft.
- We are not prescribing a specific review tool here.

## Design

### Background: Cogsguard changes (from `origin/daveey-cogsguard-v2`)

- New `CogsGuardMission` and `CogConfig` in `packages/cogames/src/cogames/cogs_vs_clips`.
- New Cogsguard station configs (junction, hub, gear stations, extractor, chest).
- New arena site and `make_cogsguard_mission` helper.
- Recipe update: `recipes/experiment/cogsguard.py` now uses `make_cogsguard_mission` and `suite="cogsguard"`.
- Mettagrid adds handler/filter/mutation config support and tests for AOE and collectives.

### Naming and mission discovery

Two options:

- Option A: keep module path `cogames.cogs_vs_clips`, add a Cogsguard mission set and suite name.
- Option B: rename module path to `cogames.cogsguard` and update imports across recipes and tooling.

This spec assumes Option A unless we decide otherwise.

Preferred direction: `cogames` acts as an umbrella runner for many games. In that model, Cogsguard missions can live in
the Cogsguard game repo/package while `cogames` provides CLI integration and a stable mission discovery surface (e.g.,
via registered mission providers or import hooks).

### Training entrypoints

Update the entrypoints that currently hard-code `cogames.cogs_vs_clips` so they can target Cogsguard:

- `packages/cogames/src/cogames/train.py`: stop hard-coding `env_name` or add a config switch.
- `recipes/experiment/cogs_v_clips.py`: add a Cogsguard variant or allow mission/suite overrides.
- `recipes/experiment/cvc/*`: allow Cogsguard missions in curricula and eval suites.

### PPO support

Ensure PPO runs on Cogsguard with no manual code edits:

- `recipes/experiment/cogs_v_clips.py` for base PPO curriculum and eval suites.
- `recipes/experiment/machina_1.py` and other wrappers that call the base recipe.

### Behavioral cloning support

BC has three modes that must be validated on Cogsguard:

1. Supervised (teacher actions)

- Loss: `metta/rl/loss/action_supervised.py` (exposed as `losses.supervisor`).
- Activated via `TeacherConfig(mode="supervisor")` in `metta/rl/training/teacher.py`.

2. Scripted cloner (sliced cloner)

- Loss: `metta/rl/loss/sliced_scripted_cloner.py`.
- Recipes: `recipes/experiment/cvc/cloner.py`, `recipes/experiment/cvc/sliced_cloner.py`,
  `recipes/experiment/cvc/machina1_cloner.py`.

3. Kickstarting from pre-trained agents

- Losses: `kickstarter`, `sliced_kickstarter`, `logit_kickstarter`, `eer_kickstarter`, `sl_checkpointed_kickstarter`.
- Recipes: `recipes/experiment/abes/kickstart/*`.

All three modes rely on stable action and observation layouts. Cogsguard resources and vibes must not invalidate
existing assumptions or require manual patching.

### Evaluation and mission listing

- Extend mission discovery in `cogames.cli.mission` and `cogames.evaluate` to include Cogsguard missions.
- Ensure `packages/cogames/scripts/run_evaluation.py` and diagnose flows can target Cogsguard.
- Update Gridworks mission listing endpoints to include Cogsguard missions.

## Open Questions

1. Are we keeping `cogames.cogs_vs_clips` as the module path, or renaming to `cogames.cogsguard`?
2. Which Cogsguard mission(s) should be the default training and evaluation targets?
