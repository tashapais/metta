# Alo CLI Commands (eval, diagnose, pickup, scrimmage)

> **Status:** Draft **Author:** Richard **Created:** 2026-01-13

## Summary

Define the four Cogames CLI commands and track progress on the Alo stack integration. The stack delivers eval, diagnose,
and pickup but does not yet implement scrimmage.

## Problem

The CLI commands for evaluation are being refactored to use Alo rollouts and scoring, but the command set is incomplete
and inconsistent. Scrimmage is missing, and there are no end-to-end tests for the command suite.

## Solution

Document the desired behavior of each command and align the Alo stack changes with those behaviors. Implement the
missing scrimmage command, set pickup defaults, and add tests for all four commands.

## Goals

- [ ] All four commands appear in `cogames --help` and run end-to-end.
- [ ] `eval/evaluate` uses Alo rollouts and assignment validation.
- [ ] `diagnose` uses the Alo single-episode runner.
- [ ] `pickup` defaults to a pool of `thinky` and `ladybug` when no pool is provided.
- [ ] `scrimmage` runs a single policy controlling all agents (no pool agents).
- [ ] Each command has at least one automated test.

## Non-Goals

- Changing training workflows or reward definitions.
- Redesigning the Alo scoring model.

## Spec Process Notes

- Status is for internal tracking; specs can be merged while still in Draft.
- We are not prescribing a specific review tool here.

## Design

### Command behavior

1. `cogames eval` / `cogames evaluate`

- Evaluate one or more policies across missions or mission sets.
- Uses Alo rollouts and assignment validation.
- Outputs per-mission summaries and overall results.

2. `cogames diagnose`

- Runs diagnostic evaluation suites for a policy checkpoint.
- Uses `packages/cogames/scripts/run_evaluation.py` under the hood.
- Uses Alo single-episode runner.

3. `cogames pickup`

- Evaluates a candidate policy against a pool of other agents.
- Default pool: `thinky` and `ladybug` (always-available scripted agents).
- Computes scenario scores and VOR (value-over-replacement) summaries.

4. `cogames scrimmage`

- Single policy controls all agents on a team (no pool agents).
- Equivalent to a self-play team run with uniform assignments.
- Should report a summary score and replay paths.

### Alo stack coverage

The graphite stack is linear:

- `origin/rt-alo-package` -> `origin/rt-alo-plumbing` -> `origin/rt-cogames-eval-alo` -> `origin/rt-cogames-cli` ->
  `origin/rt-cogames-pickup` -> `origin/rt-app-backend-alo-scoring`.

What it covers:

- Alo module and tests in `packages/alo`.
- Evaluation refactor (`cogames.evaluate`) to use Alo rollouts and assignment validation.
- CLI policy parsing updates.
- Pickup command implementation.
- Diagnose path updated to use Alo single-episode runner.

### Future package scope (name TBD)

We expect a new package (name TBD; candidates: `melee`, `gauntlet`, `marlpit`) to own the shared plumbing for policy
rollouts and judging. This spec does not rename anything yet, but the intended package scope includes:

- Policy rollout orchestration and result aggregation.
- Protobuf definitions and code generation for policy server interfaces.
- Policy packaging and metadata utilities (e.g., `policy-spec.json` creation/validation).
- Standard scoring/judging primitives (shared across eval, diagnose, pickup, scrimmage).

### Progress assessment

- `eval/evaluate`: implemented via Alo rollouts. Done.
- `diagnose`: implemented via Alo runner. Done.
- `pickup`: implemented. Done (but needs default pool behavior confirmed).
- `scrimmage`: not implemented.

### Work remaining

- Implement `scrimmage` command:
  - add CLI wiring in `packages/cogames/src/cogames/main.py`.
  - implement logic (use Alo rollouts with uniform assignments).
  - add output summary and replay handling.
- Ensure `pickup` defaults to `thinky` + `ladybug` when no pool is provided.
- Add CLI integration tests for all four commands.

## Open Questions

1. Should scrimmage reuse the same output format as pickup or match eval summaries?
2. Do we need a standardized replay location or naming convention for these commands?
