# Kickstarting for Launch

> **Status:** Draft **Author:** Richard **Created:** 2026-01-13

## Summary

Kickstarting is on the critical path to launch because it is the only demonstrated way to produce a reliable learned
policy in our environment. This spec defines the pre-launch success criteria and the post-launch dissemination plan.

## Problem

We must show a learned policy that reliably succeeds in the new game rules. Historically, only kickstarting from a
scripted policy has worked. The new rules invalidate prior results, so we need to rerun experiments and potentially
search for new kickstarting variants. The primary risk is experiment turnaround time rather than engineering effort.

## Solution

Pre-launch: rerun kickstarting experiments on the new game rules, validate cloning and transition to reward learning,
and identify fast-training variants to reduce iteration time. Post-launch: publish a practitioner blog post and a
research paper with ablations and exploratory results.

## Goals

- [ ] Demonstrate a learned policy that reliably succeeds under the new game rules.
- [ ] Validate kickstarting from a deterministic scripted policy.
- [ ] Validate a stable transition from cloning to reward-based learning.
- [ ] Identify at least one fast-training variant (e.g., breadcrumbs) to shorten iteration cycles.
- [ ] Produce a blog post aimed at practitioners explaining how others can accelerate their interactions with cogames
      using our methods.
- [ ] Produce a paper aimed at researchers with empirical results and ablations.

## Non-Goals

- Inventing a brand-new training paradigm beyond kickstarting.
- Large-scale engine or rules redesign to make training easier.

## Spec Process Notes

- Status is for internal tracking; specs can be merged while still in Draft.
- We are not prescribing a specific review tool here.

## Design

### Pre-launch plan

- Re-run kickstarting experiments from scratch under the new game rules.
- Confirm two hard requirements:
  - Cloning works from a deterministic scripted policy.
  - The handoff to reward-based learning is stable.
- If existing methods fail, expect a method-search phase gated by training time.
- Measure schedule risk primarily as experiment turnaround time.

### Post-launch plan

- Practitioner blog post (target: ~1 day to write once results are stable).
- Research paper that demonstrates why Cogames is an interesting research substrate and how our kickstarting methods
  perform compared to the canonical approach (longer, requires experiments + ablations).
- Include a capstone exploratory experiment that points to future Softmax research directions.

### Fast-training variants

- Identify environment variants that reduce training time (e.g., breadcrumbs or other scaffolds).
- These variants are required to speed up iteration and provide an approachable onramp for users.
- Expect up to a week of experimental iteration, with parallel work possible while runs execute.

## Open Questions

1. Which scripted policy should serve as the default kickstarting teacher on the new rules?
2. Which kickstarting recipe will be most performant under the new game rules?
3. Which fast-training variant is most representative while still accelerating experiments?
