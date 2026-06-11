# Claim Ledger — paper_exp_reward_type.py collapse hunt

Symptom: late-training return collapse (peak ~1.0-1.5 rolling quality, then
50-90% degradation before 5M steps), both reward conditions, healthy entropy,
rising action diversity, small value loss, persists at near-zero LR, surviving
seeds flip across recipe variants.

## Claims

### C1 — adv-norm-on-noise
- Topic: advantage normalization
- Status: OPEN
- Cycle introduced: 0 (mapper)
- Claim: Per-minibatch advantage normalization rescales pure value-noise
  minibatches (rewards ~1 event per 1e4 samples; many 512-sample minibatches
  contain zero reward events) to unit-std policy gradients, injecting
  full-strength noise updates.
- Best supporting evidence: paper_exp_reward_type.py:634; reward sparsity from
  map; mechanism consistent with stochastic peak-then-decay + healthy entropy.
- Best counterevidence: none yet; needs quantification of zero-event minibatch
  fraction and pre-norm adv std.
- Files: v3_experiments/paper_exp_reward_type.py:624-669
- Next test: instrument or reason precisely about adv std in reward-free
  minibatches; consider batch-level normalization or adv-std floor.

### C2 — coherent-noise-reuse
- Topic: minibatch reuse / staleness
- Status: OPEN
- Cycle introduced: 0 (mapper)
- Claim: Advantages computed once per rollout are reused for 96 Adam steps
  (8 epochs x 12 minibatches) with no KL guard, so noise directions from C1
  apply coherently and compound.
- Files: same loop.
- Next test: interacts with C1; verify ratio drift across epochs.

### C3 — zero-bootstrap-at-truncation
- Topic: GAE / truncation
- Status: STRENGTHENED (pre-skill ablation evidence)
- Cycle introduced: 0 (pre-skill, Phase A-3)
- Claim: Every episode ends at the wrapper time limit but GAE treats done=1 as
  terminal (zero bootstrap); bias grows with V as policy improves.
- Best supporting evidence: A-3 ablation — shared arm final/p90 0.15/0.14 ->
  0.86/0.53 with the fix.
- Best counterevidence: individual arm regressed under the fix (0.99 -> 0.07
  on one seed) — but see C4.
- Next test: see C4 (the fix as implemented has a leak).

### C4 — trunc_bootstrap-shared-leak + self-reference
- Topic: truncation fix implementation
- Status: OPEN
- Cycle introduced: 0 (mapper)
- Claim: The --trunc_bootstrap patch adds PER-AGENT gamma*V(s_T) to the final
  reward BEFORE the shared-mean transform is applied?? (verify order) — and in
  the shared condition injects per-agent-differentiated signal into training
  that should be team-uniform; it also places V in its own unclipped MSE
  target (self-reference).
- Files: paper_exp_reward_type.py:512-523 (patch), env step (sharing order).
- Next test: trace whether the patch lands after the team-mean transform
  (it does — patch applied in trainer on returned rews, which are already
  shared-transformed), so per-agent V values break reward uniformity in the
  shared condition. Also assess value self-reference dynamics.

### C5 — map-seed-not-wired
- Topic: environment seeding / nonstationarity
- Status: OPEN
- Cycle introduced: 0 (mapper)
- Claim: MapGen.seed=None and no Simulator parent means run seed and
  seed_offset=global_step never reach map generation: every episode uses a
  fresh OS-entropy random map. Consequences: (a) run "seeds" do not control
  the environment (explains seed-flipping across variants), (b) per-episode
  task distribution is high-variance, (c) reproducibility broken.
- Files: packages/mettagrid/python/src/mettagrid/simulator/simulator.py:270-279.
- Next test: verify statically that Simulation(cfg, seed=...) does not seed
  MapGen; check whether identical seeds give different maps.

### C6 — value-grad-through-shared-trunk
- Topic: actor-critic interference
- Status: OPEN
- Cycle introduced: 0 (mapper)
- Claim: Unclipped 0.5*MSE value gradients flow through the shared encoder and
  can reshape actor features late in training (when V targets shift due to
  C3/C5 variance).
- Next test: lower-priority; consider value clip or smaller vf coef only if
  C1/C3/C4/C5 fixes insufficient.

### C7 — infonce-gradient-dead (latent; off in failing runs)
- Topic: contrastive path
- Status: OPEN (flagged for historical claims, not this symptom)
- Claim: emb_t for InfoNCE is built from numpy buffers with no autograd graph;
  the --contrastive loss is constant wrt parameters. Irrelevant to current
  collapse (flag off) but invalidates historical "contrastive" conclusions if
  confirmed.
- Next test: confirm detachment statically; report to PLAN.md regardless.

## Topics

| Topic | Cycles touched | Last new evidence | Status | Note |
| --- | --- | --- | --- | --- |
| advantage normalization | 0 | 0 | ACTIVE | mapper hotspot |
| minibatch reuse | 0 | 0 | ACTIVE | couples to C1 |
| GAE/truncation | 0 | 0 | ACTIVE | A-3 evidence + C4 leak |
| env seeding/nonstationarity | 0 | 0 | ACTIVE | new mapper discovery |
| actor-critic interference | 0 | 0 | ACTIVE | secondary |
| contrastive path | 0 | 0 | ACTIVE | latent, historical impact |
| entropy collapse | - | - | OUT_OF_SCOPE | measured, ruled out |
| LR/epochs as root cause | - | - | EXHAUSTED | A-1 ablated |
| combat dynamics | - | - | EXHAUSTED | A-2 ablated (contributing) |

## Cycle 1 verifier verdicts (mech_verify_cycle_1.md)

- C1 SUPPORTED (analytic: ~25% of minibatches zero reward signal at peak,
  >=95% of samples pure critic noise; runtime fraction still unmeasured —
  watch-item, instrument line 634 if A-4 fails). Fix implemented:
  --batch_adv_norm. Status -> STRENGTHENED.
- C4 SUPPORTED (code-order fact: team-mean transform inside env.step precedes
  the per-agent patch). Self-reference sub-claim REFINED AWAY (standard TD
  bootstrapping). Fix implemented: team-mean bootstrap in shared arm.
  Status -> VERIFIED.
- C5 SUPPORTED statically AND empirically (same seed -> different maps
  pre-fix; explicit map_builder.seed -> identical). Fix implemented:
  --seed_maps. Status -> VERIFIED.
- C7 SUPPORTED statically AND empirically (requires_grad=False, param grads
  bit-identical with/without the term). Fix implemented: gradient-attached
  re-encoding. Status -> VERIFIED.

## New claims from cycle 1

### C8 — contrastive-as-seed-perturbation
- Status: VERIFIED (verifier empirical)
- Claim: historical --contrastive runs differed from baselines ONLY through
  torch RNG stream perturbation (96 extra draws/rollout); InfoNCE conditions
  were effectively reseeded baselines. Same dead wiring exists in
  paper_exp_contrastive.py (lines 359, 423-431).
- Consequence: historical Exp-5 InfoNCE conclusions ("InfoNCE collapses the
  probe", CLAUDE.md) need re-attribution — the observed differences were
  seed noise, compounded by C5 (unseeded maps).
- Next: record in PLAN.md; flag for paper claim ledger.

### C9 — shared-condition steal cancellation
- Status: PLAUSIBLE_BUT_UNPROVEN
- Claim: with combat on, paired +1/-1 steal events cancel to zero in the
  team-mean reward — an unstated asymmetry between conditions (shared never
  "sees" theft). Moot under --no_combat; relevant to historical combat runs.
