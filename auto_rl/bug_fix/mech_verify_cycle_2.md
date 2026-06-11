# Mechanistic Verification — Cycle 2

Inputs: recipe `v3_experiments/paper_exp_reward_type.py` (line numbers below refer to the current
file, which has shifted ~20 lines from the cycle-0 map after the A-4 fixes), recipe map, claim
ledger, dispatch claims C10/C6/C2. Empirical checks run with
`.venv/bin/python /tmp/mech_c2_c10_check.py` (single env build + 1 real rollout + 4-epoch PPO
update on CPU, 3.5 s total). Phase A-4 context: seed_maps, batch_adv_norm, corrected (team-mean)
trunc_bootstrap, anneal_lr, epochs=4, no_combat — still collapsing, softer (final/p90 ind
0.24/0.54/0.25, shr 0.55/0.60), entropy healthy.

## Claim Checks

### Claim C10 — obs-unnormalized
- Verdict: REFUTED (as a late-collapse mechanism). The factual substrate is fully CONFIRMED and a
  one-line hygiene fix is still recommended.
- Mechanistic trace: `MettaGridVecEnv.get_obs()` (recipe 271-276) returns
  `sim._c_sim.observations().copy()` reshaped — **uint8**, shape `(4, 12, 600)` from raw
  `(12, 200, 3)` token buffers. C++ padding constant:
  `packages/mettagrid/cpp/include/mettagrid/objects/constants.hpp:12`
  `const uint8_t EmptyTokenByte = 0xff;`. Consumption path: `obs_buf[step] = obs` (line 518,
  implicit cast into the float32 buffer declared line 471) and
  `torch.tensor(obs.reshape(...), dtype=torch.float32)` (519-521). Grep confirms **no `/255`, no
  obs normalization anywhere** in the file (only `F.normalize` on embeddings in the contrastive
  losses). Encoder input stage (358-362): `Linear(600,128) → LayerNorm(128) → ReLU → ...`.
- Evidence (measured, fresh env, default init):
  - dtype uint8, min=0 max=255 **mean=209.7**, frac==255 = **0.795**, frac==0 = 0.017.
  - Per-agent padding fraction varies 0.555-0.955 (std 0.105) with visible-object count.
  - First-Linear pre-LayerNorm activations on raw obs: std **131.9**, absmax 434 (vs std 0.52
    with obs/255). So yes, the first layer operates at a scale ~250x conventional.
- Counterevidence (why the causal link to LATE collapse fails):
  1. **LayerNorm sits immediately after the first Linear** (line 360): forward pass is exactly
     per-sample normalized; the 130-sigma pre-activations never propagate. It also makes layer 1
     scale-invariant, so the "large effective LR on input layer" sub-claim inverts: under Adam,
     scale-invariant layers' weight norms grow and effective LR on layer 1 *decays* over training
     (slow plasticity loss, not collapse).
  2. **Adam is per-coordinate scale-free**: gradient asymmetry into W1 columns measured at only
     ~2.0x (padding dims 1.45e1 vs informative dims 7.4e0 mean |grad|), and Adam's second moment
     normalizes per-coordinate magnitude away regardless.
  3. **The input distribution is stationary across training** (seeded i.i.d. maps); per-sample
     padding variation is absorbed by the per-sample LayerNorm statistics. A constant,
     stationary factor cannot be the time-dependent trigger of a peak-then-decay symptom.
- Refined mechanism: raw 0-255 inputs are an inefficiency (signal dims modulated by per-sample
  padding statistics inside LayerNorm), not a collapse driver in THIS architecture.
- Recommended hygiene fix (cheap, flag-gated `--norm_obs`): `obs.astype(np.float32) / 255.0` at
  the two tensor-construction sites (519-521, 611-613, 663, plus trunc-bootstrap 541). Do not
  expect it to change the collapse.
- Missing evidence: none material; runtime W1-norm trajectory would only inform the (secondary)
  plasticity-loss note.

### Claim C6 — value-grad-through-shared-trunk
- Verdict: SUPPORTED (mechanism present and quantitatively live; whether it dominates at collapse
  onset still needs runtime logging).
- Mechanistic trace: minibatch loss at 681-682: `vl = F.mse_loss(new_val, mb_ret)`,
  `loss = pg + 0.5 * vl - 0.01 * ent.mean()`. Grep confirms the only clamps in the file are the
  ratio clamp (679) and grad-norm clips (711, 741): **no PPO value clipping exists**. `new_val`
  comes from `critic(h)` where `h = encoder(obs)` (367-382) — same trunk as actor and emb_head;
  one Adam over all parameters (460). Critic targets `ret = adv + V` (406).
- Evidence (measured at init, real rollout, batch-normed advantages, first 512-minibatch):
  - Trunk gradient norms by term: `0.5*vl` → **1.34**, `pg` → **0.75**, `-0.01*ent` → 0.012.
    The value term is the **largest** gradient pressure on the shared trunk even though
    `vl = 0.0147` looks tiny. "Small value loss" (the symptom report) does NOT imply small trunk
    pressure — pressure scales with residual structure, not loss magnitude.
  - Return-target scale confirmed small: mean 0.48, absmax 0.79 at init (peak-policy returns
    ≤ ~1.5), so absolute explosion is not the mode; *direction capture* of the clipped (0.5)
    gradient is.
  - Coupling: in zero-reward-event rollouts (see N1) `ret` is purely the critic's own bootstrap
    chain — the value term then trains the trunk to fit its own noise, through an unclipped MSE.
- Counterevidence: pg and vl gradients are same order (factor ~2); at peak policy the critic fits
  well (value loss small and stable per symptom), so late-training dominance is plausible but
  unproven without per-term grad-norm logging.
- Refined mechanism: not "value explosion" — rather, with sparse/no events the 0.5·MSE term
  supplies the majority of the trunk's update direction, and in noise rollouts that direction is
  self-referential.
- Remedy assessment: (a) `vf_coef 0.5 → 0.25`: one character, lowest risk, standard. (b) PPO
  clipped value loss: standard in cleanrl/baselines but literature-neutral (Engstrom et al. 2020;
  Andrychowicz et al. 2021 find it neutral-to-harmful) — fine flag-gated. (c) Detached/stop-grad
  critic trunk: **too invasive for this experiment** — the paper probes emb_head(h) of the shared
  trunk; removing critic gradients changes the representation-formation process under study, and
  with rewards this sparse the trunk would receive almost no learning signal from pg alone.
- Missing evidence: per-term trunk grad-norm logging around collapse onset (one wandb scalar per
  term per rollout).

### Claim C2 — coherent reuse without KL guard
- Verdict: SUPPORTED.
- Mechanistic trace: update loop 659-712: `for _ in range(epochs)` × `range(0, n_total, mb_size)`
  → 4 × 12 = 48 Adam steps per rollout on advantages computed once (GAE at 621-631, before the
  loop). Grep for `kl`/`approx`/`target` over the whole file: the only "KL" is the
  action-diversity *metric* docstring (line 89). **No approx-KL is computed and no early stop
  exists.** The only bounds on movement: per-sample ratio clamp(0.8, 1.2) (679) and
  grad-norm clip 0.5 (711) — neither bounds cumulative policy displacement across 48 steps.
- Evidence (measured: real rollout, init policy, A-4 batch_adv_norm path, 4 epochs):

  | epoch | approx_kl | clipfrac | ratio_max | ratio_min |
  |-------|-----------|----------|-----------|-----------|
  | 0     | 0.000     | 0.000    | 1.000     | 1.000     |
  | 1     | 0.0020    | 0.001    | 1.244     | 0.811     |
  | 2     | 0.0120    | 0.210    | 1.412     | 0.756     |
  | 3     | 0.0085    | 0.118    | 1.316     | 0.695     |
  | 4     | 0.0045    | 0.018    | 1.549     | 0.705     |

  Realized ratios blow through the [0.8, 1.2] clip to [0.70, 1.55]: clipping only zeroes a
  sample's own gradient after it has passed the boundary, while every other minibatch keeps
  moving the shared parameters. And this drift was produced by **pure noise advantages** — the
  rollout contained 0/6144 nonzero rewards (see N1). ~0.005-0.012 KL of noise-directed movement
  per rollout, ~800 rollouts of 5M steps → large integrated drift even at annealed LR.
- Counterevidence: none in code. Note KL is non-monotonic within the update (noise directions
  partially cancel), so per-rollout movement is bounded in practice to ~1e-2 — the harm is the
  integral, which only an early stop or a noise gate reduces.
- Refined mechanism: C2 is the amplifier of N1 (and previously C1), not a standalone cause.
- Remedy + risk: cleanrl-style stop — compute
  `approx_kl = ((ratio - 1) - logratio).mean()` per minibatch, break both loops when
  `> target_kl (0.02)`. ~6 lines, LOW risk. One side effect: early exit consumes fewer
  `np.random.shuffle` draws → perturbs seed comparability across variants (same caveat already
  accepted for C8/A-4 changes). Note: with target_kl=0.02 alone, noise rollouts would still
  spend their full KL budget on noise — pair it with the N1 gate.

## New Mechanisms Discovered

### N1 — zero-event-rollout renormalization (batch_adv_norm does NOT close C1)
The A-4 fix moved normalization from minibatch to rollout granularity (648-654), but the same
pathology reappears one level up. Measured on a real rollout: **0/6144 transitions had nonzero
reward**; raw advantages were pure critic noise (std 0.081, mean -0.086) and line 654 rescaled
them to exactly unit std → 48 coherent Adam steps of full-strength noise gradient (the C2 table
above IS this scenario). Rate estimate at peak policy: team return ~1.0-1.5 per 1024-step
episode → λ ≈ 4 envs × ~1.25 events / 8 rollouts ≈ 0.6 events/rollout → **P(zero-event rollout)
≈ e^-0.6 ≈ 55%** even at peak (and 7 of 8 rollouts contain no episode boundary, so
trunc_bootstrap events don't help those). This precisely predicts A-4's outcome: softer (the
per-minibatch lottery inside every rollout is gone) but still failing (more than half of all
rollouts are still pure-noise updates). Candidate fixes, in order of preference: (a) noise gate —
skip the pg term (or the whole update) when raw `flat_adv.std()` is below a floor tied to the
reward scale; (b) normalize by a running cross-rollout std (EMA) instead of the per-rollout std;
(c) target_kl stop (caps but does not zero the noise). Note (N3) that merely *downweighting*
noise advantages is blunted by Adam.

### N2 — emb_head frozen in all baseline runs
Verified: after `loss.backward()` on the full PPO loss, `policy.emb_head.weight.grad is None` —
the PPO forward discards `emb` (674), so with `--contrastive`/`--reward_cl` off the emb_head
never receives a gradient and stays at random init for the entire run. Benign for returns
(no path to the loss), but every probe accuracy and emb-space geometric metric (eff_rank,
svd_ratio, expansion) in baseline conditions measures the trunk through a fixed random 128→64
projection. Worth recording for the paper-claims ledger; not a collapse mechanism.

### N3 — Adam noise-floor drift under annealed LR
Adam's per-coordinate step is ~lr·sign-like regardless of gradient *magnitude* (m/√v
self-normalizes). Consequences: (i) tiny noise gradients still move every parameter at ~full lr
step size — SGD would self-attenuate, Adam does not; (ii) LR annealing reduces drift only
linearly, and collapse onsets mid-training where lr is still ~1.5e-4 with 48 steps/rollout;
(iii) fixes that merely *rescale* noise advantages down (std floors as scaling) are partially
defeated by Adam — gating/skipping (N1 fix a) or KL stop (C2) act on direction/step-count and
are robust.

### N4 — synchronized truncation heterogeneity (minor)
Episodes are exactly 1024 env steps and rollouts 128, in lockstep from construction → all 4 envs
truncate simultaneously at t=127 of every 8th rollout. With trunc_bootstrap, that one rollout in
8 carries V-scale reward events on its final step for all 48 agent-streams, while the other 7
are reward-sparse. A single per-rollout normalization therefore alternates between two
systematically different advantage distributions. Secondary; subsumed by the N1 gate.

### Scanned and benign
- `recent_returns` unbounded list (483, 560): ~1.6k floats per run, read only via `[-100:]`/
  `[-200:]` — memory-only, no side effects.
- Geometric-metrics block (745-766): builds fresh CPU tensors from numpy buffers; all four
  metric functions are read-only, consume no RNG, mutate nothing.
- LR anneal (644-647): `frac = max(0, 1 - global_step/total)` correct, mutates only
  `param_groups[].lr`; Adam moments persist (intended).
- Checkpoint save / probe collection: I/O and list appends only.

## Recommendations To Critic
- Strongest: **N1 + C2 as a pair** — mechanistically verified end-to-end on a real rollout, and
  uniquely predicts the A-4 signature "softer but still failing with healthy entropy." Promote
  N1 to a first-class claim; C1 should be re-opened as NOT-FIXED-BY-A-4 (fix addressed minibatch
  granularity only).
- C6 is live (dominant trunk gradient at init, no value clip) but plausibly secondary; bundle the
  cheap dial (vf_coef 0.25, optional flag-gated value clip) and add per-term trunk grad-norm
  logging rather than restructuring the network. Reject detached-critic-trunk as
  experiment-invalidating.
- C10 should die as a root-cause claim: stationary factor, defused by LayerNorm placement and
  Adam; keep `obs/255` as flag-gated hygiene only.
- Decisive instrumentation for the next run (one wandb scalar each, ~10 lines): per-rollout raw
  adv std, nonzero-reward-event count, post-update full-batch approx_kl, per-term trunk grad
  norms. These four series at collapse onset cleanly separate N1/C2 from C6.
