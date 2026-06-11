# Mechanistic Verification — Cycle 1

Recipe: `/Users/relh/Code/tasha/metta/v3_experiments/paper_exp_reward_type.py`
Symptom: late-training return collapse (peak ~1.0-1.5 rolling mean, then 50-90% degradation before 5M steps; both reward conditions; healthy entropy; persists at near-zero LR).

## Claim Checks

### Claim C1 — adv-norm-on-noise
- **Verdict: SUPPORTED**
- **Mechanistic trace:**
  1. Advantages are computed exactly once per rollout, before the epoch loop. GAE at `paper_exp_reward_type.py:593-603` fills `adv_buf`/`ret_buf`; flattened once at lines 612-613 (`flat_adv = adv_buf.reshape(n_total)`). The epoch loop starts at line 624 (`for _ in range(epochs):`) and only ever *indexes* `flat_adv[mb]` (line 631). No GAE/value recompute between epochs. With defaults (`main()` lines 1049-1054: `num_envs=4`, `num_steps=128`, `minibatch_size=512`, `update_epochs=8`), that is 6144 samples → exactly 12 minibatches of 512 → **96 Adam steps per rollout on one fixed advantage vector**.
  2. Normalization at line 634:
     ```python
     mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
     ```
     Epsilon is added to the **std**, not variance, so for any pre-norm std `s >> 1e-8` the output std is `s/(s+1e-8) ≈ 1` exactly. There is no signal-magnitude floor: a minibatch whose advantages are pure critic-noise TD residuals (std perhaps 1e-3 to 1e-1) is rescaled to unit variance, i.e. amplified by `1/s` — gradient magnitude is **invariant to whether the minibatch contains any real reward signal**. (The only safe case, exact zeros → `0/1e-8 = 0`, never occurs because an imperfect critic always produces nonzero `γV_{t+1} - V_t` residuals.)
  3. Reward sparsity is structural, confirmed in the engine: the C++ reward buffer is zeroed at the top of every step (`packages/mettagrid/cpp/bindings/mettagrid_c.cpp:495-499`) and `Agent::compute_stat_rewards` writes only the **delta** of heart-count×1.0 (`packages/mettagrid/cpp/src/mettagrid/objects/agent.cpp:101-130`: `float reward_delta = new_stat_reward - this->current_stat_reward; if (reward_delta != 0.0f) *this->reward += reward_delta;`). Heart reward = 1 per unit (`packages/mettagrid/python/src/mettagrid/builder/envs.py:90-94`). So rewards are sparse ±1 impulses on heart-change steps only.
- **Sparsity arithmetic (individual condition, 12 agents, at peak):**
  - Episode = 1024 env-steps × 12 agents = 12,288 agent-steps; peak team return ~1.0-1.5 → roughly 1-3 net heart events, plus paired ±1 steal events → estimate **2-6 nonzero per-agent reward impulses per episode**.
  - A rollout covers 128 steps × 4 envs = 512 env-steps = half an episode's worth → expected impulses per rollout λ ≈ 1-3.
  - GAE smear: γλ = 0.99×0.95 = 0.9405, e-folding ≈ 17 steps, <5% influence beyond ~50 steps, confined to one (env, agent) trace and cut at `done`. Each impulse makes ~20-50 sample-advantages meaningfully reward-bearing → **~30-150 of 6144 samples (≤2.5%) carry any reward-derived signal**.
  - P(a given 512-sample minibatch contains none of m signal samples) ≈ (11/12)^m. Combining over Poisson event counts: P(rollout has zero impulses) = e^{-λ} ∈ [0.05, 0.37]; in one-event rollouts, P(minibatch misses all ~35 smeared samples) ≈ (11/12)^35 ≈ 0.05. **Overall, ~10-40% (central estimate ~25% at λ≈1-1.5) of minibatches contain zero reward-derived advantage content — and those that do carry it in only a handful of their 512 samples.** Every one of those noise minibatches is normalized to a full-strength unit-variance policy gradient, and the same noise direction is re-applied across 8 epochs (96 steps/rollout, no KL guard).
  - Shared condition: a steal (+1/-1 same step) has `rews.mean() == 0` exactly (transform at lines 283-286), so steals generate **no** training reward at all; harvest impulses are broadcast at 1/12 magnitude to all 12 agents. More samples touched, smaller magnitude — irrelevant post-normalization; the zero-event-rollout probability is comparable or higher.
- **Counterevidence:** none found in code. The exact runtime fraction and pre-norm std remain to be measured (instrumentation), but every structural element of the claim is present.
- **Refined mechanism:** per-minibatch normalization makes the policy-gradient step size independent of signal presence; with ≥95% of samples being pure critic noise, most of the 96 steps/rollout are noise steps of the same magnitude as signal steps.
- **Minimal fix:** normalize once per rollout over the full 6144-sample batch (move line 634 above the epoch loop onto `flat_adv`), ideally with an absolute scale floor, e.g. `flat_adv = (flat_adv - flat_adv.mean()) / max(flat_adv.std(), 1e-2)`, so reward-free rollouts are not amplified either.
- **Missing evidence:** runtime measurement of zero-event minibatch fraction and pre-norm adv std at/after peak (instrument one run; ~10 lines of logging at line 634).

### Claim C4 — trunc_bootstrap-shared-leak + self-reference
- **Verdict: SUPPORTED** (shared-condition leak); self-reference sub-claim refined to "standard TD bootstrapping, not an independent defect".
- **Mechanistic trace (ordering question resolved):**
  1. Inside `MettaGridVecEnv.step` (lines 259-309): raw per-agent rewards are read (line 280), then **the shared transform is applied inside the env step** at lines 283-286:
     ```python
     if reward_type == "shared":
         team_mean = rews.mean()
         rews = np.full_like(rews, team_mean)
     ```
     The returned `all_rewards` are therefore already team-mean-broadcast (uniform across the 12 agents).
  2. The `--trunc_bootstrap` patch runs in the trainer **after** `env.step` returns (lines 512-523). Line 522-523:
     ```python
     _, _, _, fo_vals, _, _ = policy.get_action_and_value(fo_t, agent_ids=fo_ids)
     rews[env_i] = rews[env_i] + cfg["gamma"] * fo_vals.cpu().numpy().reshape(n_agents)
     ```
     `fo_vals` is the critic evaluated on the 12 **per-agent** final observations (`final_obs` is `(n_agents, obs_dim)`, captured pre-reset at lines 298-301). Each agent's egocentric obs differs, so `fo_vals` is a 12-vector of distinct values. Adding it to the uniform shared `rews[env_i]` **breaks the uniformity invariant of the shared condition at every episode boundary** — and every episode ends by the wrapper's 1024-step truncation (line 294), so when the flag is on this fires for every episode.
  3. Propagation: the contaminated rewards land in `rew_buf[step]` (line 526) and enter **per-agent** GAE (lines 596-603). The differentiated boundary delta back-propagates through the λ-recursion across the preceding ~17-50 steps of the same episode, so per-agent-differentiated advantages and `ret` targets — exactly what the shared condition is designed to exclude — reach the PPO loss for a substantial window around every episode end. (Logged `episode_returns` are unaffected: accounting happens inside `env.step` before the patch.)
  4. Self-reference: at the truncation step, the patched delta is `r_t + γV(s_T) - V(s_t)` with `done=1` zeroing both the mask bootstrap and the recursion — mathematically identical to textbook truncation bootstrapping done via the mask. `ret = adv + V` then contains `γV(s_T)` in the value target (line 645, unclipped MSE). This is the inherent self-reference of TD bootstrapping; it can ratchet only via systematic overestimation (no value clip), which is a runtime question, not a code defect.
- **Counterevidence:** none for the leak. For the self-reference sub-claim, the construction is the standard correct fix in the individual condition.
- **Refined mechanism:** the bug is not the bootstrap itself but **where** it is applied: post-sharing, with per-agent values. It converts the "shared" arm into a hybrid condition (uniform rewards + per-agent value-shaped boundary signal), confounding the paper's central manipulation in any `--trunc_bootstrap` shared run.
- **Minimal fix:** preserve the invariant — in the shared condition add the team-mean bootstrap uniformly: `boot = fo_vals.cpu().numpy().reshape(n_agents); rews[env_i] += cfg["gamma"] * (boot.mean() if reward_type == "shared" else boot)`. (Equivalent to bootstrapping pre-sharing inside `env.step` and then applying the mean transform.)
- **Missing evidence:** runtime check of V mean inflation in `--trunc_bootstrap` runs (ratchet question); whether the A-3 individual-arm regression has a separate cause.

### Claim C5 — map-seed-not-wired
- **Verdict: SUPPORTED** (statically and empirically).
- **Mechanistic trace:**
  1. Recipe constructs configs with no MapGen seed: `make_arena` (`packages/mettagrid/python/src/mettagrid/builder/envs.py:57-75`) builds `MapGen.Config(num_agents=..., width=25, height=25, ...)` — no `seed` kwarg, so `seed: int | None = None` (`mapgen/mapgen.py:28-33`).
  2. Recipe constructs `Simulation(cfg, seed=seed*100+i)` at init (line 230) and `Simulation(cfg, seed=seed_offset+i)` with `seed_offset=global_step` on every reset (`_reset_sim`, lines 244-247) — **no `simulator=` parent in either call**.
  3. `Simulation.__init__` → `_make_map()` (`simulator/simulator.py:72-73, 264-268`) → `_seeded_map_builder` (lines 270-279), which returns the builder config **unchanged** on two independent grounds: line 273-274 (`if map_builder.seed is None: return map_builder`) and line 275-276 (`if self._simulator is None: return map_builder`). The `Simulator.next_map_seed` path (lines 330-336) is unreachable from this recipe.
  4. `MapGen.__init__` then seeds map RNG with `self.rng = np.random.default_rng(self.config.seed)` = `default_rng(None)` (`mapgen/mapgen.py:135`) → **OS entropy**; child scenes inherit it (`mapgen/scene.py:314`: `np.random.default_rng(self.config.seed or rng)`).
  5. The `seed` argument the recipe passes reaches only the C++ engine: `MettaGridCpp(c_cfg, map_grid, self._seed)` (`simulator/simulator.py:85`) — after the map is already built. So run seed and `seed_offset=global_step` affect in-episode engine RNG only, never map layout.
- **Empirical confirmation (run 2026-06-11, `.venv/bin/python`):** two `Simulation(make_arena(num_agents=12), seed=4200)` builds produced 62x37 maps that **differ** (`identical maps: False`; initial engine observations also differ), while setting `cfg.game.map_builder.seed = 777` explicitly produced **identical** maps across two builds (`identical maps: True`). This confirms the wiring is the only missing link — the seeding machinery works when a seed is supplied.
- **Counterevidence:** none.
- **Refined mechanism:** every episode of every run trains on a fresh OS-entropy map; "seeds" differ only in torch/numpy init and engine RNG. Per-run reproducibility is broken and cross-variant seed comparisons ("surviving seeds flip") compare different task streams — expected, not diagnostic.
- **Minimal fix:** set the map seed in the wrapper: in `MettaGridVecEnv.__init__`, `cfg.game.map_builder.seed = seed + i`; in `_reset_sim`, `cfg.game.map_builder.seed = self._base_seed + i + episode_counter` (a deterministic per-env episode counter, not `global_step`, which depends on training dynamics only via fixed increments here but is cleaner decoupled).
- **Missing evidence:** none for the claim itself.

### Claim C7 — infonce-gradient-dead
- **Verdict: SUPPORTED** (statically and empirically).
- **Mechanistic trace:**
  1. `emb_buf` is declared as a numpy array (line 454) and filled during rollout from detached, no-grad forward passes: lines 503-505 (`with torch.no_grad(): ... policy.get_action_and_value(...)`) → line 530 (`emb_buf[step] = embs.cpu().numpy()...`).
  2. In the minibatch loop, line 650: `emb_t = torch.tensor(emb_buf, dtype=torch.float32, device=device)` — `torch.tensor` from numpy creates a fresh leaf with `requires_grad=False` and no graph. `inter_agent_infonce_loss` (lines 120-157) performs only indexing, `F.normalize`, dot products, and `F.cross_entropy` on this dead tensor; its output has `grad_fn=None`.
  3. Line 658: `loss = loss + cfg["contrastive_coef"] * cl_val` — adds a constant; `loss.backward()` (line 661) receives zero gradient contribution from it. The `gamma_c` path (lines 135-138) only *samples* time offsets (`Geometric(...).sample()`, `torch.randint`) — index selection, no gradient re-attachment. The minibatch re-forward at line 638 discards its `emb` output (`_`), so no other path connects InfoNCE to parameters.
  4. **Empirical confirmation:** `cl_val.requires_grad == False`, `grad_fn is None`, and parameter gradients with vs. without the contrastive term are bit-identical (`torch.equal` over all parameters: True).
- **Counterevidence:** none.
- **Refined mechanism:** `--contrastive` is a no-op through the loss-gradient channel; its only run-affecting side effects are (a) the logged `total_cl`, and (b) **RNG-stream consumption** — `torch.randint(..., device=device)` (lines 131-133) advances the device generator and `Geometric.sample` (line 136) the CPU generator, 96 times per rollout, so subsequent `Categorical.sample` draws diverge from a non-contrastive run with the same seed. A `--contrastive` run is therefore a *seed perturbation* of baseline, nothing more.
- **Minimal fix:** compute the loss on gradient-carrying embeddings — e.g. re-forward the sampled `(t, e, agent)` anchor/positive/negative observations through `policy.encode`+`emb_head` inside the update (as the SupCon path at lines 677-693 already correctly does), or restructure to use the minibatch forward's live `emb` output.
- **Missing evidence:** none for this file.

## New Mechanisms Discovered
- **contrastive-as-seed-perturbation:** since the InfoNCE term is gradient-dead, the only mechanism by which `--contrastive` changes outcomes in this file is RNG-stream perturbation (and it does change the stream, 96 draws/rollout). Any "InfoNCE caused X" result produced by this file is a comparison of differently-seeded baselines.
- **same dead pattern in `paper_exp_contrastive.py`:** lines 329/359/423-431 of `/Users/relh/Code/tasha/metta/v3_experiments/paper_exp_contrastive.py` show the identical numpy-buffer → `torch.tensor` → loss wiring. The historical Exp-5 "InfoNCE collapses probe / EffRank rises" conclusions are mechanistically unexplainable by the contrastive gradient in EITHER file and need re-attribution (likely seed noise; the recipe's seeds don't even pin the maps, per C5).
- **shared-condition steal cancellation:** in `reward_type="shared"`, paired +1/-1 steal events average to exactly zero team-mean reward (lines 283-286), so combat theft is invisible to the shared-arm training signal — a previously unstated asymmetry between conditions beyond mean-broadcasting.

## Recommendations To Critic
- **Strongest:** C5 and C7 are closed — static trace plus direct empirical confirmation, no missing links. C4's shared-leak is a clean two-line code-order fact (transform at 283-286, patch at 523).
- **Strong but needing one runtime number:** C1 — every structural element verified (one-shot advantages, 96 reuse steps, eps-on-std normalization, delta-impulse rewards); the ~25% zero-signal-minibatch figure is an analytic estimate and should be confirmed by instrumenting line 634 before treating C1 as the root cause.
- **Should be narrowed:** C4's "self-reference ratchet" sub-claim — the construction is mathematically standard bootstrapping; keep only as a watch item (V-mean inflation) unless runtime evidence appears.
- **None of the four should die.**
