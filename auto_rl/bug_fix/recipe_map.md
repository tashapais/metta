# RL Recipe Map — v3_experiments/paper_exp_reward_type.py

## Symptom
Late-training return collapse. Runs peak (rolling 50-episode mean return ~1.0-1.5) mid-training, then degrade 50-90% before 5M agent-steps. Affects both reward conditions (individual per-agent, and shared team-mean broadcast). NOT entropy collapse: entropy stays 1.24-1.40 (max ln(5)=1.61), action diversity RISES, value loss stays small. Collapse persists even with LR annealed to ~zero. Surviving seeds flip between recipe variants (stochastic, seed-dominated).

## User Steering
- Focus: (1) per-minibatch advantage normalization; (2) SupCon/reward_cl second-optimizer path when disabled; (3) unclipped value loss / shared encoder; (4) mid-rollout reset + GAE done handling; (5) env reseeding with seed_offset=global_step; (6) --trunc_bootstrap patch x GAE; (7) MettaGridVecEnv wrapper alignment; (8) PPO ratio over stale minibatches (4-8 epochs).
- Ignore: entropy collapse in the abstract (ruled out); LR annealing / epoch count as root cause (ablated, insufficient); combat-vs-no-combat dynamics (ablated, contributing but not root).

## Top-Level Trace
1. Config: `main()` argparse → flat `cfg` dict → `train_one_seed(cfg, seed)` — `/Users/relh/Code/tasha/metta/v3_experiments/paper_exp_reward_type.py:945-1052`
2. Env: `MettaGridVecEnv` wrapping N=4 `Simulation` objects (mettagrid C++ engine) — recipe lines 220-316; `/Users/relh/Code/tasha/metta/packages/mettagrid/python/src/mettagrid/simulator/simulator.py:47-180`
3. Architecture: `ActorCritic` shared MLP encoder + 3 linear heads (actor/critic/emb) — recipe lines 323-371
4. Loss/update: inline PPO with GAE, per-minibatch advantage norm, unclipped value MSE, entropy bonus; optional InfoNCE term and optional separate SupCon optimizer step — recipe lines 593-694

## Component Inventory
| Component | Path | Role in training |
|-----------|------|------------------|
| `main()` / cfg dict | `paper_exp_reward_type.py:945-1052` | argparse → cfg; hardcodes num_envs=4, T=128, max_steps=1024, mb=512, clip 0.8-1.2, vf_coef 0.5, ent_coef 0.01, grad_clip 0.5, lr 3e-4, gamma 0.99, lambda 0.95 |
| `train_one_seed` | `paper_exp_reward_type.py:394-938` | full train loop: rollout → GAE → PPO → optional SupCon → metrics → probe |
| `MettaGridVecEnv` | `paper_exp_reward_type.py:220-316` | hand-rolled vec wrapper; reward transform per reward_type; episode accounting; reset/reseed |
| `MettaGridVecEnv.step` | `paper_exp_reward_type.py:259-309` | steps 4 sims, applies shared-mean transform, computes env-level done, captures pre-reset final obs, resets done sims in-place |
| `_reset_sim` | `paper_exp_reward_type.py:244-250` | closes sim, rebuilds `make_arena` config + new `Simulation(seed=global_step+i)` |
| `ActorCritic` | `paper_exp_reward_type.py:323-371` | shared 600→128→128 MLP (LayerNorm+ReLU) → actor(5), critic(1), emb(64); default PyTorch init; optional per-agent encoders |
| `compute_gae` | `paper_exp_reward_type.py:378-387` | textbook GAE, env-level done masks both bootstrap and λ-recursion |
| PPO minibatch loop | `paper_exp_reward_type.py:624-669` | 8 (or 4) epochs × 12 minibatches of 512 over 6144 samples (12 agents); adv normalized PER MINIBATCH at line 634 |
| trunc_bootstrap patch | `paper_exp_reward_type.py:512-523` | adds gamma·V_per_agent(final obs) to last reward; done stays 1 |
| `inter_agent_infonce_loss` | `paper_exp_reward_type.py:120-157` | only when `--contrastive`; computed from numpy `emb_buf` (see Hotspots: gradient-dead) |
| `supcon_rank_loss` + second optimizer step | `paper_exp_reward_type.py:164-213, 671-694` | only when `--reward_cl`; fully gated (collection at 574-578, update at 677) |
| Probe / metric collection | `paper_exp_reward_type.py:462-490, 534-578, 696-717, 760-899` | read-only wrt parameters (plus `torch.save` checkpoint I/O when `--corrected_probe`) |
| `Simulation` | `simulator.py:47-180` | builds map (`_make_map`), C++ engine `MettaGridCpp(c_cfg, map_grid, seed)`; `step()` advances engine |
| `_seeded_map_builder` | `simulator.py:270-279` | reseeds map builder ONLY via a `Simulator` parent; recipe has none → returned unchanged |
| `make_arena` | `packages/mettagrid/python/src/mettagrid/builder/envs.py:26-98` | heart inventory reward = 1; combat toggles attack laser cost; MapGen 25x25 instances of 6-agent rooms, `seed=None` |
| C++ step / reward / done | `packages/mettagrid/cpp/bindings/mettagrid_c.cpp:496-603`; `cpp/src/mettagrid/objects/agent.cpp:101-130` | rewards zeroed each step; stat-reward DELTA (can be negative on heart loss); terminals/truncations only set at engine max_steps (10000, never reached) |

## Verified Environment Facts (live-checked)
- Observations: `(n_agents, 200, 3)` uint8, flattened to obs_dim=600. Raw values 0-255 cast to float32 with **no normalization**; 81% of bytes are 0xFF padding. Token format (location, feature_id, value); token-slot semantics shift with the visible-object set.
- Action masks: `_c_sim.masks()` is `(n_agents, 5)` bool and exists, but the recipe **never applies it** — `Categorical(logits)` samples unmasked; n_actions=5 is read off the mask shape only.
- Rewards: per-step DELTA of stat reward (heart count × 1.0). Negative rewards occur when an agent loses hearts (combat steal). `engine rewards` buffer is zeroed at the top of every C++ step.
- Done semantics: C++ sets terminals/truncations **only** at its own `max_steps=10000` (with `episode_truncates=False` default it would set *terminals*); the wrapper truncates at `episode_steps >= 1024` first, every time. Therefore in this recipe: `terminated` is always False, every episode end is a wrapper-side time-limit truncation, and all 4 envs truncate **simultaneously** (synchronized 1024-step episodes, episode boundary every 8th rollout since T=128).
- Map generation: `MapGen.Config.seed = None` and `Simulation._simulator is None` → `np.random.default_rng(None)` → **every reset builds a brand-new OS-entropy random map**. Verified live: two consecutive builds differ. The `seed_offset=global_step` passed to `_reset_sim` seeds only the C++ in-episode RNG, NOT map layout. The run seed does not pin the environment.
- Map structure: 12 agents → 62x37 grid = two 25x25 instances of 6-agent rooms. "shared" team-mean averages across rooms that cannot interact.

## Data Flow
1. **Rollout** (recipe 493-580): for each of T=128 steps — `obs_buf[t] = obs` (raw uint8-as-float); policy forward under `no_grad` → actions, logp, V, emb, logits all cached to numpy buffers; `env.step(actions, global_step, reward_type)` steps all 4 sims.
2. **Inside env.step**: per sim — write actions into C buffer, `sim.step()`, read per-agent reward delta, accumulate `episode_true_returns` (pre-sharing), then if `reward_type=="shared"` overwrite rewards with team mean; accumulate `episode_returns` (post-sharing). Done = any(terminals) [inert] or any(truncations) [inert] or step-count ≥ 1024. On done: capture pre-reset final obs (for trunc_bootstrap), emit (team_total, per-agent post-sharing returns, true returns), `_reset_sim(i, seed_offset=global_step)`. Returns `get_obs()` — for done envs this is the **new episode's first obs**, stored as the next step's obs (correct because done masks the t→t+1 bootstrap in GAE).
3. **trunc_bootstrap** (recipe 512-523, flag-gated): for each truncated env, `rews[env_i] += gamma * V_per_agent(final_obs)` before `rew_buf[t] = rews`; `done_buf[t]` stays 1. Since every episode ends by time limit, this fires for every episode when enabled.
4. **Buffers**: obs (T,E,n,600), act/rew/logp/val (T,E,n), done (T,E) env-level, emb (T,E,n,64), logit (T,E,n,5). `global_step += E*n_agents` per env-step (counts agent-steps; 6144/rollout at 12 agents).
5. **Probe collection** (all runs): per finished episode, terminal-step embeddings + argsort rank labels appended to `probe_embeddings/labels` (read-only). `corrected_probe`/`probe_at_peak` add quality tracking + `torch.save` best checkpoint (I/O only). `episode_buffer` for SupCon is appended **only when `use_reward_cl`** (line 574).

## Gradient / Update Flow
- **GAE** (recipe 593-603 + 378-387): per-agent loop a=0..n-1, vectorized over envs. `delta = r_t + gamma·V_{t+1}·(1-d_t) - V_t`; both the bootstrap and the λ-recursion are zeroed at done. `next_value` from the post-rollout obs (masked when done at T-1 — correct). Without `--trunc_bootstrap`, **every** episode end gets a zero bootstrap despite being a pure time-limit truncation. Returns target `ret = adv + V`.
- **PPO** (recipe 615-669): flatten to 6144 samples; optional linear LR anneal; shuffle once per epoch; 12 minibatches × 8 epochs = **96 optimizer steps per rollout** on the same data with advantages computed once (stale across epochs, no KL early-stop, no recompute). Per-minibatch: `mb_adv = (mb_adv - mean)/(std + 1e-8)` (line 634) → fresh forward → `ratio = exp(new_logp - old_logp)` → clip surrogate `max(-adv·r, -adv·clamp(r, 0.8, 1.2))` → `vl = F.mse_loss(new_val, mb_ret)` (unclipped, vf_coef 0.5, gradients flow through the **shared encoder**) → `- 0.01·entropy` → backward → `clip_grad_norm_(0.5)` → Adam(lr=3e-4, eps=1e-5) step.
- **InfoNCE branch** (`use_cl`, line 648-658): `emb_t = torch.tensor(emb_buf)` — built from the **numpy** rollout buffer, no `requires_grad`, no graph to policy parameters. The term is added to `loss` but its gradient contribution is exactly zero (see Hotspots). OFF in the failing runs.
- **SupCon second step** (`use_reward_cl`, lines 671-694): re-forwards stored terminal obs **with** gradients through encoder + emb_head, separate `zero_grad/backward/step`. Fully gated: with the flag off, `episode_buffer` is never appended (574) and the block never runs (677). **Confirmed: no SupCon/reward_cl machinery executes in the failing runs.**
- **Detach boundaries**: all rollout quantities (logp, V, emb, logits) are produced under `no_grad` and stored as numpy — old-policy quantities are correctly constant. Only the minibatch re-forward carries gradients.
- **Optimizer**: single Adam over all parameters (encoder + 3 heads). LR anneal mutates param_group lr once per rollout. Adam moments persist across anneal.

## High-Risk Interfaces
- **Per-minibatch advantage normalization × extreme reward sparsity** (line 634): team return at peak is ~1.0-1.5 per 12,288 agent-step episode, i.e. roughly one reward event per ~10⁴ samples. A 512-sample minibatch frequently contains zero reward events; its advantages are then pure GAE-propagated value noise, which mean-centering + unit-std rescaling promotes to full-strength policy gradient. Because advantages are computed once and reused for 8 epochs, the noise direction is applied ~8 times coherently per rollout rather than averaging out. Mean-centering also forces ~half of every minibatch to negative advantage even when all sampled actions are good — systematic pressure away from the current (good) policy. Consistent with every observed symptom: healthy entropy, rising action diversity, small value loss, seed-dominated divergence, degradation from a peak.
- **Universal time-limit truncation treated as terminal** (default path): every episode ends at the wrapper's 1024-step cap with `done=1` and zero bootstrap. V must predict a horizon-truncated return; obs does include an `episode_completion_pct` global token, so this is learnable, but the bias grows with policy quality exactly as the `--trunc_bootstrap` help text states.
- **`--trunc_bootstrap` patch semantics** (512-523): adds **per-agent** V(s_T) to the last reward while done stays 1. (a) In the *shared* condition this injects differentiated per-agent signal into a reward stream whose defining invariant is identity across agents — condition contamination at every episode boundary. (b) V appears inside its own MSE target (`ret = adv + V` with patched reward), an unclipped self-referential loop; with value overestimation this can ratchet.
- **Map nondeterminism / broken seeding**: `MapGen.seed=None` + no `Simulator` parent means `seed`, `seed*100+i`, and `seed_offset=global_step` control **nothing about map layout**; every episode is a fresh OS-entropy map. The task is implicit procedural generalization; per-seed reproducibility is broken, which inflates seed-dominance and makes "surviving seeds flip between variants" expected rather than diagnostic.
- **Unclipped value loss through shared encoder** (645): no value clipping, vf_coef 0.5, critic gradient shapes the actor's features. Value loss magnitude is small in absolute terms only because returns are ≤~1.5.
- **8-epoch reuse with no KL guard** (624): ratio clipping bounds per-sample updates but nothing bounds cumulative policy movement per rollout; combined with re-normalized noise advantages this is the amplification stage.
- **Raw uint8 observations, no normalization**: inputs 0-255 (81% padding=255) into the first Linear; LayerNorm normalizes activations but first-layer gradient scale is input-proportional. Constant across training, so unlikely the late-collapse trigger by itself.
- **Env-level done from `any(terminals)`** (293): currently inert (terminals never fire pre-1024), but a latent landmine — a single agent terminal would reset the whole env for all agents.
- **Action masks ignored**: invalid actions are sampled and presumably no-op in the engine; logp/ratio bookkeeping is still self-consistent, so this is inefficiency, not misalignment.

## Hotspots
- **adv-norm-on-noise**: line 634 per-minibatch normalization with sparse rewards; quantify fraction of minibatches with zero reward events at peak and the pre-normalization adv std. Primary suspect for stochastic late collapse with healthy entropy and rising action diversity.
- **coherent-noise-reuse**: lines 624-669, advantages computed once, reused 8 epochs (96 steps/rollout); noise directions applied coherently, no KL early stop.
- **zero-bootstrap-at-truncation**: lines 293-296 + 378-387 default path; every done is a time limit treated as terminal.
- **trunc_bootstrap-self-reference + shared-condition leak**: lines 512-523; per-agent V added to shared rewards; V in its own unclipped target.
- **map-seed-not-wired**: `simulator.py:270-279` + `mapgen.py:28` — seeds never reach map generation; reproducibility and seed-dominated outcomes.
- **value-grad-through-shared-trunk**: line 645-646, unclipped MSE × 0.5 into the actor's encoder.
- **infonce-gradient-dead (latent)**: lines 648-658, `torch.tensor(emb_buf)` has no graph — when `--contrastive` is on the loss term is a constant wrt parameters. Off in failing runs, but it confounds any past "InfoNCE caused X" conclusions drawn from THIS file (paper_exp_contrastive.py may differ).
- **negative-reward steal dynamics**: `agent.cpp:101-130` reward delta goes negative on heart loss; with combat, individual condition rewards theft (zero-sum transfers) — contributes nonstationary opponent pressure (user: contributing, not root).

## Open Questions
- What fraction of 512-sample minibatches contain zero nonzero-reward transitions at/after the return peak, and what is the pre-normalization advantage std there? (Direct test of the adv-norm hotspot; measurable by instrumenting one run or replaying from wandb-era checkpoints.)
- Did the original failing runs (e.g. bswht815) use update_epochs=8 or 4, and anneal on/off? Both reportedly fail, but the amplification factor differs 2x.
- How stochastic is the C++ engine within an episode given a fixed map (does `seed=global_step+i` matter at all in practice)?
- Magnitude of the zero-bootstrap value bias near episode end at peak policy (V(s) vs realized truncated return in the last ~100 steps) — distinguishes the truncation hotspot from the adv-norm hotspot.
- Do per-agent true returns show a rising negative-reward (theft) component at collapse onset in individual-condition runs?
- Does logit norm / actor weight norm grow monotonically through collapse (drift signature) even as entropy stays flat?
- For `--trunc_bootstrap` runs: does the value function's mean prediction inflate over training (self-reference ratchet)?

## Notes On Ignored Areas
- Entropy collapse: not re-examined; recipe wiring of the entropy bonus (sign, coefficient 0.01) verified correct in passing.
- LR anneal / epoch count: mapped as amplification factors (frac formula line 617, epochs CLI), not pursued as root cause per steering.
- Combat dynamics: mapped only where they touch the reward interface (negative deltas, steal-induced nonstationarity); no deeper game-dynamics analysis.
- SupCon/reward_cl and InfoNCE: confirmed fully gated off in failing runs (collection gate line 574, update gate line 677, loss gate line 649); flagged the latent InfoNCE dead-gradient issue but did not investigate its historical impact.
