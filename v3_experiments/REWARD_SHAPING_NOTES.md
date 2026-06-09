# Reward Shaping Notes for Tribal Village

These notes track the reward-design decisions behind the canonical reward
geometry rerun so later paper edits can cite the reasoning without mixing pilot
details into the Overleaf source.

## Source Log

- Ng, Harada, and Russell, "Policy Invariance Under Reward Transformations:
  Theory and Application to Reward Shaping"
  (`https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf`).
  Main takeaway: potential-based reward shaping is the safe default when we add
  dense progress information, because it uses `gamma * Phi(s') - Phi(s)` rather
  than independent bonuses or penalties that can change the optimal policy.
- Devlin and Kudenko, "Dynamic Potential-Based Reward Shaping"
  (`https://www.ifaamas.org/Proceedings/aamas2012/papers/2C_3.pdf`).
  Main takeaway: potential-based shaping can be extended beyond the simplest
  single-agent case, but the assumptions have to be handled explicitly.
- Toro Icarte et al., "Reward Machines: Exploiting Reward Function Structure in
  Reinforcement Learning" (`https://arxiv.org/abs/2010.03950`).
  Main takeaway: long sequential tasks should expose task structure instead of
  forcing the learner to infer a hidden chain from rare scalar events.
- Andrychowicz et al., "Hindsight Experience Replay"
  (`https://arxiv.org/abs/1707.01495`) and Nair et al., "Overcoming Exploration
  in Reinforcement Learning with Demonstrations"
  (`https://arxiv.org/abs/1709.10089`).
  Main takeaway: sparse long-horizon tasks often need curriculum, relabeling,
  demonstrations, or warm starts; simply making penalties larger is usually not
  the right exploration fix.
- Trott et al., "Keeping Your Distance: Solving Sparse Reward Tasks Using
  Self-Balancing Shaped Rewards" (`https://arxiv.org/abs/1911.01417`).
  Main takeaway: naive dense distance rewards can become the task and trap
  learning in local optima, so shaping should be bounded, diagnostic, and
  validated against the original task behavior.

## Experiment Log

- `event_v1` was a sparse event-only reward. It did not get discovered by PPO in
  1M-step Stage-1 runs even though random/use-sweep baselines could trigger
  resource and craft events.
- `event_v2` and later breadcrumbs made exploration more active, but early
  designs still collapsed or chased easy interactions instead of the heart
  chain.
- `event_v7_chain_compass_breadcrumbs` was the first clear positive signal:
  adding the chain-compass observation plus chain-only rewards produced
  deterministic and stochastic checkpoint rollouts with nonzero heart deposits
  for all three seeds.
- `event_v8_clean_chain_compass_breadcrumbs` added explicit negative
  coefficients for off-chain successful events. It failed the promotion gate:
  seed 1 lost heart deposits, and off-chain behavior was not reliably reduced.
- `event_v9_potential_chain_compass_breadcrumbs` removed the explicit negative
  event coefficients and replaced clipped distance rewards with a bounded
  potential-difference term. It restored meaningful heart-chain behavior in all
  three seeds by the weak criterion that each seed had nonzero deterministic or
  stochastic deposits, but it did not solve cleanup: seed 0 remained weak and
  off-chain resource, craft, handoff, and combat events were still substantial.

## Current V9 Rule

For v9, do not add more independent negative reward terms. Use:

- positive terminal/success events for the ore -> battery -> heart chain;
- the v7 chain-compass observation so the target is observable;
- small potential-based chain progress, recorded as
  `F(s,s') = gamma * Phi(s') - Phi(s)`;
- positive oracle-action breadcrumbs only as a diagnostic affordance scaffold;
- behavior-gate evaluation in the full Tribal Village world with distractors
  restored.

The potential-difference term can be signed by construction, but it is not a
new hand-authored punishment for off-chain behavior. It is a bounded progress
term whose paper status depends on the behavior gate.

## V9 Stage-1 Outcome

Run date: 2026-06-08.

- Commit: `d4708a0b8a`.
- Stage-1 root:
  `/workspace/tribal_event_mask_runs/stage1_v9_potential_chain_compass_1m`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage1_v9_potential_chain_compass_d4708a0b8a`.
- Result JSON validation passed for all three seeds.
- Behavior-output validation passed for 9 files on `relh-sandbox-1` and 2 files
  on `relh-sandbox-2`.

Headline behavior over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Resources | Crafts | Deposits | Combat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `-5.767` | `42.3` | `53` | `49` | `25` | `0` |
| `random` | `-35.467` | `42.3` | `111` | `5` | `0` | `0` |
| `seed0_deterministic` | `-41.883` | `139.3` | `335` | `8` | `0` | `67` |
| `seed0_stochastic` | `-33.763` | `169.0` | `408` | `37` | `2` | `43` |
| `seed1_deterministic` | `7.967` | `259.3` | `347` | `145` | `48` | `17` |
| `seed1_stochastic` | `6.167` | `201.7` | `326` | `129` | `49` | `0` |
| `seed2_deterministic` | `-2.367` | `114.7` | `182` | `82` | `37` | `2` |
| `seed2_stochastic` | `-13.900` | `262.3` | `506` | `125` | `30` | `22` |

Decision:

- V9 is a better reward-shaping direction than v8 because it avoids explicit
  negative event penalties and recovers deposits for all seeds under at least
  one rollout mode.
- V9 is not ready as the final paper condition because seed 0 is weak and
  off-chain activity remains large.
- The next useful step is not stronger reward penalties. It should be a
  curriculum, affordance mask, chain-only start distribution, or short oracle
  warm start that makes the heart chain cleaner without punishing exploratory
  off-chain successes.

## Current V10 Cleanup Ramp

`event_v10_chain_affordance_compass_breadcrumbs` keeps the v9 reward exactly in
spirit:

- no explicit negative reward coefficients;
- positive ore, battery, and heart events;
- bounded potential-difference chain progress;
- chain-compass observation.

The new intervention is not reward shaping. It is a curriculum/affordance mask
that lets the policy move freely but only use the current chain target. In
practice that disables off-chain successful `use`, `put`, `attack`, `plant`,
and `swap` actions during this cleanup ramp. Checkpoint rollouts use the
effective mask from the checkpoint config or, for legacy v10 checkpoints, infer
it from `reward_design`. V10 should be read as a constrained
affordance/curriculum condition, not as an unconstrained final paper condition.

Promotion question:

- Does masking off-chain affordances keep v9's seed-1/seed-2 deposit signal
  while rescuing seed 0 and reducing off-chain resource/craft/combat counts?

## V10 First Ramp Outcome

Run date: 2026-06-08.

- Commit: `6b2442b2b4`.
- Stage-1 root:
  `/workspace/tribal_event_mask_runs/stage1_v10_chain_affordance_compass_1m`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage1_v10_chain_affordance_compass_6b2442b2b4`.
- Result JSON validation passed for all three seeds.
- Behavior-output validation passed for 4 checkpoint files on
  `relh-sandbox-1` and 7 baseline/seed-2 files on `relh-sandbox-2`.

Training eval showed positive shaped returns but negative raw env returns:

| Seed | Eval raw return | Eval shaped/individual return | `D_act_JS` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `-1.483` | `173.838` | `0.474` | `0.245` | `0.345` |
| `1` | `-1.819` | `144.540` | `0.463` | `0.215` | `0.333` |
| `2` | `-7.994` | `114.768` | `0.496` | `0.171` | `0.227` |

Behavior gate over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Resources | Crafts | Deposits | Combat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `8.000` | `74.7` | `91` | `81` | `52` | `0` |
| `random` | `-28.800` | `42.7` | `115` | `6` | `0` | `0` |
| `seed0_det` | `-30.467` | `112.0` | `106` | `5` | `0` | `0` |
| `seed0_stoch` | `0.600` | `242.0` | `226` | `96` | `42` | `0` |
| `seed1_det` | `-28.800` | `20.3` | `36` | `24` | `0` | `1` |
| `seed1_stoch` | `-41.697` | `112.3` | `261` | `16` | `0` | `5` |
| `seed2_det` | `-28.800` | `11.7` | `26` | `9` | `0` | `0` |
| `seed2_stoch` | `-28.800` | `23.7` | `52` | `10` | `0` | `0` |

Decision:

- The first v10 ramp is useful but not promotable. It produced one strong
  learned chain mode (`seed0_stoch`) and showed that the oracle can complete
  the chain under the gate, but learned heart deposits were not robust across
  seeds or deterministic/stochastic modes.
- The gate exposed a cleanup bug: when an agent had no valid chain move/use,
  the chain-affordance mask restored the full native action mask. That reopened
  noop and sometimes off-chain `put`, `attack`, and `plant` actions.
- The follow-up is a strict fallback fix: no valid chain move/use should fall
  back to noop-only, not to the full native action surface. Then rerun Stage 1
  before deciding whether to anneal the mask, add oracle warm starts, or move to
  a reward-machine/curriculum setup.

## V10 Strict-Mask Outcome

Run date: 2026-06-08.

- Training commit: `f9b78f1937`.
- Checkpoint metadata / behavior-loader fix commit: `76fbf21da6`.
- Stage-1 root:
  `/workspace/tribal_event_mask_runs/stage1_v10_chain_affordance_compass_1m`.
- Corrected behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage1_v10_chain_affordance_compass_masked_76fbf21da6`.
- Result JSON validation passed for all three seeds.
- Corrected behavior-output validation passed for 11 rollout files split across
  `relh-sandbox-1` and `relh-sandbox-2`.

Training eval:

| Seed | Eval raw return | Eval shaped/individual return | `D_act_JS` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `-3.351` | `79.441` | `0.418` | `0.251` | `0.347` |
| `1` | `-11.275` | `108.642` | `0.502` | `0.285` | `0.361` |
| `2` | `-0.548` | `147.130` | `0.472` | `0.246` | `0.366` |

Corrected masked behavior over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Resources | Battery crafts | Heart deposits |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `-3.433` | `59.0` | `76` | `62` | `39` |
| `random` | `-45.103` | `33.0` | `87` | `0` | `0` |
| `seed0_det` | `12.533` | `75.0` | `88` | `81` | `56` |
| `seed0_stoch` | `-5.190` | `67.0` | `89` | `67` | `45` |
| `seed1_det` | `6.300` | `71.3` | `87` | `74` | `53` |
| `seed1_stoch` | `-3.600` | `81.3` | `102` | `82` | `60` |
| `seed2_det` | `5.933` | `74.0` | `90` | `82` | `50` |
| `seed2_stoch` | `-25.957` | `49.0` | `74` | `46` | `27` |

Decision:

- The strict v10 curriculum passes the constrained behavior gate. All three
  seeds complete the ore -> battery -> heart chain, deterministic rollouts beat
  no-op and random on raw reward, and corrected checkpoint attempts contain no
  `put`, `attack`, `plant`, or `swap`.
- The paper should not yet treat this as an unconstrained Tribal Village
  representation result. It is a strong cleanup/curriculum result showing that
  meaningful behavior is learnable when the distractor affordance surface is
  controlled.
- The next experiment should anneal or transfer from strict v10 into a less
  constrained action surface before any representation sweep.

## V10 Stage-2 Strict-Mask Outcome

Run date: 2026-06-09.

- Commit: `129dbac38b`.
- Stage-2 root:
  `/workspace/tribal_event_mask_runs/stage2_v10_chain_affordance_compass_10m`.
- Behavior gate root:
  `/workspace/tribal_event_mask_runs/behavior_gate_stage2_v10_chain_affordance_compass_129dbac38b`.
- Training jobs completed on `relh-sandbox-1` job `134` for seeds `0,1` and
  `relh-sandbox-2` job `133` for seed `2`.
- Result JSON validation passed for all three seeds.
- Behavior-output validation passed for 11 rollout files.

Training eval:

| Seed | Eval raw return | Eval shaped/individual return | `D_act_JS` | EffRank/n | Role probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `-10.949` | `-67.389` | `0.440` | `0.307` | `0.323` |
| `1` | `-10.627` | `-50.329` | `0.462` | `0.446` | `0.345` |
| `2` | `-4.907` | `33.563` | `0.391` | `0.362` | `0.283` |

Corrected masked behavior over 3 episodes x 240 steps:

| Rollout | Raw reward | Task events | Resources | Battery crafts | Heart deposits |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chain_oracle` | `-19.210` | `54.0` | `71` | `56` | `35` |
| `random` | `-28.733` | `12.7` | `37` | `0` | `0` |
| `seed0_det` | `-44.093` | `4.0` | `12` | `0` | `0` |
| `seed0_stoch` | `-35.227` | `11.0` | `30` | `3` | `0` |
| `seed1_det` | `-8.183` | `59.3` | `79` | `62` | `37` |
| `seed1_stoch` | `2.433` | `57.0` | `72` | `61` | `38` |
| `seed2_det` | `-11.567` | `34.3` | `47` | `44` | `12` |
| `seed2_stoch` | `-25.063` | `32.0` | `52` | `32` | `12` |

Decision:

- The strict mask still works mechanically: corrected checkpoint attempts had no
  `put`, `attack`, `plant`, or `swap`.
- Scaling strict v10 from 1M to 10M did not improve behavior. Seed `0` lost the
  heart chain almost completely, seed `2` weakened, and only seed `1` remained
  behavior-valid.
- Do not use the 10M strict-v10 checkpoints for a representation sweep.
- The next ramp should diagnose budget/overtraining and transfer from the clean
  1M strict-v10 checkpoints before relaxing the action surface.
