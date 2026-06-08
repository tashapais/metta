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
and `swap` actions during this cleanup ramp. Checkpoint rollouts preserve the
same mask from the checkpoint config, so v10 should be read as a constrained
affordance/curriculum condition, not as an unconstrained final paper condition.

Promotion question:

- Does masking off-chain affordances keep v9's seed-1/seed-2 deposit signal
  while rescuing seed 0 and reducing off-chain resource/craft/combat counts?
