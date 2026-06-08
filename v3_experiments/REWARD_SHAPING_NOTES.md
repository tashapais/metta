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
