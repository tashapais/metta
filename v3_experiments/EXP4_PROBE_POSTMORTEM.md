# Postmortem: The Arena Reward-Attribution Probe (Exp 4)

Date: 2026-06-11. Authors: Richard (audit + reruns), with full provenance in
`PLAN.md` ("Arena Exp-4 Probe Audit" section) and artifacts on
`relh-sandbox-1/2`. Everything below is traceable to a wandb run, a saved
log, a git commit, or a result JSON.

## TL;DR

The headline result — probe accuracy **0.784 (individual) vs 0.502 (shared)**
— is a measurement artifact, not a property of the learned representations.
The two conditions were scored by different label code against different
chance baselines, traceable to a crash fix applied between the two launch
waves. A pre-registered corrected rerun (ground-truth contribution labels,
identical code, matched 0.500 baselines) finds **chance-level decoding in
both conditions**. The hypothesis itself is *not* refuted — but this arena
cannot test it, because per-agent contributions are too sparse and tied to
decode. The behavioral effect (individual rewards → higher win rates) is
real and directionally stable across every rerun.

Nothing here suggests bad faith. The bug is the kind that happens at 1am
when sklearn throws an error at the end of a 5M-step run: the fix was
locally reasonable, but it silently changed what the probe measures for one
condition only, and the one logged metric that would have caught it
(`probe_lift`) was never surfaced.

## 1. The original claim and runs

Ten wandb runs, project `tashapais/representation-collapse`, 2026-02-27:

- individual 12ag: `bswht815, 6htfbjes, mnzynfmk, m9ibxhrq, isvll1u7`
- shared 12ag: `vilbrhtl, n1ee7uzj, 63ebzghl, quue0asw, ud88lyms`

Claim: individual rewards leave "who is contributing" decodable from the
shared encoder (probe 0.784 ± 0.068); team-averaged rewards collapse it to
chance (0.502 ± 0.007).

## 2. What the runs' own logs show

| Finding | Evidence |
| --- | --- |
| Different chance baselines per condition | `final/probe_chance`: individual 0.932–0.965 (labels ~95/5 imbalanced); shared exactly 0.500 (balanced) |
| Individual probe is *below* its own baseline in every seed | `final/probe_lift`: −0.287, −0.104, −0.216, −0.129, −0.096 |
| Shared labels were information-free | per-agent returns are identical under team-mean reward; rank tie-breaking reduces to a fixed agent-index split |
| Probe data spans all of training | n=4848 = 404 episodes × 12 agents, collected from step 0 onward, not the final encoder |

The 0.784-vs-0.502 comparison places two numbers from two different tests on
one axis. By each test's own baseline, neither condition shows decoding.

## 3. How it happened (timeline, all times UTC, Feb 26–27)

- **23:49–00:05** — three launch waves crash in minutes (launcher debugging).
- **00:12:22** — first working batch: seed 0 of *everything*, from
  uncommitted code ("version A": threshold-style labels, no `reward_cl`
  args). Individual runs finish. **All four shared runs train fully, then
  crash at the probe**: run `b29c3pyp` preserves the traceback —
  `LogisticRegression`: *"the data contains only one class: 0"*. Under
  team-mean reward all returns tie, so version A's labels had no positive
  class. The same rule explains the ~95/5 imbalance under individual
  rewards.
- **01:16:57** — fixed script ("version B": balanced argsort-rank labels;
  the code comment cites this exact failure) relaunches **only the shared
  arm**. These finish, scoring ~0.50 on labels that are an arbitrary
  agent-index split under ties.
- **01:23:32** — version B committed (`ce4d46ba8`; `paper_exp_reward_type.py`
  is a new file in this commit, so version A was never committed).
- **01:46, 02:14** — individual seeds 3 and 4 launch *after* the fix exists
  but still run version A (their configs lack the `reward_cl` keys).
- **04:48:13** — result JSONs committed (`9391a951a`) with the mismatched
  baselines and negative lifts recorded inside them, unflagged.

One asymmetric crash fix → two different measuring instruments → one
headline comparison.

## 4. The corrected experiment

Pre-registered in `PLAN.md` before launch (commits `ed90af380`, `c22d2b24a`):

- Both conditions labeled by **ground-truth per-agent returns**, computed by
  the environment *before* any reward averaging — so the shared condition
  trains on shared reward but is labeled by true contribution.
- Identical label code in both conditions; balanced labels; strict
  separation required (tied episodes are skipped and counted, never
  arbitrarily labeled); chance = 0.500 by construction.
- Probe fit only on final-20%-of-training embeddings; GroupKFold over
  episodes (no episode spans train and test).
- 5 seeds × 2 conditions × 5M steps, same architecture/hparams as the
  originals. 20 runs total across protocol v1 and v2; all succeeded.

**Protocol v1 (top/bottom quartiles)**: >90% of episodes tie-skipped — most
agents contribute indistinguishably in most episodes. 9/10 runs below the
pre-registered validity threshold. This is itself a finding: the arena has
very little "who contributed" signal to decode.

**Protocol v2 (most vs least contributing agent per episode)**:

| Condition | Valid runs (≥25 separated episodes, ≥50 samples) | Probe accuracy |
| --- | --- | ---: |
| individual | seed 0 (n=70), seed 2 (n=56) | 0.586, 0.413 → pooled **0.509** |
| shared | seed 4 (n=56) | **0.503** |

Chance-level decoding in both conditions. Win rates in the rerun: individual
0.308, shared 0.261 (direction matches the originals: 0.330 vs 0.239).

## 5. Claim-by-claim status

| Claim | Status |
| --- | --- |
| "Individual rewards make contribution decodable (0.784); shared rewards collapse it (0.502)" | **Discard.** Measurement artifact; corrected probe finds chance in both. |
| "Shared rewards collapse representation geometry (EffRank/n)" | **Discard for this stream.** EffRank/n is *higher* under shared (0.70–0.99) than individual (0.52–0.84) in both the originals and reruns. |
| "Individual rewards improve task performance (win rate)" | **Keep, with caveats.** Directionally stable across originals + 2 reruns (≈0.31–0.33 vs ≈0.24–0.28); not individually significant at n=5; worth a powered test if it matters. |
| The feedback-attribution hypothesis itself | **Untested, not refuted.** This arena lacks per-agent contribution variance; a fair test needs an environment where contributions are dense and heterogeneous. |
| Tribal Village MAPPO role claims in `main.tex` (probe 0.921 → 0.371) | **Already discarded** by the behavior-first audit (separate stream; failed no-op/random/replay gates). |

## 6. Paper recommendation

- `main.tex`'s positive claims are unsupported and should not be submitted
  or left standing anywhere as-is.
- **No withdrawal is forced**: `main_richard.tex` already exists as the
  corrected paper — same project, honest framing, and a real contribution
  (behavior-first audits + the probe-baseline lesson, with this exp4 case
  study as Experiment 4). The realistic choice is which venue/framing for
  *that* paper, not whether to have one.
- The constructive next experiment, if we want a positive result: rebuild
  the attribution test in an environment with dense per-agent contribution
  (e.g., heterogeneous role-forced tasks), with the corrected probe protocol
  from day one. The corrected harness is ready
  (`paper_exp_reward_type.py --corrected_probe`).

## 7. Artifact index

- Forensic timeline + audit: `PLAN.md`, "Arena Exp-4 Probe Audit" section.
- Crash traceback: wandb run `b29c3pyp` `output.log` (final 25 lines).
- Corrected harness: `v3_experiments/paper_exp_reward_type.py`
  (`--corrected_probe`, `--probe_extremes`), commits `ed90af380` + `c22d2b24a`.
- Rerun artifacts: `/workspace/tribal_event_mask_runs/exp4_corrected_probe_ed90af380/`
  and `.../exp4_corrected_probe_v2_c22d2b24a/` on `relh-sandbox-1/2`
  (per-seed JSONs include episodes used/skipped, n, chance, lift).
- Corrected paper: Overleaf `samples/main_richard.tex` (Experiment 4),
  commit `2ec2b56`.
