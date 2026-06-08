# CLAUDE.md

Guidance for AI assistants working on this codebase. See also `STYLE_GUIDE.md`.

## Setup

```bash
./install.sh              # Initial setup
metta status              # Check component status
metta install             # Reinstall if imports fail
```

Most of the time, you shouldn't need to run install.sh or metta install. Only run these if you're having trouble with
imports or other setup issues.

## Commands

```bash
# Training (always use timestep limit to avoid hanging)
uv run ./tools/run.py train arena run=my_experiment trainer.total_timesteps=100000

# Evaluation
uv run ./tools/run.py evaluate arena policy_uri=file://./train_dir/my_run/checkpoints

# List available tools
uv run ./tools/run.py arena --list
```

## Repository Structure

```
metta/          # Private - core RL training, not published separately
packages/
  mettagrid/    # Public - C++/Python grid environment
  cogames/      # Public - game configs, depends on mettagrid
recipes/
  prod/         # Production recipes with CI validation
  experiment/   # Work-in-progress recipes
```

## Package Dependencies

```
metta/ ──────► cogames/ ──────► mettagrid/
app_backend/ ──► common/ (only)
```

- Nothing depends on `metta/` (it's the top-level consumer)
- `mettagrid` has no internal Python dependencies (C++/Python hybrid)
- `app_backend` is isolated, can only import from `common/`
- Enforced by `import-linter`. Run `uv run lint-imports` to check. See `.importlinter`.

## Testing

- Don't speculatively run tests; only run when asked or in a targeted way for changes you made and want to validate

```bash
metta pytest tests/path/to/test.py -v    # Run specific test
metta pytest --changed                    # Run only tests affected by your changes
```

## Proto Files

Files in `proto/` define schemas for cross-system boundaries (network APIs, files on disk). Do not modify proto schemas
as part of other refactoring work. Schema changes require explicit discussion because they can break compatibility with
files written using older schemas or services that haven't been redeployed.

## Recipe System

```bash
./tools/run.py train arena run=test           # Two-token form
./tools/run.py arena --list                   # Show available tools
```

See `common/src/metta/common/tool/README.md` for details.

## Git Hub Integration

Use graphite ("gt") to create PRs. Name the branch $user-short-issue-name (5 words or less)

---

## Representation Collapse Paper (v3_experiments/)

Paper at `/home/ubuntu/698bd9a65fba5c04a962e794/samples/main.tex`.
Runs tracker at `/home/ubuntu/698bd9a65fba5c04a962e794/runs.md`.
WandB: project=`representation-collapse`, entity=`tashapais`.

### Running experiments

```bash
# Single seed on one GPU (use --seed_start to parallelize across GPUs)
CUDA_VISIBLE_DEVICES=N WANDB_API_KEY=... .venv/bin/python -u \
    v3_experiments/paper_exp_reward_type.py \
    --reward_type [individual|shared] [--contrastive] \
    --gpu 0 --num_agents [12|24] --num_seeds 1 --seed_start S

# Full parallel launch (all 8 GPUs)
bash v3_experiments/launch_2x2_contrastive.sh
```

Scikit-learn is declared in `pyproject.toml` for the linear probes.
`install.sh` fails in no-TTY — verify env with:
`.venv/bin/python -c "import mettagrid, wandb, sklearn, torch; print('OK')"`

### GPU throughput
- 12-agent runs: ~2763 agent-steps/sec → 5M steps ≈ 30 min/seed
- 24-agent runs: ~1760 agent-steps/sec → 5M steps ≈ 47 min/seed

### Key empirical findings (FINAL — as of 2026-03-11)

**Collapse is SEMANTIC not geometric:**
- shared, 12ag: EffRank/n=0.807 (HIGHER than individual's 0.657!)
- shared rewards collapse probe_acc to chance (0.502) while EffRank/n stays high
- **Probe accuracy is the correct diagnostic. EffRank/n alone is insufficient.**

**Probe accuracy results (Exp 4, no contrastive, CONFIRMED):**
- individual, 12ag: 0.784 ± 0.068  (Cohen's d=5.2 vs shared, p<0.001 ***)
- shared, 12ag:     0.502 ± 0.007  (chance)
- individual, 24ag: 0.669 ± 0.071  (Cohen's d=3.0 vs shared, p=0.002 **)
- shared, 24ag:     0.502 ± 0.008  (chance)

**InfoNCE contrastive (Exp 5 / 2×2 ablation, CONFIRMED):**
- individual+InfoNCE, 12ag: probe=0.500 ± 0.010  (COLLAPSES — was 0.784!)
- shared+InfoNCE, 12ag:     probe=0.503 ± 0.005  (still chance)
- Mechanism: InfoNCE encodes agent *identity* not role quality → EffRank/n ↑ to 0.840 but probe→chance

**SupCon(rank) contrastive (Exp 5, CONFIRMED):**
- individual+SupCon, 12ag: probe=0.499 ± 0.002  (COLLAPSES, n=3 seeds)
- shared+SupCon, 12ag:     probe=0.501 ± 0.006  (still chance, n=5 seeds)
- Mechanism: EffRank/n DROPS to 0.208-0.290 (binary clusters from near-identical episode returns)
- BOTH contrastive objectives fail — the collapse is fundamental to reward structure, not fixable with aux losses

**Exp 6 scaling curve (COMPLETE — all 40 seeds done)**
- All result files: results_reward_{individual,shared}_{6,12,18,24}agents.json
- Paper fully filled: no TBDs remain (table + result paragraph + abstract all updated)

**KEY REVISED FINDING (2026-03-11): Non-monotonic team-size × reward interaction**
Full picture (n=5 each, shared 18ag pending):
- 6ag:  individual probe=0.496±0.008 (CHANCE), shared=0.499±0.003 (chance)
- 12ag: individual probe=0.784±0.068 (HIGH!), shared=0.502±0.007 (chance)
- 18ag: individual probe=0.499±0.004 (CHANCE!), shared TBD (~0.502 expected)
- 24ag: individual probe=0.669±0.071 (HIGH!), shared=0.502±0.008 (chance)

PATTERN: Non-monotonic! Individual rewards produce role-aware representations at 12 and 24 agents
but collapse to chance at 6 and 18 agents. Shared rewards ALWAYS collapse to chance.

IMPLICATIONS FOR PAPER NARRATIVE:
- Shared rewards = necessary and sufficient for probe collapse (robust across all 4 sizes)
- Individual rewards = necessary but NOT sufficient for role-aware representations
- WIN-RATE CORRELATION: probe above-chance ↔ individual rewards improve win rate over shared
  - 6ag: ind_wr=0.098, shr_wr=0.074 (tiny gap) → probe at chance for both
  - 12ag: ind_wr=0.330, shr_wr=0.239 (clear gap) → probe above chance under ind
  - 18ag: ind_wr=0.349, shr_wr=0.350 (IDENTICAL!) → probe at chance under ind
  - 24ag: ind_wr=0.608, shr_wr=0.480 (clear gap) → probe above chance under ind
- INTERPRETATION: Role-differentiated representations only emerge when the game REWARDS
  specialization. At 6ag and 18ag, no role advantage exists → no role structure learned.
  At 12ag and 24ag, specialization helps → individual rewards enable role learning.
- Analysis section in paper updated with this win-rate correlation insight.

**Exp 8 (= 7b) — SepEnc at 6ag + 18ag (6ag DONE, 18ag RUNNING)**
- 6ag RESULTS (n=5 ind, n=3 shr pending 2 more seeds):
  - ind+sep_enc 6ag: probe=0.697±0.089, eff/n=2.293 ← ABOVE CHANCE (was 0.496 chance!)
  - shr+sep_enc 6ag: probe=0.711±0.066, eff/n=2.308 ← ABOVE CHANCE (was 0.499 chance!) [n=5 FINAL]
  - CONCLUSION: sep_enc RESCUES probe at 6ag for BOTH reward types!
  - NON-MONOTONIC MYSTERY RESOLVED: 6-agent games DO have learnable role structure.
    The Exp 6 failure was gradient homogenization overpowering per-agent credit at small team sizes.
  - NOT game dynamics — gradient sharing is the causal bottleneck at every tested team size.
- 18ag COMPLETE (n=5 each):
  - ind+sep_enc 18ag: probe=0.663±0.031 ← ABOVE CHANCE (was 0.499!)
  - shr+sep_enc 18ag: probe=0.719±0.046 ← ABOVE CHANCE (was 0.502!)
- Paper updated: Exp 8 section added, analysis paragraph rewritten, abstract + contributions updated
- 24ag COMPLETE (n=5 each):
  - ind+sep_enc 24ag: probe=0.703±0.018, eff/n=1.382
  - shr+sep_enc 24ag: probe=0.696±0.035, eff/n=1.415
- Exp 9 COMPLETE — CL+sep_enc (n=3 each):
  - ind+CL+sep_enc: probe=0.669±0.035 (vs ind+sep_enc 0.677 — neutral)
  - shr+CL+sep_enc: probe=0.651±0.008 (vs shr+sep_enc 0.686 — mild drop)
  - CL does NOT collapse with sep_enc (unlike shared_enc where CL→0.500!)
  - InfoNCE's catastrophic behavior is specific to shared encoder cross-agent gradient pressure
- Paper: Exp 9 section added to Analysis. All TBDs filled. Experiments complete.

**Exp 7 — Separate Encoder Ablation (COMPLETE, n=5 all conditions)**
Full 2×2 results:
- ind+shared_enc:  0.784±0.068, eff/n=0.657 (BEST — cross-agent info + per-agent credit)
- ind+sep_enc:     0.677±0.035, eff/n=2.050
- shared+sep_enc:  0.686±0.033, eff/n=2.025 (reward type BARELY MATTERS with sep encoders!)
- shared+shared_enc: 0.502±0.007, eff/n=0.807 (COLLAPSE — both mechanisms active)
- ΔProbe(ind vs shr, sep_enc) = 0.677-0.686 = -0.009 (negligible! reward type irrelevant with sep enc)
- ΔProbe(sep vs shared_enc, shr) = 0.686-0.502 = +0.184 (large! gradient isolation rescues)
- CONCLUSION: Gradient sharing is PRIMARY cause. With sep encoders, reward type barely matters.
  ind+shared_enc best overall (shared encoder benefits from cross-agent info when credit is per-agent)
- Paper COMPLETE: Exp 7 section, table, result paragraph, causal chain, abstract, contributions all updated

### Paper framing (REVISED 2026-03-11 after Exp 8)
The correct story: "Gradient sharing in the encoder is the proximate causal mechanism.
With a shared encoder, shared rewards collapse probe to chance at every team size. Individual
rewards help at 12ag and 24ag (enough gradient differentiation), fail at 6ag and 18ag (gradient
homogenization wins). Separate encoders rescue probe at all tested sizes regardless of reward type.
ALL contrastive objectives also collapse probe — InfoNCE via identity encoding, SupCon via
degenerate binary clusters. Grad sharing, not reward averaging per se, is the root cause."
Do NOT claim CL helps — it doesn't (probe→chance under both).
Do NOT claim "game dynamics limit 6ag" — Exp 8 disproves this (sep_enc rescues).
