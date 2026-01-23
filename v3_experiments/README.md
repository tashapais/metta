# V3 Experiments for Contrastive Learning Paper

This folder contains experiment configurations and baselines for the contrastive learning paper.

## Required Experiments

### 1. Craftax Evaluation (CRITICAL)
Currently missing evaluations on non-MettaGrid environments. Need to add Craftax.

```bash
# TODO: Setup Craftax environment integration
# - Install Craftax: pip install craftax
# - Create adapter for Craftax observation space
# - Run PPO baseline on Craftax
# - Run PPO+Contrastive on Craftax
# - Run GC-CRL on Craftax
```

### 2. Baseline Comparisons

#### Representation Learning Baselines
- [ ] CURL (Contrastive Unsupervised Representations for RL) - arXiv:2004.04136
- [ ] SPR (Self-Predictive Representations) - arXiv:2007.05929
- [ ] ATC (Augmented Temporal Contrast) - arXiv:2009.08319

#### Intrinsic Motivation Baselines
- [ ] ICM (Intrinsic Curiosity Module) - arXiv:1705.05363
- [ ] RND (Random Network Distillation) - arXiv:1810.12894

### 3. Ablation Studies to Complete

- [ ] Temperature sweep: tau in {0.01, 0.05, 0.1, 0.2, 0.5}
- [ ] Embedding dimension sweep: {32, 64, 128, 256}
- [ ] Contrastive coefficient sweep: {0.0001, 0.001, 0.01, 0.1}
- [ ] Geometric discount sweep: {0.9, 0.95, 0.977, 0.99}
- [ ] Number of seeds: increase to 5+ per condition

### 4. Multi-Agent Specific Experiments

- [ ] Vary number of agents: {4, 8, 16, 24, 48}
- [ ] Test with heterogeneous vs homogeneous policies
- [ ] Measure inter-agent embedding similarity over training

## Paper Novelty Analysis

### What's Novel in Our Approach:
1. **Geometric positive sampling tied to RL discount factor** - Links contrastive temporal horizon to value function horizon
2. **Systematic comparison of auxiliary vs primary contrastive objectives** in same framework
3. **Variance reduction as primary benefit** - Novel finding that auxiliary contrastive helps consistency more than mean performance
4. **Multi-agent population as natural negative mining** - Different from prior work that uses random negatives

### What's NOT Novel (need to cite properly):
1. InfoNCE loss itself (van den Oord et al., 2018)
2. Goal-conditioned contrastive RL (Nimonkar et al., 2025)
3. Temporal contrastive learning in RL (CURL, SPR, ATC)

### Key Differentiators from Related Work:
- **vs Nimonkar et al. (2509.10656)**: They focus on cooperation emergence; we focus on representation quality and training stability
- **vs Liu et al. (2408.05804)**: They use single-goal setting; we use multi-agent population-based training
- **vs CURL/SPR/ATC**: They use augmentation-based positives; we use geometric temporal sampling

## Experiment Configuration Files

See `configs/` subfolder for specific hyperparameter configurations.

## Results Storage

Results should be stored in:
- `results/craftax/` - Craftax environment results
- `results/arena/` - MettaGrid Arena results
- `results/ablations/` - Ablation study results
- `results/baselines/` - Baseline comparison results
