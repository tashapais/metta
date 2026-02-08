"""Extract learning curves from wandb and generate variance analysis plots."""
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

api = wandb.Api()

# Arena PPO Baseline runs (3 seeds)
ppo_run_ids = [
    "baseline_ppo.01_21_26",
    "baseline_ppo.seed2.01_21_26",
    "baseline_ppo.seed3.01_21_26",
]

# Arena PPO+C runs (3 seeds)
ppoc_run_ids = [
    "ppo_plus_contrastive.01_21_26",
    "ppo_plus_contrastive.seed2.01_21_26",
    "ppo_plus_contrastive.seed3.01_21_26",
]

# Ablation: embed_dim=64 (3 seeds)
embed64_run_ids = [
    "ablation_embed_64.01_21_26",
    "ablation_embed_64.seed2.01_21_26",
    "ablation_embed_64.seed3.01_21_26",
]

METRIC = "env_game/assembler.heart.created"
ENTITY = "tashapais"
PROJECT = "metta"


def extract_metric(run_ids, metric_key=METRIC):
    """Extract a metric's history from multiple wandb runs using history() (fast)."""
    all_histories = []
    for run_id in run_ids:
        print(f"  Fetching {run_id}...", end=" ", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        # Use history() with samples parameter for speed
        df = run.history(keys=[metric_key], samples=500)
        if metric_key in df.columns:
            df = df.dropna(subset=[metric_key])
            steps = df["_step"].values
            values = df[metric_key].values
            print(f"{len(values)} points, final={values[-1]:.2f}")
            all_histories.append((steps, values))
        else:
            print(f"NO DATA for {metric_key}")
    return all_histories


def interpolate_to_common(histories, num_points=100):
    """Interpolate all runs to common step grid."""
    if not histories:
        return None, None, None
    min_step = max(h[0][0] for h in histories)
    max_step = min(h[0][-1] for h in histories)
    common = np.linspace(min_step, max_step, num_points)
    interp = np.array([np.interp(common, h[0], h[1]) for h in histories])
    return common, np.mean(interp, axis=0), np.std(interp, axis=0), interp


print("=" * 60)
print("Extracting PPO Baseline learning curves...")
ppo_h = extract_metric(ppo_run_ids)

print("\nExtracting PPO+C learning curves...")
ppoc_h = extract_metric(ppoc_run_ids)

print("\nExtracting PPO+C dim=64 learning curves...")
e64_h = extract_metric(embed64_run_ids)

ppo_s, ppo_m, ppo_sd, ppo_all = interpolate_to_common(ppo_h)
ppoc_s, ppoc_m, ppoc_sd, ppoc_all = interpolate_to_common(ppoc_h)
e64_s, e64_m, e64_sd, e64_all = interpolate_to_common(e64_h)

# ============================================================================
# Plot 1: Learning curves with confidence bands
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for label, steps, mean, std, color in [
    ("PPO Baseline", ppo_s, ppo_m, ppo_sd, "C0"),
    ("PPO+C", ppoc_s, ppoc_m, ppoc_sd, "C1"),
    ("PPO+C dim=64", e64_s, e64_m, e64_sd, "C2"),
]:
    if steps is not None:
        ax.plot(steps, mean, color=color, label=f'{label} ({mean[-1]:.1f})', linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)
ax.set_xlabel('Training Step')
ax.set_ylabel('Hearts Created')
ax.set_title('Arena: Mean +/- Std (3 seeds)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: Cross-seed std over training
ax = axes[1]
for label, steps, all_runs, color in [
    ("PPO Baseline", ppo_s, ppo_all, "C0"),
    ("PPO+C", ppoc_s, ppoc_all, "C1"),
    ("PPO+C dim=64", e64_s, e64_all, "C2"),
]:
    if steps is not None:
        cross_std = np.std(all_runs, axis=0)
        ax.plot(steps, cross_std, color=color, label=label, linewidth=2)
ax.set_xlabel('Training Step')
ax.set_ylabel('Cross-Seed Std (Hearts)')
ax.set_title('Training Variance Over Time')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/metta/v3_experiments/learning_curves.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: v3_experiments/learning_curves.png")

# ============================================================================
# Plot 2: Individual seed trajectories
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, histories, title in [
    (axes[0], ppo_h, "PPO Baseline"),
    (axes[1], ppoc_h, "PPO+C"),
    (axes[2], e64_h, "PPO+C dim=64"),
]:
    for i, (steps, values) in enumerate(histories):
        ax.plot(steps, values, f'C{i}-', alpha=0.7, label=f'Seed {i+1} (final: {values[-1]:.1f})')
    ax.set_title(title)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Hearts Created')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/metta/v3_experiments/individual_seeds.png', dpi=150, bbox_inches='tight')
print(f"Saved: v3_experiments/individual_seeds.png")

# ============================================================================
# Summary statistics
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

results = {}
for name, histories in [("PPO Baseline", ppo_h), ("PPO+C", ppoc_h), ("PPO+C dim=64", e64_h)]:
    if histories:
        finals = [h[1][-1] for h in histories]
        results[name] = finals
        print(f"\n{name}:")
        print(f"  Final values: {[f'{v:.2f}' for v in finals]}")
        print(f"  Mean +/- Std: {np.mean(finals):.2f} +/- {np.std(finals):.2f}")
        print(f"  CV: {np.std(finals)/np.mean(finals)*100:.1f}%")

if "PPO Baseline" in results and "PPO+C" in results:
    ppo_f = results["PPO Baseline"]
    ppoc_f = results["PPO+C"]
    print(f"\n{'='*60}")
    print("VARIANCE REDUCTION ANALYSIS")
    print(f"{'='*60}")
    print(f"PPO  std: {np.std(ppo_f):.2f}")
    print(f"PPO+C std: {np.std(ppoc_f):.2f}")
    reduction = (1 - np.std(ppoc_f)/np.std(ppo_f))*100
    print(f"Variance reduction: {reduction:.1f}%")
    f_ratio = np.var(ppo_f)/np.var(ppoc_f)
    print(f"F-test ratio (PPO var / PPO+C var): {f_ratio:.2f}")

    # Bootstrap CI for variance ratio
    np.random.seed(42)
    n_boot = 10000
    ratios = []
    for _ in range(n_boot):
        ppo_boot = np.random.choice(ppo_f, size=len(ppo_f), replace=True)
        ppoc_boot = np.random.choice(ppoc_f, size=len(ppoc_f), replace=True)
        if np.var(ppoc_boot) > 0:
            ratios.append(np.var(ppo_boot) / np.var(ppoc_boot))
    ratios = np.array(ratios)
    print(f"Bootstrap 95% CI for F-ratio: [{np.percentile(ratios, 2.5):.2f}, {np.percentile(ratios, 97.5):.2f}]")
    print(f"  (>1 means PPO has higher variance)")
