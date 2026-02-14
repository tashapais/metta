"""
Statistical analysis for "Representation Collapse Predicts Coordination Failure".

Implements:
  - Bootstrap BCa 95% confidence intervals (10,000 resamples)
  - Welch's t-test (with Shapiro-Wilk normality check)
  - Cohen's d effect size

Comparisons:
  - MettaGrid team sizes: 6v12, 12v18, 18v24 (EffRank, expansion ratio)
  - Craftax depth ablation: 2v4, 4v8 layers (EffRank@1M)
  - Craftax batch ablation: 2048v8192, 8192v32768 (EffRank@1M)
  - SMAC: expansion ratio success vs failure (within each map)

Usage: python statistical_analysis.py [--latex]
"""

import json
import os
import argparse
import warnings
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    """Cohen's d effect size (pooled std)."""
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def bootstrap_bca_ci(a, b, n_boot=10_000, alpha=0.05, stat_func=None):
    """Bootstrap BCa confidence interval for difference in means.

    Returns (ci_low, ci_high, point_estimate).
    """
    rng = np.random.default_rng(42)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)

    if stat_func is None:
        stat_func = lambda x, y: np.mean(x) - np.mean(y)

    observed = stat_func(a, b)

    # Bootstrap distribution
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        a_boot = rng.choice(a, size=len(a), replace=True)
        b_boot = rng.choice(b, size=len(b), replace=True)
        boot_stats[i] = stat_func(a_boot, b_boot)

    # Bias correction
    z0 = stats.norm.ppf(np.mean(boot_stats < observed))

    # Acceleration (jackknife)
    combined = np.concatenate([a, b])
    n = len(combined)
    jackknife_stats = np.empty(n)
    for i in range(n):
        jk = np.delete(combined, i)
        jk_a, jk_b = jk[:len(a)], jk[len(a):]
        if len(jk_a) == 0 or len(jk_b) == 0:
            jackknife_stats[i] = observed
        else:
            jackknife_stats[i] = stat_func(jk_a, jk_b)
    jk_mean = jackknife_stats.mean()
    num = np.sum((jk_mean - jackknife_stats) ** 3)
    den = 6.0 * (np.sum((jk_mean - jackknife_stats) ** 2)) ** 1.5
    acc = num / den if den != 0 else 0.0

    # BCa percentiles
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1alpha = stats.norm.ppf(1 - alpha / 2)

    def bca_percentile(z_val):
        p = stats.norm.cdf(z0 + (z0 + z_val) / (1 - acc * (z0 + z_val)))
        return np.clip(p, 0.001, 0.999)

    ci_low = np.percentile(boot_stats, 100 * bca_percentile(z_alpha))
    ci_high = np.percentile(boot_stats, 100 * bca_percentile(z_1alpha))
    return ci_low, ci_high, observed


def welch_t_test(a, b):
    """Welch's t-test with Shapiro-Wilk normality check.

    Returns (t_stat, p_value, normal_a, normal_b).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)

    # Shapiro-Wilk normality (need n >= 3)
    normal_a = normal_b = True
    if len(a) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p_a = stats.shapiro(a)
        normal_a = p_a > 0.05
    if len(b) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p_b = stats.shapiro(b)
        normal_b = p_b > 0.05

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    return t_stat, p_value, normal_a, normal_b


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_json(filename):
    fpath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(fpath):
        print(f"Warning: {fpath} not found")
        return None
    with open(fpath) as f:
        return json.load(f)


def get_mettagrid_by_agents(data, n_agents):
    for entry in data:
        if entry.get("num_agents") == n_agents:
            return entry
    return None


def get_craftax_depth(data, depth, batch=8192):
    """Get depth ablation entry: horizon=default, given depth, batch=8192."""
    for entry in data:
        if (entry.get("horizon") == "default"
            and entry.get("num_encoder_layers") == float(depth)
            and entry.get("batch_size_actual") == float(batch)):
            return entry
    return None


def get_craftax_batch(data, batch_size):
    """Get batch ablation entry: horizon=default, depth=2, given batch."""
    for entry in data:
        if (entry.get("horizon") == "default"
            and entry.get("num_encoder_layers") == 2.0
            and entry.get("batch_size_actual") == float(batch_size)):
            return entry
    return None


# ---------------------------------------------------------------------------
# Run all comparisons
# ---------------------------------------------------------------------------

def run_comparison(label, a_seeds, b_seeds, a_label, b_label):
    """Run full statistical comparison between two groups."""
    a = np.array(a_seeds, dtype=float)
    b = np.array(b_seeds, dtype=float)

    d = cohens_d(a, b)
    ci_low, ci_high, point_est = bootstrap_bca_ci(a, b)
    t_stat, p_val, norm_a, norm_b = welch_t_test(a, b)

    sig = ""
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"

    normality_note = ""
    if not norm_a or not norm_b:
        normality_note = " (non-normal)"

    return {
        "label": label,
        "a_label": a_label,
        "b_label": b_label,
        "a_mean": np.mean(a),
        "b_mean": np.mean(b),
        "d": d,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "point_est": point_est,
        "t_stat": t_stat,
        "p_val": p_val,
        "sig": sig,
        "normality_note": normality_note,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    args = parser.parse_args()

    results = []

    # --- MettaGrid team size comparisons ---
    mg = load_json("results_mettagrid.json")
    if mg:
        for metric, metric_key in [("EffRank", "final_eff_rank_seeds"),
                                     ("ExpRatio", "expansion_ratio_seeds")]:
            pairs = [(6, 12), (12, 18), (18, 24)]
            for n_a, n_b in pairs:
                ea = get_mettagrid_by_agents(mg, n_a)
                eb = get_mettagrid_by_agents(mg, n_b)
                if ea and eb and metric_key in ea and metric_key in eb:
                    results.append(run_comparison(
                        f"MettaGrid {metric}: {n_a} vs {n_b} agents",
                        ea[metric_key], eb[metric_key],
                        f"{n_a} agents", f"{n_b} agents",
                    ))

    # --- Craftax depth ablation ---
    cx = load_json("results_craftax_part2.json")
    if cx:
        for d_a, d_b in [(2, 4), (4, 8)]:
            ea = get_craftax_depth(cx, d_a)
            eb = get_craftax_depth(cx, d_b)
            if ea and eb:
                results.append(run_comparison(
                    f"Craftax EffRank@1M: {d_a}L vs {d_b}L",
                    ea["eff_rank_at_1m_seeds"], eb["eff_rank_at_1m_seeds"],
                    f"{d_a} layers", f"{d_b} layers",
                ))

    # --- Craftax batch ablation ---
    if cx:
        for b_a, b_b in [(2048, 8192), (8192, 32768)]:
            ea = get_craftax_batch(cx, b_a)
            eb = get_craftax_batch(cx, b_b)
            if ea and eb:
                results.append(run_comparison(
                    f"Craftax EffRank@1M: batch {b_a} vs {b_b}",
                    ea["eff_rank_at_1m_seeds"], eb["eff_rank_at_1m_seeds"],
                    f"batch {b_a}", f"batch {b_b}",
                ))

    # --- SMAC expansion ratio success vs failure ---
    smac = load_json("results_smac_partial.json")
    if smac:
        for entry in smac:
            map_name = entry.get("map_name", "unknown")
            if "expansion_ratio_success_seeds" in entry and "expansion_ratio_failure_seeds" in entry:
                results.append(run_comparison(
                    f"SMAC {map_name}: ExpRatio success vs failure",
                    entry["expansion_ratio_success_seeds"],
                    entry["expansion_ratio_failure_seeds"],
                    "success", "failure",
                ))

    # --- Print results ---
    print("=" * 100)
    print("STATISTICAL ANALYSIS: Representation Collapse Predicts Coordination Failure")
    print("Bootstrap BCa CIs with 10,000 resamples, alpha = 0.05")
    print("=" * 100)
    print()
    print(f"{'Comparison':<50} {'d':>7} {'95% BCa CI':>20} {'p-value':>10} {'Sig':>5}")
    print("-" * 100)

    for r in results:
        ci_str = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        print(f"{r['label']:<50} {r['d']:>+7.3f} {ci_str:>20} {r['p_val']:>10.4f} {r['sig']:>5}{r['normality_note']}")

    print()
    print(f"Total comparisons: {len(results)}")
    print("Effect size interpretation: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large")
    print("Note: With n=5 seeds per condition, statistical power is limited. CIs should be interpreted cautiously.")

    # --- LaTeX output ---
    if args.latex:
        print()
        print("% ---- LaTeX table for paper appendix ----")
        print(r"\begin{table}[h]")
        print(r"\caption{Statistical comparisons (5 seeds, bootstrap BCa 95\% CIs with 10{,}000 resamples).}")
        print(r"\label{tab:stats}")
        print(r"\small")
        print(r"\begin{tabular}{lcccc}")
        print(r"\toprule")
        print(r"Comparison & Cohen's $d$ & 95\% BCa CI & $p$-value & Sig. \\")
        print(r"\midrule")

        # Group by environment
        current_env = ""
        for r in results:
            env = r["label"].split(":")[0].strip() if ":" in r["label"] else r["label"]
            if env != current_env:
                if current_env:
                    print(r"\addlinespace")
                current_env = env

            short_label = r["label"].split(": ", 1)[1] if ": " in r["label"] else r["label"]
            ci_str = f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
            sig_str = r["sig"] if r["sig"] else "n.s."
            p_str = f"{r['p_val']:.3f}" if r['p_val'] >= 0.001 else "$<$0.001"
            print(f"{short_label} & {r['d']:+.2f} & {ci_str} & {p_str} & {sig_str} \\\\")

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\par\smallskip\noindent\textit{Note:} With $n=5$ seeds, statistical power is limited.")
        print(r"\end{table}")


if __name__ == "__main__":
    main()
