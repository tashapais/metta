#!/usr/bin/env python3
"""
update_paper_table3.py
======================
Reads the 4 contrastive experiment result JSON files and updates Table 3 (and
abstract numbers) in main.tex with real values.

Usage:
    python update_paper_table3.py [--dry-run]
"""
import json
import re
import math
import argparse
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("/home/devuser/metta/v3_experiments")
TEX_FILE    = Path("/home/devuser/698bd9a65fba5c04a962e794/samples/main.tex")

RESULT_FILES = {
    ("baseline",     18): RESULTS_DIR / "results_contrastive_baseline_18agents.json",
    ("contrastive",  18): RESULTS_DIR / "results_contrastive_contrastive_18agents.json",
    ("baseline",     24): RESULTS_DIR / "results_contrastive_baseline_24agents.json",
    ("contrastive",  24): RESULTS_DIR / "results_contrastive_contrastive_24agents.json",
}

# ── Bootstrap BCa CI ─────────────────────────────────────────────────────────
def bca_ci(data, stat_func=np.mean, n_boot=10_000, ci=0.95, rng=None):
    """Bootstrap BCa confidence interval for a scalar statistic."""
    if rng is None:
        rng = np.random.default_rng(42)
    data = np.asarray(data, dtype=float)
    n = len(data)
    theta_hat = stat_func(data)

    # Bootstrap distribution
    boots = [stat_func(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)]
    boots = np.array(boots)

    # Bias correction z0
    z0 = stats.norm.ppf((boots < theta_hat).mean())

    # Acceleration a (jackknife)
    jk = np.array([stat_func(np.delete(data, i)) for i in range(n)])
    jk_mean = jk.mean()
    numer = ((jk_mean - jk) ** 3).sum()
    denom = (6 * ((jk_mean - jk) ** 2).sum() ** 1.5)
    a = numer / denom if denom != 0 else 0.0

    alpha = 1 - ci
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    p_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    p_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

    lo = np.percentile(boots, p_lo * 100)
    hi = np.percentile(boots, p_hi * 100)
    return float(lo), float(hi)


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled_std = math.sqrt(((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else float("nan")


def load(method, n_agents):
    path = RESULT_FILES[(method, n_agents)]
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path) as f:
        return json.load(f)


def fmt(val, n_digits=2):
    return f"{val:.{n_digits}f}"


def fmt_pm(mean, std, n_digits=2):
    return f"{mean:.{n_digits}f} $\\pm$ {std:.{n_digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing to disk")
    args = parser.parse_args()

    # ── Load results ─────────────────────────────────────────────────────────
    data = {}
    for (method, n) in RESULT_FILES:
        try:
            data[(method, n)] = load(method, n)
            print(f"  Loaded {method}/{n}agents: {len(data[(method, n)])} seeds")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            data[(method, n)] = None

    rng = np.random.default_rng(42)

    # ── Helper: extract metric vectors ────────────────────────────────────────
    def vec(method, n, key):
        d = data[(method, n)]
        if d is None:
            return None
        return np.array([r[key] for r in d], dtype=float)

    # ── 18-agent statistics ───────────────────────────────────────────────────
    b18_rank  = vec("baseline",    18, "final_eff_rank")
    c18_rank  = vec("contrastive", 18, "final_eff_rank")
    b18_exp   = vec("baseline",    18, "final_exp_ratio")
    c18_exp   = vec("contrastive", 18, "final_exp_ratio")
    b18_div   = vec("baseline",    18, "final_act_div")
    c18_div   = vec("contrastive", 18, "final_act_div")
    b18_ret   = vec("baseline",    18, "final_return")
    c18_ret   = vec("contrastive", 18, "final_return")
    b18_wr    = vec("baseline",    18, "win_rate")
    c18_wr    = vec("contrastive", 18, "win_rate")

    # ── 24-agent statistics ───────────────────────────────────────────────────
    b24_rank  = vec("baseline",    24, "final_eff_rank")
    c24_rank  = vec("contrastive", 24, "final_eff_rank")
    b24_exp   = vec("baseline",    24, "final_exp_ratio")
    c24_exp   = vec("contrastive", 24, "final_exp_ratio")
    b24_div   = vec("baseline",    24, "final_act_div")
    c24_div   = vec("contrastive", 24, "final_act_div")
    b24_ret   = vec("baseline",    24, "final_return")
    c24_ret   = vec("contrastive", 24, "final_return")
    b24_wr    = vec("baseline",    24, "win_rate")
    c24_wr    = vec("contrastive", 24, "win_rate")

    # ── Clip pathological exp_ratio values ───────────────────────────────────
    def clip_exp(v):
        if v is None:
            return None
        return np.clip(v, 0, 100)

    b18_exp = clip_exp(b18_exp)
    c18_exp = clip_exp(c18_exp)
    b24_exp = clip_exp(b24_exp)
    c24_exp = clip_exp(c24_exp)

    # ── Build table rows ─────────────────────────────────────────────────────
    def row(rank_v, n_agents, exp_v, div_v, wr_v, n_digits_wr=3):
        if rank_v is None:
            return "\\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD}"
        eff_mean = rank_v.mean();  eff_std  = rank_v.std(ddof=1)
        epn_mean = (rank_v / n_agents).mean(); epn_std = (rank_v / n_agents).std(ddof=1)
        exp_mean = exp_v.mean() if exp_v is not None else float("nan")
        exp_std  = exp_v.std(ddof=1) if exp_v is not None else float("nan")
        div_mean = div_v.mean() if div_v is not None else float("nan")
        div_std  = div_v.std(ddof=1) if div_v is not None else float("nan")
        wr_mean  = wr_v.mean() if wr_v is not None else float("nan")
        wr_std   = wr_v.std(ddof=1) if wr_v is not None else float("nan")
        return (
            f"{eff_mean:.2f} $\\pm$ {eff_std:.2f} & "
            f"{epn_mean:.3f} $\\pm$ {epn_std:.3f} & "
            f"{exp_mean:.2f} $\\pm$ {exp_std:.2f} & "
            f"{div_mean:.4f} $\\pm$ {div_std:.4f} & "
            f"{wr_mean:.{n_digits_wr}f} $\\pm$ {wr_std:.{n_digits_wr}f}"
        )

    def delta_row(c_rank, b_rank, n_agents, c_exp, b_exp, c_div, b_div, c_wr, b_wr):
        if c_rank is None or b_rank is None:
            return "\\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD}"
        d_rank = (c_rank - b_rank).mean()
        d_epn  = ((c_rank - b_rank) / n_agents).mean()
        d_exp  = (c_exp - b_exp).mean() if c_exp is not None and b_exp is not None else float("nan")
        d_div  = (c_div - b_div).mean() if c_div is not None and b_div is not None else float("nan")
        d_wr   = (c_wr - b_wr).mean() if c_wr is not None and b_wr is not None else float("nan")
        s = lambda v, fmt=".2f": f"{v:+{fmt}}"
        return (
            f"{s(d_rank)} & "
            f"{s(d_epn, '.3f')} & "
            f"{s(d_exp)} & "
            f"{s(d_div, '.4f')} & "
            f"{s(d_wr, '.3f')}"
        )

    b18_row = row(b18_rank, 18, b18_exp, b18_div, b18_wr)
    c18_row = row(c18_rank, 18, c18_exp, c18_div, c18_wr)
    d18_row = delta_row(c18_rank, b18_rank, 18, c18_exp, b18_exp, c18_div, b18_div, c18_wr, b18_wr)

    b24_row = row(b24_rank, 24, b24_exp, b24_div, b24_wr)
    c24_row = row(c24_rank, 24, c24_exp, c24_div, c24_wr)
    d24_row = delta_row(c24_rank, b24_rank, 24, c24_exp, b24_exp, c24_div, b24_div, c24_wr, b24_wr)

    # ── Statistical test (18-agent EffRank) ──────────────────────────────────
    if c18_rank is not None and b18_rank is not None:
        d_val = cohen_d(c18_rank, b18_rank)
        ci_lo, ci_hi = bca_ci(c18_rank - b18_rank, rng=rng)
        t, pval = stats.ttest_ind(c18_rank, b18_rank)
        sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s.")
        stats_row_18 = (
            f"{d_val:+.2f} & "
            f"$[{ci_lo:+.2f}, {ci_hi:+.2f}]$ & "
            f"{pval:.3f} & "
            f"{sig}"
        )
    else:
        stats_row_18 = "\\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD}"

    # ── Abstract numbers ──────────────────────────────────────────────────────
    if c18_rank is not None and b18_rank is not None:
        pct_rank_improvement = ((c18_rank.mean() - b18_rank.mean()) / b18_rank.mean() * 100)
        abs_exp_improvement  = (c18_exp - b18_exp).mean() if (c18_exp is not None and b18_exp is not None) else float("nan")
    else:
        pct_rank_improvement = 31.0   # keep placeholder
        abs_exp_improvement  = 0.21

    print(f"\n{'='*65}")
    print(f"18-agent baseline:    {b18_row}")
    print(f"18-agent contrastive: {c18_row}")
    print(f"18-agent delta:       {d18_row}")
    print(f"18-agent stats:       {stats_row_18}")
    print(f"24-agent baseline:    {b24_row}")
    print(f"24-agent contrastive: {c24_row}")
    print(f"24-agent delta:       {d24_row}")
    print(f"Abstract: {pct_rank_improvement:.0f}% rank improvement, {abs_exp_improvement:+.2f} exp_ratio")
    print(f"{'='*65}\n")

    if args.dry_run:
        print("[dry-run] Not writing to disk.")
        return

    # ── Patch main.tex ────────────────────────────────────────────────────────
    tex = TEX_FILE.read_text()

    # ────── Table 3: replace the 3 data rows ──────────────────────────────────
    # Original table only has 18 agents. Expand to include 24 agents.
    old_table_body = (
        " & MAPPO (baseline)    & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} \\\\\n"
        " & MAPPO+Contrastive   & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} \\\\\n"
        "\\cmidrule{2-7}\n"
        " & $\\Delta$ (Contrastive $-$ Baseline) & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} \\\\\n"
        "\\addlinespace\n"
        "\\multirow{3}{*}{24}\n"
        " & MAPPO (baseline)    & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} \\\\\n"
        " & MAPPO+Contrastive   & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} \\\\\n"
        "\\cmidrule{2-7}\n"
        " & $\\Delta$ (Contrastive $-$ Baseline) & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} & \\todo{TBD} \\\\"
    )

    new_table_body = (
        f" & MAPPO (baseline)    & {b18_row} \\\\\n"
        f" & MAPPO+Contrastive   & {c18_row} \\\\\n"
        f"\\cmidrule{{2-7}}\n"
        f" & $\\Delta$ (Contrastive $-$ Baseline) & {d18_row} \\\\\n"
        f"\\addlinespace\n"
        f"\\multirow{{3}}{{*}}{{24}}\n"
        f" & MAPPO (baseline)    & {b24_row} \\\\\n"
        f" & MAPPO+Contrastive   & {c24_row} \\\\\n"
        f"\\cmidrule{{2-7}}\n"
        f" & $\\Delta$ (Contrastive $-$ Baseline) & {d24_row} \\\\"
    )

    if old_table_body in tex:
        tex = tex.replace(old_table_body, new_table_body)
        print("[OK] Replaced Table 3 body")
    else:
        print("[WARN] Could not find Table 3 body to replace — check manually")

    # ────── Result paragraph ──────────────────────────────────────────────────
    old_result = (r"\textbf{Result.} \todo{TBD: fill in once inter-agent InfoNCE runs complete "
                  r"(5 seeds $\times$ 2 team sizes).}")

    if c18_rank is not None and b18_rank is not None:
        rank_dir = "higher" if pct_rank_improvement > 0 else "lower"
        exp_sign = "+" if abs_exp_improvement > 0 else ""
        new_result = (
            f"\\textbf{{Result.}} "
            f"MAPPO+Contrastive achieves {abs(pct_rank_improvement):.0f}\\% {rank_dir} effective rank "
            f"($\\Delta$EffRank/n\\,$={((c18_rank / 18).mean() - (b18_rank / 18).mean()):+.3f}$) "
            f"and {exp_sign}{abs_exp_improvement:.2f} expansion ratio improvement "
            f"over baseline MAPPO at 18 agents, averaged across 5 seeds. "
            f"Cohen's $d = {cohen_d(c18_rank, b18_rank):.2f}$; "
            f"BCa~95\\%~CI on EffRank difference: "
            f"$[{bca_ci(c18_rank - b18_rank, rng=rng)[0]:+.2f},\\ "
            f"{bca_ci(c18_rank - b18_rank, rng=rng)[1]:+.2f}]$."
        )
    else:
        new_result = r"\textbf{Result.} \todo{Fill in from runs.}"

    if old_result in tex:
        tex = tex.replace(old_result, new_result)
        print("[OK] Replaced Result paragraph")
    else:
        print("[WARN] Could not find Result paragraph — check manually")

    # ────── Stats table row ───────────────────────────────────────────────────
    old_stats_row = r"& MAPPO vs MAPPO+CL & \todo{TBD} & \todo{TBD} & \todo{TBD} & \todo{TBD} \\"
    new_stats_row = f"& MAPPO vs MAPPO+CL (18 agents) & {stats_row_18} \\\\"
    if old_stats_row in tex:
        tex = tex.replace(old_stats_row, new_stats_row)
        print("[OK] Replaced stats table row")
    else:
        print("[WARN] Could not find stats table row — check manually")

    # ────── Abstract numbers ──────────────────────────────────────────────────
    # Replace "$31\%$ higher effective rank and $+0.21$ expansion ratio improvement"
    old_abs = r"$31\%$ higher effective rank and $+0.21$ expansion ratio improvement"
    if c18_rank is not None and b18_rank is not None:
        exp_sign = "+" if abs_exp_improvement >= 0 else ""
        new_abs = (
            f"${abs(pct_rank_improvement):.0f}\\%$ "
            f"{'higher' if pct_rank_improvement > 0 else 'lower'} effective rank "
            f"and ${exp_sign}{abs_exp_improvement:.2f}$ expansion ratio improvement"
        )
        if old_abs in tex:
            tex = tex.replace(old_abs, new_abs)
            print("[OK] Replaced abstract numbers")
        else:
            print("[WARN] Could not find abstract numbers — check manually")

    # ── Write updated .tex ─────────────────────────────────────────────────────
    TEX_FILE.write_text(tex)
    print(f"\n[OK] Written updated LaTeX → {TEX_FILE}")


if __name__ == "__main__":
    main()
