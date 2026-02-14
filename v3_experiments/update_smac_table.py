"""
Read results_smac.json and print the LaTeX table rows for tab:exp1.
Run this after SMAC experiments complete.
"""
import json
import sys

results_file = "/home/ubuntu/metta/v3_experiments/results_smac.json"

with open(results_file) as f:
    results = json.load(f)

# Map display info
map_info = {
    "3s5z": ("3s5z", 8),
    "5m_vs_6m": ("5m\\_vs\\_6m", 5),
    "corridor": ("corridor", 6),
    "3s_vs_5z": ("3s\\_vs\\_5z", 3),
}

print("% SMAC Lead Time Table (tab:exp1) - from 5-seed results")
print("% Paste these rows into the table body:")
print()

for map_name in ["3s5z", "5m_vs_6m", "corridor", "3s_vs_5z"]:
    key = f"smac_{map_name}"
    if key not in results:
        print(f"% WARNING: {key} not found in results")
        continue

    data = results[key]
    display_name, n_agents = map_info[map_name]

    lead_time = data.get("lead_time_steps", 0)
    lead_time_std = data.get("lead_time_steps_std", 0)
    lead_time_ratio = data.get("lead_time_ratio", 0)
    lead_time_ratio_std = data.get("lead_time_ratio_std", 0)

    # Format lead time
    lt_str = f"{int(lead_time):,}"
    lt_std_str = f"{int(lead_time_std):,}" if lead_time_std > 0 else "---"

    # Format ratio
    ratio_str = f"{lead_time_ratio:.1f}"
    ratio_std_str = f"{lead_time_ratio_std:.1f}" if lead_time_ratio_std > 0 else "---"

    print(f"{display_name} & {n_agents} & {lt_str} $\\pm$ {lt_std_str} & {ratio_str} $\\pm$ {ratio_std_str}$\\times$ \\\\")

print()
print("% Also available metrics per map:")
for map_name in ["3s5z", "5m_vs_6m", "corridor", "3s_vs_5z"]:
    key = f"smac_{map_name}"
    if key not in results:
        continue
    data = results[key]
    print(f"% {map_name}: effrank={data.get('final_effective_rank', '?')}±{data.get('final_effective_rank_std', '?')}, "
          f"winrate={data.get('final_win_rate', '?')}±{data.get('final_win_rate_std', '?')}, "
          f"svd={data.get('svd_ratio', '?')}±{data.get('svd_ratio_std', '?')}")
