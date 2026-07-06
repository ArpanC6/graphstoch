import json
import numpy as np
from scipy import stats

with open("synthetic_homophily_sweep_results_v2b_reaction.json") as f:
    data = json.load(f)

results = data["results"]
for gs in ["42", "43", "44"]:
    for h in ["0.5", "0.6", "0.7", "0.8", "0.9"]:
        reaction = np.array(results[gs][h]["logreg_denoised_reaction"])
        for method in ["gcn", "graphsage", "gat"]:
            other = np.array(results[gs][h][method])
            t, p = stats.ttest_rel(reaction, other)
            diff = reaction.mean() - other.mean()
            if diff > 0:
                sig = "SIGNIFICANT" if p < 0.05 else "not sig"
                print(f"seed{gs} h={h}: reaction beats {method} "
                      f"(diff={diff:+.4f}, p={p:.4f}, {sig})")
