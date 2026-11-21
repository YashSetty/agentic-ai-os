"""MAS-RL comparison on the 504-run test grid (8 windows x loads 0.7/0.9/1.1 x failure seeds 1-3).

Usage: python analyze_masrl.py results/grid/per_run.csv results/masrl/test/per_run.jsonl [prefix]

"mas-rl" = mean over the frozen training seeds (mas-rl-s1..s5) of each run cell, so
each (window, load, failure seed) contributes one paired observation. Reports:
  * means by load, incl. per-seed spread of MAS-RL
  * paired differences A - B with 95% window-cluster bootstrap CIs (resampling windows)
  * Wilcoxon signed-rank test on the 8 window-level mean differences (per load and pooled)
  * test-set objective J (equal weights, priority-gated normaliser) per strategy
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

M = ["avg_latency_s", "util_pct", "avg_recovery_s", "violation_rate_pct", "fairness_jain"]
SHORT = {"avg_latency_s": "lat", "util_pct": "util", "avg_recovery_s": "rec",
         "violation_rate_pct": "viol", "fairness_jain": "jain"}
grid = pd.read_csv(sys.argv[1])
ml = pd.read_json(sys.argv[2], lines=True)
prefix = sys.argv[3] if len(sys.argv) > 3 else "mas-rl"
seeds = sorted(s for s in ml.strategy.unique() if s.startswith(prefix + "-s"))
K = ["window_id", "load", "seed"]
agg = ml[ml.strategy.isin(seeds)].groupby(K)[M].mean().reset_index().assign(strategy=prefix)
df = pd.concat([grid, ml, agg], ignore_index=True)
rng = np.random.default_rng(0)


def boot_cluster(d, n=4000):
    wins = d.window_id.unique()
    arr = {w: d[d.window_id == w]["diff"].values for w in wins}
    means = [np.concatenate([arr[w] for w in rng.choice(wins, len(wins))]).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def paired(g, a, b):
    x = g[g.strategy == a].set_index(K)[M]
    y = g[g.strategy == b].set_index(K)[M]
    return (x - y).dropna(how="all").reset_index()


def J(g):
    ref = g[g.strategy == "priority-gated"].set_index(K)
    out = {}
    for s, h in g.groupby("strategy"):
        h = h.set_index(K)
        r = ref.loc[h.index]
        rec = (h.avg_recovery_s / r.avg_recovery_s).where(r.avg_recovery_s > 0, 1.0).fillna(1.0)
        out[s] = float((0.2 * (h.avg_latency_s / r.avg_latency_s + 1 - h.util_pct / 100 + rec
                               + h.violation_rate_pct / 100 + 1 - h.fairness_jain)).mean())
    return pd.Series(out)


order = ["priority-gated", "agentic-nollm", "random-rank", prefix] + seeds
print("== Means by load (24 runs per cell; mas-rl = mean of", len(seeds), "training seeds)")
for load, g in df.groupby("load"):
    t = g.groupby("strategy")[M].mean().reindex([s for s in order if s in set(g.strategy)])
    t["J"] = J(g)
    print(f"\nload {load}\n" + t.round(3).to_string())
allJ = J(df[df.strategy.isin(order)])
print("\n== Test-set J over all loads (lower is better; equal weights)\n" + allJ.reindex([s for s in order if s in allJ.index]).round(4).to_string())
print("\n== Overall means\n" + df[df.strategy.isin(order)].groupby("strategy")[M].mean().reindex([s for s in order if s in allJ.index]).round(3).to_string())

pairs = [(prefix, "agentic-nollm"), (prefix, "priority-gated"), (prefix, "random-rank"),
         ("random-rank", "agentic-nollm")]
print("\n== Paired differences A - B: mean [95% window-cluster CI] (Wilcoxon p on 8 window means)")
for load, g in list(df.groupby("load")) + [("all", df)]:
    print(f"\nload {load}")
    for a, b in pairs:
        if not {a, b} <= set(g.strategy):
            continue
        d = paired(g, a, b)
        cells = []
        for m in M:
            dd = d[["window_id", m]].rename(columns={m: "diff"}).dropna()
            lo, hi = boot_cluster(dd)
            wm = dd.groupby("window_id")["diff"].mean()
            p = wilcoxon(wm).pvalue if (wm != 0).sum() >= 1 else float("nan")
            nb = int((wm < 0).sum())
            cells.append(f"{SHORT[m]}={dd['diff'].mean():+.2f}[{lo:+.2f},{hi:+.2f}] p={p:.3f} lower_in={nb}/8")
        print(f"  {a} - {b}:\n    " + "\n    ".join(cells))
