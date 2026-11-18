"""Paper-ready tables and figures from a results directory.

Usage:  python analyze.py results/medium
Writes into <dir>/report/:
  table1.md / table1.csv     mean and 95% bootstrap CI per strategy (across windows)
  per_window.md              headline metrics per window and strategy
  paired.md                  paired differences between strategies (same window)
  llm_stats.md               LLM call counts, latency, proposal rejections
  fig_latency.png, fig_utilisation.png, fig_recovery.png, fig_latency_per_window.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

STRATS = ["fifo", "priority", "agentic-nollm", "agentic"]
LABEL = {"fifo": "FIFO", "priority": "Priority heuristic",
         "agentic-nollm": "Agentic OS (no LLM)", "agentic": "Agentic OS (LLM)"}
METRICS = [("avg_latency_s", "Avg. task completion latency (s)", "{:.1f}"),
           ("p95_latency_s", "P95 task completion latency (s)", "{:.1f}"),
           ("util_pct", "Resource utilisation (%)", "{:.1f}"),
           ("avg_recovery_s", "Failure recovery time (s)", "{:.1f}"),
           ("violation_rate_pct", "Policy violation rate (%)", "{:.2f}"),
           ("fairness_jain", "Scheduling fairness (Jain)", "{:.3f}")]


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt_ci(x, f):
    lo, hi = boot_ci(x)
    return f"{f.format(np.nanmean(x))} [{f.format(lo)}, {f.format(hi)}]"


def main(d):
    out = os.path.join(d, "report")
    os.makedirs(out, exist_ok=True)
    r = pd.read_json(os.path.join(d, "per_run.jsonl"), lines=True)
    r = r.drop_duplicates(["window_id", "load", "seed", "strategy"], keep="last")
    strats = [s for s in STRATS if s in set(r.strategy)]
    complete = r.groupby(["window_id", "load", "seed"]).strategy.nunique()
    keys = complete[complete == len(strats)].index
    r = r.set_index(["window_id", "load", "seed"]).loc[keys].reset_index()
    n_w = len(keys)

    # Table 1
    rows, csv = [], []
    for m, name, f in METRICS:
        cells = {s: fmt_ci(r[r.strategy == s][m], f) for s in strats}
        rows.append(f"| {name} | " + " | ".join(cells[s] for s in strats) + " |")
        for s in strats:
            x = r[r.strategy == s][m]
            lo, hi = boot_ci(x)
            csv.append({"metric": m, "strategy": s, "mean": x.mean(), "ci_lo": lo, "ci_hi": hi, "n": len(x)})
    head = "| Metric | " + " | ".join(LABEL[s] for s in strats) + " |\n|---|" + "---|" * len(strats)
    with open(os.path.join(out, "table1.md"), "w", encoding="utf-8") as fh:
        fh.write(f"Mean [95% bootstrap CI] over {n_w} runs (window x load; Google 2019 Cluster Sample, "
                 f"load {sorted(r.load.unique())}).\n\n{head}\n" + "\n".join(rows) + "\n")
    pd.DataFrame(csv).to_csv(os.path.join(out, "table1.csv"), index=False)

    # Per window
    pw = r.pivot_table(index=["window_id", "load"], columns="strategy",
                       values=["avg_latency_s", "avg_recovery_s", "violation_rate_pct"])
    with open(os.path.join(out, "per_window.md"), "w", encoding="utf-8") as fh:
        for m in ["avg_latency_s", "avg_recovery_s", "violation_rate_pct"]:
            t = pw[m][strats].rename(columns=LABEL).round(1)
            fh.write(f"### {m}\n\n{t.to_markdown()}\n\n")

    # Paired comparisons (same window, same seed)
    pairs = [("agentic", "agentic-nollm"), ("agentic", "priority"), ("agentic", "fifo"),
             ("agentic-nollm", "priority"), ("agentic-nollm", "fifo")]
    lines = ["Paired by window. diff = A - B (negative latency/recovery = A better). "
             "Bootstrap 95% CI of the mean difference; wins = windows where A is better.\n",
             "| A vs B | Metric | Mean diff [95% CI] | A wins |", "|---|---|---|---|"]
    idx = ["window_id", "load", "seed"]
    for a, b in pairs:
        if a not in strats or b not in strats:
            continue
        A = r[r.strategy == a].set_index(idx)
        B = r[r.strategy == b].set_index(idx).loc[A.index]
        for m, better in [("avg_latency_s", -1), ("avg_recovery_s", -1), ("util_pct", 1),
                          ("violation_rate_pct", -1), ("fairness_jain", 1)]:
            dlt = (A[m] - B[m]).to_numpy(float)
            lo, hi = boot_ci(dlt)
            wins = int(np.sum(np.sign(dlt) == better))
            lines.append(f"| {LABEL[a]} vs {LABEL[b]} | {m} | {np.nanmean(dlt):.2f} [{lo:.2f}, {hi:.2f}] | {wins}/{len(dlt)} |")
    with open(os.path.join(out, "paired.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # LLM stats
    if "agentic" in strats:
        a = r[r.strategy == "agentic"]
        s = {"windows": len(a), "llm_calls": int(a.llm_calls.sum()),
             "mean_decision_latency_s (per-window avg)": round(a.avg_decision_latency_s.mean(), 2),
             "proposals": int(a.proposals.sum()),
             "proposals_rejected_by_policy": int(a.proposals_rejected.sum()),
             "policy_rejection_pct": round(100 * a.proposals_rejected.sum() / max(a.proposals.sum(), 1), 2),
             "proposals_deferred_capacity": int(a.proposals_deferred_capacity.sum()),
             "parse_failures": int(a.llm_parse_failures.sum()), "invalid_ids": int(a.llm_invalid_ids.sum()),
             "starts_from_llm": int(a.starts_llm.sum()), "starts_from_backfill": int(a.starts_backfill.sum())}
        for c in [c for c in a.columns if c.startswith("rejected_P")]:
            s[c] = int(a[c].fillna(0).sum())
        with open(os.path.join(out, "llm_stats.md"), "w", encoding="utf-8") as fh:
            fh.write("| Statistic | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in s.items()) + "\n")

    # Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"fifo": "#9aa0a6", "priority": "#5f8dd3", "agentic-nollm": "#e0a030", "agentic": "#3a9e6f"}
    for m, name, fname in [("avg_latency_s", "Average task completion latency (s)", "fig_latency.png"),
                           ("util_pct", "Resource utilisation (%)", "fig_utilisation.png"),
                           ("avg_recovery_s", "Failure recovery time (s)", "fig_recovery.png")]:
        fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
        means = [r[r.strategy == s][m].mean() for s in strats]
        cis = [boot_ci(r[r.strategy == s][m]) for s in strats]
        err = np.array([[mu - lo, hi - mu] for mu, (lo, hi) in zip(means, cis)]).T
        ax.bar([LABEL[s] for s in strats], means, yerr=err, capsize=4, color=[colors[s] for s in strats])
        ax.set_ylabel(name)
        ax.set_title(f"{name}\n(mean, 95% CI over {n_w} runs)", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    r["run"] = (r.window_id if r.load.nunique() == 1 and r.seed.nunique() == 1
                else r.window_id + " L" + r.load.astype(str) + " s" + r.seed.astype(str))
    wins = list(dict.fromkeys(r.run))
    w = 0.8 / len(strats)
    for i, s in enumerate(strats):
        v = r[r.strategy == s].set_index("run").loc[wins, "avg_latency_s"]
        ax.bar(np.arange(len(wins)) + i * w, v, w, label=LABEL[s], color=colors[s])
    ax.set_xticks(np.arange(len(wins)) + w * (len(strats) - 1) / 2, wins, fontsize=7)
    ax.set_ylabel("Avg. latency (s)")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_latency_per_window.png"))
    plt.close(fig)
    print(open(os.path.join(out, "table1.md"), encoding="utf-8").read())
    print(open(os.path.join(out, "paired.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/medium")
