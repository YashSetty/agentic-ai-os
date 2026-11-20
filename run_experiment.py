"""Run the matched trace-replay experiment.

Every strategy is run on the same windows, capacities, failure seeds, policy
rules and metric code. Outputs (in --out):
  windows.csv       trace-window manifest (cell, time range, task/failure counts)
  per_run.csv       one row per (window, load, seed, strategy)
  summary.csv       mean and 95% bootstrap CI per strategy (and per load level)
  run_manifest.json config, dataset hash, LLM model digest, software versions
  audit/*.jsonl     decision audit logs of the agentic runs (with --audit)

Examples:
  python run_experiment.py --quick
  python run_experiment.py --config config.json --out results/full --audit
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

from agentic_os import metrics
from agentic_os.ablations import FIFOGated, NoLLMNoGate, NoLLMNoRecoveryOrder, PriorityGated
from agentic_os.agentic import AgenticScheduler
from agentic_os.llm import OllamaClient
from agentic_os.policy import PolicyEngine
from agentic_os.schedulers import FIFOScheduler, PriorityScheduler
from agentic_os.simulator import Simulator
from agentic_os.trace import build_windows, calibrate_capacity, load_trace, tasks_for_window

HEADLINE = ["avg_latency_s", "util_pct", "avg_recovery_s", "violation_rate_pct", "fairness_jain"]


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def acquire_lock(out_dir):
    """Refuse to run if another process is already writing to out_dir. The OS
    releases the lock automatically when the holding process exits."""
    f = open(os.path.join(out_dir, ".lock"), "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"Another run is already using {out_dir}. Wait for it to finish (or stop it) first.")
    return f


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key), n_runs=len(g))
        for m in HEADLINE + ["p95_latency_s", "n_completed", "n_abandoned"]:
            lo, hi = bootstrap_ci(g[m])
            row[m] = g[m].mean()
            row[m + "_ci_lo"], row[m + "_ci_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def make_scheduler(name, cfg, client, audit_path):
    if name == "fifo":
        return FIFOScheduler()
    if name == "priority":
        return PriorityScheduler()
    policy = PolicyEngine(**cfg["policy"])
    if name == "agentic":
        return AgenticScheduler(cfg, policy, client, use_llm=True, audit_path=audit_path)
    if name == "agentic-nollm":
        return AgenticScheduler(cfg, policy, None, use_llm=False, audit_path=audit_path)
    if name == "fifo-gated":
        return FIFOGated(policy)
    if name == "priority-gated":
        return PriorityGated(policy)
    if name == "nollm-noreco":
        return NoLLMNoRecoveryOrder(cfg, policy, audit_path=audit_path)
    if name == "nollm-nogate":
        return NoLLMNoGate(cfg, policy, audit_path=audit_path)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strategies", default="fifo,priority,agentic-nollm,agentic")
    ap.add_argument("--quick", action="store_true", help="1 short window, 1 load level (smoke test)")
    ap.add_argument("--audit", action="store_true", help="write per-decision audit logs")
    ap.add_argument("--keep-awake", action="store_true",
                    help="Windows: block idle sleep while this process runs (lid close still sleeps)")
    args = ap.parse_args()
    if args.keep_awake and sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS|SYSTEM_REQUIRED

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    if args.quick:
        cfg.update(cells=cfg["cells"][:1], windows_per_cell=1, window_hours=1, min_tasks=20,
                   load_levels=[0.9], seeds=cfg["seeds"][:1])
    out = args.out or ("results/quick" if args.quick else "results/full")
    os.makedirs(out, exist_ok=True)
    lock = acquire_lock(out)  # noqa: F841  (held until the process exits)
    if args.audit:
        os.makedirs(os.path.join(out, "audit"), exist_ok=True)
    strategies = args.strategies.split(",")
    dataset = os.path.expanduser(cfg["dataset_path"])

    client = None
    model_info = None
    if "agentic" in strategies:
        llm_cfg = dict(cfg["llm"])
        llm_cfg["cache_path"] = os.path.join(out, "llm_cache.jsonl")
        client = OllamaClient(**llm_cfg)
        model_info = client.model_info()
        client.warm_up()

    print("loading trace ...", flush=True)
    df = load_trace(dataset)
    windows = build_windows(df, cfg["cells"], cfg["windows_per_cell"], cfg["window_hours"], cfg["min_tasks"])
    windows.to_csv(os.path.join(out, "windows.csv"), index=False)
    print(f"{len(df)} tasks after cleaning; {len(windows)} windows", flush=True)

    manifest = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": cfg, "strategies": strategies,
        "dataset": {"path": dataset, "sha256": sha256_file(dataset), "raw_rows": df.attrs["n_raw_rows"],
                    "tasks_after_cleaning": len(df)},
        "llm_model": model_info,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    with open(os.path.join(out, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    per_run_path = os.path.join(out, "per_run.jsonl")  # append-only; resumable
    done_keys = set()
    if os.path.exists(per_run_path):
        with open(per_run_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_keys.add((r["window_id"], r["load"], r["seed"], r["strategy"]))

    window_s = cfg["window_hours"] * 3600
    for w in windows.to_dict("records"):
        tasks = tasks_for_window(df, w, cfg["max_duration_s"])
        for load in cfg["load_levels"]:
            cap_cpu, cap_mem = calibrate_capacity(tasks, load, window_s)
            for seed in cfg["seeds"]:
                for strat in strategies:
                    if (w["window_id"], load, seed, strat) in done_keys:
                        continue
                    audit = (os.path.join(out, "audit", f"{w['window_id']}_L{load}_s{seed}_{strat}.jsonl")
                             if args.audit and strat.startswith("agentic") else None)
                    sched = make_scheduler(strat, cfg, client, audit)
                    t0 = time.time()
                    sim = Simulator(copy.deepcopy(tasks), cap_cpu, cap_mem, cfg, seed).run(sched)
                    row = {"window_id": w["window_id"], "cell": w["cell"], "load": load, "seed": seed,
                           "strategy": strat, "cap_cpu": cap_cpu, "cap_mem": cap_mem,
                           **metrics.compute(sim), "wall_s": time.time() - t0}
                    if hasattr(sched, "extra_metrics"):
                        row.update(sched.extra_metrics())
                        sched.close()
                    with open(per_run_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, default=float) + "\n")
                    print(f"[{w['window_id']} load={load} seed={seed}] {strat:14s} "
                          f"lat={row['avg_latency_s']:8.1f}s util={row['util_pct']:5.1f}% "
                          f"rec={row['avg_recovery_s']:7.1f}s viol={row['violation_rate_pct']:5.2f}% "
                          f"fair={row['fairness_jain']:.3f} ({row['wall_s']:.0f}s)", flush=True)

    res = pd.read_json(per_run_path, lines=True)
    res = res[res["strategy"].isin(strategies)]
    res.to_csv(os.path.join(out, "per_run.csv"), index=False)
    res["strategy"] = pd.Categorical(res["strategy"], strategies, ordered=True)
    summ = summarise(res.sort_values("strategy"), ["strategy"])
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    summarise(res.sort_values(["load", "strategy"]), ["load", "strategy"]).to_csv(
        os.path.join(out, "summary_by_load.csv"), index=False)
    print("\n" + summ[["strategy", "n_runs"] + HEADLINE].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
