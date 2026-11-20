"""Evaluate frozen MAS-RL policies and the random-ranking control on the test grid.

Runs on exactly the windows, loads and failure seeds of config.grid.json (the
504-run grid): strategy "mas-rl-s<k>" for every policy file, and "random-rank".
Appends to <out>/per_run.jsonl (resumable), same row schema as run_experiment.py.

    python scripts/eval_masrl.py --config config.grid.json --policies results/masrl/train \
        --out results/masrl/test
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentic_os import metrics  # noqa: E402
from agentic_os.masrl import MASRLScheduler, RandomRankScheduler, load_policy  # noqa: E402
from agentic_os.policy import PolicyEngine  # noqa: E402
from agentic_os.simulator import Simulator  # noqa: E402
from agentic_os.trace import build_windows, calibrate_capacity, load_trace, tasks_for_window  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.grid.json")
    ap.add_argument("--policies", default="results/masrl/train")
    ap.add_argument("--out", default="results/masrl/test")
    ap.add_argument("--prefix", default="mas-rl")
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--trace-pickle", default=None)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    import pandas as pd
    df = pd.read_pickle(args.trace_pickle) if args.trace_pickle else load_trace(os.path.expanduser(cfg["dataset_path"]))
    windows = build_windows(df, cfg["cells"], cfg["windows_per_cell"], cfg["window_hours"], cfg["min_tasks"])

    pols = {}
    for p in sorted(glob.glob(os.path.join(args.policies, "policy_seed*.json"))):
        k = re.search(r"seed(\d+)", p).group(1)
        pols[f"{args.prefix}-s{k}"] = load_policy(p)
    strategies = list(pols) + ([] if args.no_random else ["random-rank"])
    print("strategies:", strategies, flush=True)

    path = os.path.join(args.out, "per_run.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["window_id"], r["load"], r["seed"], r["strategy"]))
    for w in windows.to_dict("records"):
        tasks = tasks_for_window(df, w, cfg["max_duration_s"])
        for load in cfg["load_levels"]:
            cap = calibrate_capacity(tasks, load, cfg["window_hours"] * 3600)
            for seed in cfg["seeds"]:
                for st in strategies:
                    if (w["window_id"], load, seed, st) in done:
                        continue
                    pol = PolicyEngine(**cfg["policy"])
                    sch = RandomRankScheduler(cfg, pol, seed) if st == "random-rank" else MASRLScheduler(cfg, pol, pols[st], name=st)
                    t0 = time.time()
                    sim = Simulator(copy.deepcopy(tasks), *cap, cfg, seed).run(sch)
                    row = {"window_id": w["window_id"], "cell": w["cell"], "load": load, "seed": seed,
                           "strategy": st, "cap_cpu": cap[0], "cap_mem": cap[1], **metrics.compute(sim),
                           "wall_s": time.time() - t0, **sch.extra_metrics()}
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, default=float) + "\n")
        print("done", w["window_id"], flush=True)


if __name__ == "__main__":
    main()
