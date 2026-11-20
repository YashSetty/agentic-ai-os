"""Train MAS-RL (Section 5.5 / Supplement S2) with REINFORCE, leakage-free.

Windows. For each cell, the valid 6-h windows that END before that cell's test
window (the window used in every reported experiment) are split chronologically:
the earlier ones are candidates for training, the later ones for validation.
`--train-per-cell` / `--val-per-cell` windows are taken evenly from each part.
Test windows are never loaded during training or model selection.

Episodes. A training episode is (training window, load in {0.7,0.9,1.1},
failure seed >= 1001), so failure realisations also differ from the test seeds 1-3.

Normalisation. Every J term is dimensionless and O(1): latency and recovery are
ratios to the same-episode reference, utilisation, violation share and Jain's
index lie in [0,1]. State/candidate features use fixed constants (capacity,
starvation threshold), never statistics of any window set.

Policy. Stochastic Plackett-Luce ranking over the candidate set:
order = argsort(score + Gumbel noise), score_i = f_i^T W s_t. The policy gate,
backfill, simulator and metrics are the unchanged shared code.

Reward. Episode reward = -J (paper Section 4.6); default equal weights 0.2 (--weights):
J = 0.2*(L/L_ref + (1-U) + R/R_ref + V + (1-F)), where L, R are mean latency and
recovery time, U utilisation (0-1), V violation fraction, F Jain fairness, and
L_ref, R_ref are those of the policy-gated priority baseline on the same episode
(a fixed normaliser, not a learned quantity).

Update. REINFORCE with batch-normalised advantages (batch of episodes),
log-likelihood of the ranking prefix up to the last executed proposal, Adam.
Every --val-every updates the deterministic policy is scored on the validation
set; the best validation checkpoint is the frozen policy.

    python scripts/train_masrl.py --config config.grid.json --seed 1 --out results/masrl/train
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentic_os import metrics  # noqa: E402
from agentic_os.ablations import PriorityGated  # noqa: E402
from agentic_os.masrl import CAND_FEATURES, N_PARAMS, STATE_FEATURES, MASRLScheduler  # noqa: E402
from agentic_os.policy import PolicyEngine  # noqa: E402
from agentic_os.simulator import Simulator  # noqa: E402
from agentic_os.trace import TRACE_HOURS, build_windows, calibrate_capacity, load_trace, tasks_for_window  # noqa: E402

LOADS = (0.7, 0.9, 1.1)
WEIGHTS = (0.2, 0.2, 0.2, 0.2, 0.2)


def split_windows(df, cfg, n_train, n_val, max_tasks):
    """Chronologically earlier train/val windows per cell, disjoint from the test window."""
    w = cfg["window_hours"] * 3600.0
    test = build_windows(df, cfg["cells"], cfg["windows_per_cell"], cfg["window_hours"], cfg["min_tasks"])
    train, val = [], []
    for r in test.to_dict("records"):
        sub = df[df["cluster"] == r["cell"]]
        starts = np.arange(24 * 3600.0, TRACE_HOURS * 3600.0 - w + 1, w)
        n = lambda s: ((sub["arrival_s"] >= s) & (sub["arrival_s"] < s + w)).sum()
        valid = [s for s in starts if s + w <= r["start_s"] and cfg["min_tasks"] <= n(s) <= max_tasks]
        cut = int(round(len(valid) * 2 / 3))
        pick = lambda xs, k: [xs[int(round(i))] for i in np.linspace(0, len(xs) - 1, k)] if xs else []
        for part, k, out in ((valid[:cut], n_train, train), (valid[cut:], n_val, val)):
            for s in sorted(set(pick(part, min(k, len(part))))):
                out.append({"window_id": f"cell{r['cell']}_h{int(s // 3600)}", "cell": r["cell"],
                            "start_s": s, "end_s": s + w})
    return test, train, val


def objective(m, ref):
    L = m["avg_latency_s"] / ref["avg_latency_s"]
    R = (m["avg_recovery_s"] / ref["avg_recovery_s"]
         if ref["n_recoveries"] and m["n_recoveries"] and ref["avg_recovery_s"] > 0 else 1.0)
    a, b, g, d, e = WEIGHTS
    return a * L + b * (1 - m["util_pct"] / 100) + g * R + d * m["violation_rate_pct"] / 100 + e * (1 - m["fairness_jain"])


class Episode:
    def __init__(self, df, win, load, seed, cfg):
        self.tasks = tasks_for_window(df, win, cfg["max_duration_s"])
        self.cap = calibrate_capacity(self.tasks, load, cfg["window_hours"] * 3600)
        self.seed, self.cfg = seed, cfg
        self.ref = self.run(PriorityGated(PolicyEngine(**cfg["policy"])))

    def run(self, sched):
        sim = Simulator(copy.deepcopy(self.tasks), *self.cap, self.cfg, self.seed).run(sched)
        return metrics.compute(sim)


class StochasticMASRL(MASRLScheduler):
    """Training-time policy: Gumbel (Plackett-Luce) sampling + log-likelihood gradient."""

    def __init__(self, cfg, policy, theta, rng):
        super().__init__(cfg, policy, theta)
        self.rng = rng
        self.grad = np.zeros_like(self.W)
        self._pending = None

    def rank(self, sim, cands, t):
        s = self.state_vector(sim, t)
        F = self.candidate_matrix(sim, cands, t)
        sc = F @ (self.W @ s)
        order = list(np.argsort(-(sc + self.rng.gumbel(size=len(sc)))))
        self._pending = (F, s, sc, order, cands)
        return [cands[i] for i in order]

    def decide(self, sim, t):
        self._pending = None
        lat = super().decide(sim, t)
        if self._pending is not None:
            F, s, sc, order, cands = self._pending
            started = [k for k, i in enumerate(order) if cands[i].state == "running"]
            if started:  # likelihood of the ranking prefix that determined the executed proposals
                last = started[-1]
                remaining = np.ones(len(order), bool)
                g = np.zeros(F.shape[1])
                for k in range(last + 1):
                    i = order[k]
                    z = np.where(remaining, sc, -np.inf)
                    p = np.exp(z - z.max())
                    p /= p.sum()
                    g += F[i] - p @ F
                    remaining[i] = False
                self.grad += np.outer(g, s)
        return lat


def evaluate(theta, episodes, cfg):
    js = []
    for ep in episodes:
        m = ep.run(MASRLScheduler(cfg, PolicyEngine(**cfg["policy"]), theta))
        js.append(objective(m, ep.ref))
    return float(np.mean(js))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.grid.json")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="results/masrl/train")
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--train-per-cell", type=int, default=4)
    ap.add_argument("--val-per-cell", type=int, default=2)
    ap.add_argument("--max-tasks", type=int, default=2000,
                    help="skip train/val windows with more tasks (bounds simulation cost)")
    ap.add_argument("--init-std", type=float, default=0.01)
    ap.add_argument("--weights", default="0.2,0.2,0.2,0.2,0.2",
                    help="J coefficients alpha,beta,gamma,delta,epsilon (latency,1-util,recovery,violation,1-fairness)")
    ap.add_argument("--trace-pickle", default=None, help="optional cached load_trace() output")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    global WEIGHTS
    WEIGHTS = tuple(float(x) for x in args.weights.split(","))
    assert len(WEIGHTS) == 5 and abs(sum(WEIGHTS) - 1) < 1e-9
    rng = np.random.default_rng(args.seed)

    import pandas as pd
    df = pd.read_pickle(args.trace_pickle) if args.trace_pickle else load_trace(os.path.expanduser(cfg["dataset_path"]))
    test, train_w, val_w = split_windows(df, cfg, args.train_per_cell, args.val_per_cell, args.max_tasks)
    test_ids = set(test["window_id"])
    assert not test_ids & {w["window_id"] for w in train_w + val_w}, "leakage"
    manifest = {"test": sorted(test_ids), "train": [w["window_id"] for w in train_w],
                "validation": [w["window_id"] for w in val_w]}
    with open(os.path.join(args.out, "window_split.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"seed {args.seed}: {len(train_w)} train / {len(val_w)} val windows; building episodes", flush=True)
    # validation set: every val window x 3 loads x 2 held-out failure seeds
    val_eps = [Episode(df, w, L, s, cfg) for w in val_w for L in LOADS for s in (901, 902)]
    ep_cache = {}

    def train_episode():
        w = train_w[rng.integers(len(train_w))]
        L = LOADS[rng.integers(3)]
        s = int(1001 + rng.integers(50))
        key = (w["window_id"], L, s)
        if key not in ep_cache:
            ep_cache[key] = Episode(df, w, L, s, cfg)
        return ep_cache[key]

    # seeded small random start (no hand-coded ordering prior; distinct per training seed)
    theta = rng.normal(0.0, args.init_std, N_PARAMS)
    m_adam, v_adam = np.zeros(N_PARAMS), np.zeros(N_PARAMS)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = {"val_J": evaluate(theta, val_eps, cfg), "update": 0, "theta": theta.tolist()}
    log = [{"update": 0, "val_J": best["val_J"]}]
    print(f"update 0 val J={best['val_J']:.4f}", flush=True)
    t0 = time.time()
    for u in range(1, args.updates + 1):
        grads, rets = [], []
        for _ in range(args.batch):
            ep = train_episode()
            sch = StochasticMASRL(cfg, PolicyEngine(**cfg["policy"]), theta, rng)
            m = ep.run(sch)
            grads.append(sch.grad.ravel())
            rets.append(-objective(m, ep.ref))
        rets = np.array(rets)
        adv = (rets - rets.mean()) / (rets.std() + 1e-8)
        g = np.mean([a * gr / max(np.linalg.norm(gr), 1.0) for a, gr in zip(adv, grads)], axis=0)
        m_adam = b1 * m_adam + (1 - b1) * g
        v_adam = b2 * v_adam + (1 - b2) * g * g
        theta = theta + args.lr * (m_adam / (1 - b1 ** u)) / (np.sqrt(v_adam / (1 - b2 ** u)) + eps)
        rec = {"update": u, "train_J": float(-rets.mean()), "elapsed_s": round(time.time() - t0)}
        if u % args.val_every == 0 or u == args.updates:
            rec["val_J"] = evaluate(theta, val_eps, cfg)
            if rec["val_J"] < best["val_J"]:
                best = {"val_J": rec["val_J"], "update": u, "theta": theta.tolist()}
            print(f"update {u} train J={rec['train_J']:.4f} val J={rec['val_J']:.4f} "
                  f"(best {best['val_J']:.4f} @ {best['update']}) {rec['elapsed_s']}s", flush=True)
        log.append(rec)
    out = {"seed": args.seed, "algorithm": "REINFORCE (Plackett-Luce ranking policy, Adam)",
           "hyperparameters": {k: getattr(args, k) for k in ("updates", "batch", "lr", "val_every",
                                                              "train_per_cell", "val_per_cell", "init_std", "max_tasks")},
           "reward_weights": WEIGHTS, "cand_features": CAND_FEATURES, "state_features": STATE_FEATURES,
           "best_update": best["update"], "best_val_J": best["val_J"], "theta": best["theta"],
           "final_theta": theta.tolist()}
    with open(os.path.join(args.out, f"policy_seed{args.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.out, f"log_seed{args.seed}.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in log)


if __name__ == "__main__":
    main()
