"""Evaluation metrics (identical code for every strategy).

avg_latency_s    mean(finish - submit) over completed tasks (includes queueing,
                 failed attempts and recovery)
util_cpu/mem     time-averaged allocated share of capacity over [0, makespan]
util_pct         mean of util_cpu and util_mem, in percent
avg_recovery_s   mean(failure time -> restart time) over all recoveries
violation_rate   executed starts violating >=1 policy rule / all starts, percent
fairness_jain    Jain's index over users of x_u = mean(duration / latency) of the
                 user's completed tasks: (sum x)^2 / (n * sum x^2), in (0, 1]
"""
from __future__ import annotations

import numpy as np


def compute(sim) -> dict:
    tasks = sim.tasks
    done = [t for t in tasks if t.state == "done"]
    lat = np.array([t.finish - t.submit for t in done]) if done else np.array([np.nan])
    wait = np.array([t.first_start - t.submit for t in tasks if t.first_start is not None])
    rec = np.array([d for t in tasks for d in t.recovery_delays])
    makespan = max(t.finish for t in tasks)
    cl = sim.cluster

    per_user: dict[str, list[float]] = {}
    for t in done:
        per_user.setdefault(t.user, []).append(t.duration / max(t.finish - t.submit, 1e-9))
    x = np.array([np.mean(v) for v in per_user.values()])
    jain = float(x.sum() ** 2 / (len(x) * (x ** 2).sum())) if len(x) else float("nan")

    util_cpu = cl.cpu_integral / (cl.cap_cpu * makespan)
    util_mem = cl.mem_integral / (cl.cap_mem * makespan)
    out = {
        "n_tasks": len(tasks),
        "n_completed": len(done),
        "n_killed": sum(t.state == "killed" for t in tasks),
        "n_abandoned": sum(t.state == "abandoned" for t in tasks),
        "avg_latency_s": float(np.nanmean(lat)),
        "p95_latency_s": float(np.nanpercentile(lat, 95)),
        "avg_wait_s": float(wait.mean()) if len(wait) else float("nan"),
        "util_cpu": util_cpu,
        "util_mem": util_mem,
        "util_pct": 100 * (util_cpu + util_mem) / 2,
        "n_recoveries": len(rec),
        "avg_recovery_s": float(rec.mean()) if len(rec) else float("nan"),
        "n_starts": sim.n_starts,
        "violation_rate_pct": 100 * sim.n_violating_starts / max(sim.n_starts, 1),
        "fairness_jain": jain,
        "makespan_s": makespan,
        "n_decisions_with_latency": len(sim.decision_latencies),
        "avg_decision_latency_s": float(np.mean(sim.decision_latencies)) if sim.decision_latencies else 0.0,
    }
    for rule, n in sim.violations.items():
        out[f"viol_{rule}"] = n
    for src, n in sim.start_sources.items():
        out[f"starts_{src}"] = n
    return out
