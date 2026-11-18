"""Trace loading, cleaning, and window selection for the Google 2019 Cluster Sample.

Dataset: Kaggle "derrickmwiti/google-2019-cluster-sample" (borg_traces_data.csv),
a sample of Google ClusterData2019 v3 (8 Borg cells, May 2019).

How a trace row becomes a simulated task (documented assumptions):
  * one task per (collection_id, instance_index); the earliest event row is kept
  * arrival  = the row's event `time` (microseconds -> seconds); rows with time 0
    (event before trace start) or the int64 sentinel are dropped
  * duration = end_time - start_time of the row's usage window, clipped to
    [1, max_duration_s]. The sample joins events to usage windows, so this is a
    proxy for runtime, not the task's true lifetime.
  * cpu/mem  = resource_request (normalised to the largest machine)
  * failure  = the row's event if it is FAIL / EVICT / LOST / KILL
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAILURE_EVENTS = ("FAIL", "EVICT", "LOST", "KILL")
TRACE_HOURS = 744  # May 2019 = 31 days
_USECOLS = [
    "time", "collection_id", "instance_index", "cluster", "event", "priority",
    "scheduling_class", "user", "resource_request", "start_time", "end_time",
]


def load_trace(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=_USECOLS)
    n_raw = len(df)
    df = df[(df["time"] > 0) & (df["time"] < 2**62)].copy()
    req = df["resource_request"].astype(str).str.extract(
        r"'cpus':\s*([0-9.eE+-]+|None).*?'memory':\s*([0-9.eE+-]+|None)")
    df["cpu"] = pd.to_numeric(req[0], errors="coerce")
    df["mem"] = pd.to_numeric(req[1], errors="coerce")
    df = df[(df["cpu"] > 0) & (df["mem"] > 0)]
    df["arrival_s"] = df["time"] / 1e6
    df["usage_s"] = (df["end_time"] - df["start_time"]) / 1e6
    df = df.sort_values("time", kind="stable")
    df = df.drop_duplicates(["collection_id", "instance_index"], keep="first")
    df["task_id"] = df["collection_id"].astype(str) + "-" + df["instance_index"].astype(str)
    df["fail_kind"] = df["event"].where(df["event"].isin(FAILURE_EVENTS), None)
    df.attrs["n_raw_rows"] = n_raw
    return df.reset_index(drop=True)


def build_windows(df: pd.DataFrame, cells, windows_per_cell: int, window_hours: float,
                  min_tasks: int) -> pd.DataFrame:
    """Deterministically pick `windows_per_cell` evenly spaced windows per cell
    (skipping the first day) among those with at least `min_tasks` tasks."""
    w = window_hours * 3600.0
    rows = []
    for cell in cells:
        sub = df[df["cluster"] == cell]
        starts = np.arange(24 * 3600.0, TRACE_HOURS * 3600.0 - w + 1, w)
        counts = [((sub["arrival_s"] >= s) & (sub["arrival_s"] < s + w)).sum() for s in starts]
        valid = [s for s, c in zip(starts, counts) if c >= min_tasks]
        if not valid:
            continue
        # interior points of an even split, e.g. k=1 -> the middle valid window
        pos = np.linspace(0, len(valid) - 1, windows_per_cell + 2)[1:-1]
        picks = sorted({int(round(i)) for i in pos})
        for k in picks:
            s = valid[k]
            win = sub[(sub["arrival_s"] >= s) & (sub["arrival_s"] < s + w)]
            ev = win["fail_kind"].value_counts()
            rows.append({
                "window_id": f"cell{cell}_h{int(s // 3600)}",
                "cell": cell, "start_s": s, "end_s": s + w, "n_tasks": len(win),
                "n_users": win["user"].nunique(),
                **{f"n_{k.lower()}": int(ev.get(k, 0)) for k in FAILURE_EVENTS},
            })
    return pd.DataFrame(rows)


def tasks_for_window(df: pd.DataFrame, window: dict, max_duration_s: float) -> list[dict]:
    win = df[(df["cluster"] == window["cell"]) & (df["arrival_s"] >= window["start_s"])
             & (df["arrival_s"] < window["end_s"])]
    out = []
    for r in win.itertuples(index=False):
        out.append({
            "tid": r.task_id, "user": r.user, "priority": int(r.priority),
            "sched_class": int(r.scheduling_class), "cpu": float(r.cpu), "mem": float(r.mem),
            "duration": float(np.clip(r.usage_s, 1.0, max_duration_s)),
            "submit": float(r.arrival_s - window["start_s"]),
            "fail_kind": r.fail_kind if isinstance(r.fail_kind, str) else None,
        })
    return out


def calibrate_capacity(tasks: list[dict], target_load: float, window_s: float) -> tuple[float, float]:
    """Capacity such that offered work / (capacity * window) == target_load,
    never smaller than the largest single request (so every task can run)."""
    cpu_work = sum(t["cpu"] * t["duration"] for t in tasks)
    mem_work = sum(t["mem"] * t["duration"] for t in tasks)
    cap_cpu = max(cpu_work / (target_load * window_s), max(t["cpu"] for t in tasks))
    cap_mem = max(mem_work / (target_load * window_s), max(t["mem"] for t in tasks))
    return cap_cpu, cap_mem
