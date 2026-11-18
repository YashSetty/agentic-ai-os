"""Discrete-event trace-replay cluster simulator (shared by every strategy).

Model: one pooled cluster with fixed CPU and memory capacity (homogeneous nodes),
non-preemptive execution, trace-derived failures. All strategies see the same
tasks, capacity, failure realisations (seeded per task/attempt) and metric code.

Failure model:
  * a task whose trace event is FAIL / EVICT / LOST fails on its first attempt
    after a seeded fraction U(0.1, 0.9) of its runtime; each retry fails again
    with probability `p_refail`
  * KILL tasks are terminated by the user on their first attempt (no recovery)
  * a failure is detected `heartbeat_s` later; the strategy's on_failure() then
    decides retry or abandon. Recovery time = failure -> restart of the task.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .policy import PolicyEngine, RULES

RECOVERABLE = ("FAIL", "EVICT", "LOST")


@dataclass
class Task:
    idx: int
    tid: str
    user: str
    priority: int
    sched_class: int
    cpu: float
    mem: float
    duration: float
    submit: float
    fail_kind: str | None
    state: str = "pending"  # pending|queued|running|failed|done|killed|abandoned
    enqueue_time: float = 0.0
    attempts: int = 0
    retries: int = 0
    first_start: float | None = None
    finish: float | None = None
    last_fail: float | None = None
    recovery_delays: list = field(default_factory=list)


class Cluster:
    def __init__(self, cap_cpu: float, cap_mem: float):
        self.cap_cpu, self.cap_mem = cap_cpu, cap_mem
        self.used_cpu = self.used_mem = 0.0
        self.user_cpu: dict[str, float] = {}
        self._last_t = 0.0
        self.cpu_integral = self.mem_integral = 0.0

    def advance(self, t: float):
        dt = t - self._last_t
        if dt > 0:
            self.cpu_integral += self.used_cpu * dt
            self.mem_integral += self.used_mem * dt
            self._last_t = t

    def fits(self, task: Task) -> bool:
        eps = 1e-12
        return (self.used_cpu + task.cpu <= self.cap_cpu + eps
                and self.used_mem + task.mem <= self.cap_mem + eps)

    def alloc(self, task: Task):
        self.used_cpu += task.cpu
        self.used_mem += task.mem
        self.user_cpu[task.user] = self.user_cpu.get(task.user, 0.0) + task.cpu

    def release(self, task: Task):
        # snap float residue (e.g. 8.7e-19) to zero so policy rules see "nothing held"
        snap = lambda x: 0.0 if x < 1e-12 else x
        self.used_cpu = snap(self.used_cpu - task.cpu)
        self.used_mem = snap(self.used_mem - task.mem)
        self.user_cpu[task.user] = snap(self.user_cpu.get(task.user, 0.0) - task.cpu)


class Simulator:
    ARRIVAL, END, DETECT, WAKE = 0, 1, 2, 3

    def __init__(self, task_dicts: list[dict], cap_cpu: float, cap_mem: float, cfg: dict, seed: int):
        self.tasks = [Task(idx=i, **d) for i, d in enumerate(task_dicts)]
        self.cluster = Cluster(cap_cpu, cap_mem)
        self.policy = PolicyEngine(**cfg["policy"])
        self.heartbeat_s = cfg["heartbeat_s"]
        self.p_refail = cfg["p_refail"]
        self.seed = seed
        self.queue: list[Task] = []  # kept in enqueue-time order
        self.now = 0.0
        self._heap: list = []
        self._seq = 0
        self.busy_until = 0.0
        self._wake_at: float | None = None
        self.n_starts = 0
        self.violations = {r: 0 for r in RULES}
        self.n_violating_starts = 0
        self.decision_latencies: list[float] = []
        self.start_sources: dict[str, int] = {}

    # ---- event plumbing -------------------------------------------------
    def _push(self, t, kind, task=None, payload=None):
        self._seq += 1
        heapq.heappush(self._heap, (t, self._seq, kind, task, payload))

    def enqueue(self, task: Task, t: float):
        task.state = "queued"
        task.enqueue_time = t
        self.queue.append(task)

    def start(self, task: Task, t: float, latency: float = 0.0, source: str = "heuristic"):
        """Execute a start action. Every start is audited against the same
        policy rules regardless of strategy (baselines are not blocked)."""
        assert task.state == "queued" and self.cluster.fits(task), "infeasible start"
        viol = self.policy.check(self, task, t)
        for r in viol:
            self.violations[r] += 1
        self.n_violating_starts += bool(viol)
        self.n_starts += 1
        self.start_sources[source] = self.start_sources.get(source, 0) + 1

        self.queue.remove(task)
        self.cluster.alloc(task)  # reserved from decision time
        task.state = "running"
        task.attempts += 1
        begin = t + latency
        if task.first_start is None:
            task.first_start = begin
        if task.last_fail is not None and len(task.recovery_delays) < task.retries:
            task.recovery_delays.append(begin - task.last_fail)

        rng = np.random.default_rng([self.seed, task.idx, task.attempts])
        u, frac = rng.random(), rng.uniform(0.1, 0.9)
        fails = False
        if task.fail_kind == "KILL" and task.attempts == 1:
            fails = True
        elif task.fail_kind in RECOVERABLE and (task.attempts == 1 or u < self.p_refail):
            fails = True
        run = task.duration * frac if fails else task.duration
        self._push(begin + run, self.END, task, fails)

    # ---- main loop ------------------------------------------------------
    def run(self, scheduler) -> "Simulator":
        for task in self.tasks:
            self._push(task.submit, self.ARRIVAL, task)
        while self._heap:
            t = self._heap[0][0]
            self.cluster.advance(t)
            self.now = t
            while self._heap and self._heap[0][0] == t:
                _, _, kind, task, payload = heapq.heappop(self._heap)
                self._handle(kind, task, payload, t, scheduler)
            if self.queue:
                if t >= self.busy_until:
                    latency = scheduler.decide(self, t) or 0.0
                    if latency > 0:
                        self.decision_latencies.append(latency)
                        self.busy_until = t + latency
                        self._schedule_wake(self.busy_until)
                else:
                    self._schedule_wake(self.busy_until)
        self.cluster.advance(self.now)
        stuck = [x for x in self.tasks if x.state in ("queued", "pending", "running", "failed")]
        assert not stuck, f"{len(stuck)} tasks never finished"
        return self

    def _schedule_wake(self, t):
        if self._wake_at is None or self._wake_at <= self.now or t < self._wake_at:
            self._wake_at = t
            self._push(t, self.WAKE)

    def _handle(self, kind, task, payload, t, scheduler):
        if kind == self.ARRIVAL:
            self.enqueue(task, t)
        elif kind == self.END:
            self.cluster.release(task)
            if not payload:
                task.state, task.finish = "done", t
            elif task.fail_kind == "KILL":
                task.state, task.finish = "killed", t
            else:
                task.state, task.last_fail = "failed", t
                scheduler.observe_failure(self, task, t)
                self._push(t + self.heartbeat_s, self.DETECT, task)
        elif kind == self.DETECT:
            if scheduler.on_failure(self, task, t) == "retry":
                task.retries += 1
                self.enqueue(task, t)
            else:
                task.state, task.finish = "abandoned", t
        # WAKE: no-op, just triggers a decision
