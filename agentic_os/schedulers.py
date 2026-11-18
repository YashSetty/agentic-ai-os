"""Conventional baselines. Neither enforces policy rules (they are audited only)
and both always retry failed tasks (re-queued at detection time)."""
from __future__ import annotations


class FIFOScheduler:
    """Strict first-come-first-served: start the queue head while it fits;
    no backfilling past a head task that does not fit."""
    name = "fifo"

    def decide(self, sim, t):
        while sim.queue and sim.cluster.fits(sim.queue[0]):
            sim.start(sim.queue[0], t, 0.0, "heuristic")
        return 0.0

    def observe_failure(self, sim, task, t):
        pass

    def on_failure(self, sim, task, t):
        return "retry"


class PriorityScheduler(FIFOScheduler):
    """Highest Borg priority first (ties: earliest enqueued), greedy backfill of
    any task that fits."""
    name = "priority"

    def decide(self, sim, t):
        for task in sorted(sim.queue, key=lambda x: (-x.priority, x.enqueue_time)):
            if sim.cluster.fits(task):
                sim.start(task, t, 0.0, "heuristic")
        return 0.0
