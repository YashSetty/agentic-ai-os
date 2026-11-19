"""Policy-matched baselines and control-plane ablations (paper Section 6.4).
Used by run_experiment.make_scheduler (strategy names below).

fifo-gated      FIFO order (no backfill past a head that does not fit); every start
                must pass the same policy engine (P1-P4); same retry budget as the
                agentic recovery agent (abandon after max_retries).
priority-gated  Priority order with greedy backfill; same policy gate and retry budget.
nollm-noreco    Agentic control plane without LLM and without recovery-first ordering
                (safe order = starving > priority > age).
nollm-nogate    Agentic control plane without LLM and without the policy gate
                (starts are only capacity-checked; still audited like the baselines).
"""
from agentic_os.agentic import AgenticScheduler, RecoveryAgent
from agentic_os.policy import PolicyEngine


class _Gated:
    def __init__(self, policy):
        self.policy = policy
        self.recovery = RecoveryAgent(policy.max_retries)
        self.abandoned_by_policy = 0

    def observe_failure(self, sim, task, t):
        pass

    def on_failure(self, sim, task, t):
        d = self.recovery.on_failure(task)
        self.abandoned_by_policy += d == "abandon"
        return d

    def extra_metrics(self):
        return {"abandoned_by_policy": self.abandoned_by_policy}

    def close(self):
        pass


class FIFOGated(_Gated):
    name = "fifo-gated"

    def decide(self, sim, t):
        for task in list(sim.queue):  # enqueue order
            if not sim.cluster.fits(task):
                break  # strict FIFO: no backfill past a head that does not fit
            if not self.policy.check(sim, task, t):
                sim.start(task, t, 0.0, "heuristic")
        return 0.0


class PriorityGated(_Gated):
    name = "priority-gated"

    def decide(self, sim, t):
        for task in sorted(sim.queue, key=lambda x: (-x.priority, x.enqueue_time)):
            if task.state == "queued" and sim.cluster.fits(task) and not self.policy.check(sim, task, t):
                sim.start(task, t, 0.0, "heuristic")
        return 0.0


class NoLLMNoRecoveryOrder(AgenticScheduler):
    def __init__(self, cfg, policy, **kw):
        super().__init__(cfg, policy, None, use_llm=False, **kw)
        self.name = "nollm-noreco"

    def _safe_order(self, queue, t):
        return sorted(queue, key=lambda x: (not self.policy.is_starving(x, t), -x.priority, x.enqueue_time))


class NoLLMNoGate(AgenticScheduler):
    def __init__(self, cfg, policy, **kw):
        super().__init__(cfg, policy, None, use_llm=False, **kw)
        self.name = "nollm-nogate"

    def decide(self, sim, t):
        for x in self._safe_order(list(sim.queue), t):
            if x.state == "queued" and sim.cluster.fits(x):
                sim.start(x, t, 0.0, "backfill")
        return 0.0
