"""Policy engine: the safe-action constraints of the control plane.

The same rule set is used (a) by the Agentic AI OS to *block* unsafe actions
before execution and (b) by the simulator to *audit* every executed start of
every strategy, so the policy-violation rate is comparable across strategies.

Rules (a start action of task x at time t violates ...):
  P1_capacity     x does not fit in free CPU/memory
  P2_retry_budget x has already been retried more than `max_retries` times
  P3_starvation   x is not itself starving, while another queued task that fits
                  has waited longer than `starvation_s`
  P4_user_share   x's user already holds CPU and would exceed `user_cpu_cap`
                  of capacity while another user's queued task fits
"""
from __future__ import annotations

RULES = ("P1_capacity", "P2_retry_budget", "P3_starvation", "P4_user_share")


class PolicyEngine:
    def __init__(self, starvation_s: float, user_cpu_cap: float, max_retries: int):
        self.starvation_s = starvation_s
        self.user_cpu_cap = user_cpu_cap
        self.max_retries = max_retries

    def is_starving(self, task, t: float) -> bool:
        return t - task.enqueue_time > self.starvation_s

    def check(self, sim, task, t: float) -> list[str]:
        cl = sim.cluster
        v = []
        if not cl.fits(task):
            v.append("P1_capacity")
        if task.retries > self.max_retries:
            v.append("P2_retry_budget")
        if not self.is_starving(task, t):
            # queue is in enqueue order, so starving tasks form a prefix
            for y in sim.queue:
                if not self.is_starving(y, t):
                    break
                if y is not task and cl.fits(y):
                    v.append("P3_starvation")
                    break
        held = cl.user_cpu.get(task.user, 0.0)
        if held > 1e-12 and (held + task.cpu) / cl.cap_cpu > self.user_cpu_cap:
            if any(y.user != task.user and cl.fits(y) for y in sim.queue):
                v.append("P4_user_share")
        return v
