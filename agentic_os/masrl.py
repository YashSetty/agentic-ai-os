"""MAS-RL: matched adaptive comparator (paper Section 5.5, Supplement S2).

MAS-RL is the Agentic AI OS control plane with exactly one component changed:
the LLM Orchestrator's ranking of the candidate set A_t is replaced by a learned
policy pi_phi(a | s_t). Everything else is shared code: the same candidate set
(the same bounded subset the LLM ranks), state snapshot, policy gate (P1-P4),
tool interface, retry-budget recovery agent, deterministic safe backfill,
simulator, trace windows, capacity calibration and metric code.

Policy. Each candidate x gets a score  phi . [f(x), f(x) (x) s_t]  where f(x) are
candidate features and s_t the state vector of Supplement S2 (outer product, so
the ranking is state-dependent). Candidates are proposed in descending score
(ties: enqueue time); proposals then pass through the unchanged policy gate.
Decision latency is 0 s (inference is a dot product).

State s_t (S2): normalized queue length, free CPU and memory, aggregate queued
CPU/memory demand, queue-age statistics (mean, max), priority-class mix
(production share), recent utilisation, recent failure count, recovery backlog.

Training reward is -J (paper Section 4.6), computed per episode; see
scripts/train_masrl.py for the leakage-free protocol.
"""
from __future__ import annotations

import json

import numpy as np

from agentic_os.agentic import AgenticScheduler

CAND_FEATURES = ["bias", "is_retry", "is_starving", "prio", "is_prod", "sched_class",
                 "cpu_share", "mem_share", "waited", "user_cpu_share"]
STATE_FEATURES = ["one", "queue_len", "free_cpu", "free_mem", "queued_cpu", "queued_mem",
                  "age_mean", "age_max", "prod_mix", "util_now", "recent_failures", "recovery_backlog"]
N_PARAMS = len(CAND_FEATURES) * len(STATE_FEATURES)


class MASRLScheduler(AgenticScheduler):
    """Agentic control plane with the LLM ranking replaced by a learned linear policy."""

    def __init__(self, cfg: dict, policy, theta, audit_path: str | None = None, name: str = "mas-rl"):
        super().__init__(cfg, policy, None, use_llm=False, audit_path=audit_path)
        self.name = name
        self.max_candidates = cfg["llm"]["max_candidates"]  # same bounded A_t as the LLM
        self.W = np.asarray(theta, dtype=float).reshape(len(CAND_FEATURES), len(STATE_FEATURES))
        self.stats.update(policy_calls=0)

    # identical candidate construction to the LLM variant (retry > starving > age, top-K)
    def _candidates(self, sim, t):
        fit = [x for x in sim.queue if sim.cluster.fits(x)]
        fit.sort(key=lambda x: (x.retries == 0, not self.policy.is_starving(x, t), x.enqueue_time))
        return fit[: self.max_candidates]

    def state_vector(self, sim, t) -> np.ndarray:
        snap = self.memory.snapshot(sim, t)
        cl, q = sim.cluster, sim.queue
        ages = np.array([t - x.enqueue_time for x in q]) if q else np.zeros(1)
        st = self.policy.starvation_s
        return np.array([
            1.0,
            np.log1p(len(q)) / 5.0,
            snap["free_cpu_pct"] / 100.0,
            snap["free_mem_pct"] / 100.0,
            min(sum(x.cpu for x in q) / cl.cap_cpu, 5.0) / 5.0,
            min(sum(x.mem for x in q) / cl.cap_mem, 5.0) / 5.0,
            min(ages.mean() / st, 4.0) / 4.0,
            min(ages.max() / st, 4.0) / 4.0,
            np.mean([x.priority >= 120 for x in q]) if q else 0.0,
            0.5 * (cl.used_cpu / cl.cap_cpu + cl.used_mem / cl.cap_mem),
            np.log1p(snap["recent_failures"]) / 3.0,
            np.log1p(sum(x.retries > 0 for x in q)) / 3.0,
        ])

    def candidate_matrix(self, sim, cands, t) -> np.ndarray:
        cl, st = sim.cluster, self.policy.starvation_s
        return np.array([[
            1.0,
            float(x.retries > 0),
            float(self.policy.is_starving(x, t)),
            x.priority / 400.0,
            float(x.priority >= 120),
            x.sched_class / 3.0,
            x.cpu / cl.cap_cpu,
            x.mem / cl.cap_mem,
            min((t - x.enqueue_time) / st, 4.0) / 4.0,
            cl.user_cpu.get(x.user, 0.0) / cl.cap_cpu,
        ] for x in cands])

    def rank(self, sim, cands, t):
        s = self.state_vector(sim, t)
        scores = self.candidate_matrix(sim, cands, t) @ (self.W @ s)
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], cands[i].enqueue_time))
        return [cands[i] for i in order]

    def decide(self, sim, t):
        cands = self._candidates(sim, t)
        if not cands:
            return 0.0
        proposed = self.rank(sim, cands, t) if len(cands) > 1 else []
        self.stats["policy_calls"] += len(cands) > 1
        executed, rejected = [], []
        for x in proposed:  # unchanged policy gate
            self.stats["proposals"] += 1
            v = self.policy.check(sim, x, t) if x.state == "queued" else ["not_queued"]
            if v == ["P1_capacity"]:
                self.stats["proposals_deferred_capacity"] += 1
                continue
            if v:
                self.stats["proposals_rejected"] += 1
                for r in v:
                    self.rejected_by_rule[r] = self.rejected_by_rule.get(r, 0) + 1
                rejected.append((x.tid, v))
                continue
            sim.start(x, t, 0.0, "policy")  # tool interface
            executed.append((x.tid, "policy"))
        for x in self._safe_order(list(sim.queue), t):  # unchanged deterministic safe backfill
            if x.state == "queued" and sim.cluster.fits(x) and not self.policy.check(sim, x, t):
                sim.start(x, t, 0.0, "backfill")
                executed.append((x.tid, "backfill"))
        self._log({"t": t, "type": "decision", "n_candidates": len(cands), "latency_s": 0.0,
                   "rejected": rejected, "executed": executed})
        return 0.0


def load_policy(path: str) -> np.ndarray:
    with open(path, encoding="utf-8") as f:
        return np.asarray(json.load(f)["theta"], dtype=float)


class RandomRankScheduler(MASRLScheduler):
    """Uninformed control: same interface, candidates proposed in uniformly random order
    (seeded per run), then the same policy gate and safe backfill."""

    def __init__(self, cfg: dict, policy, seed: int, audit_path: str | None = None):
        super().__init__(cfg, policy, np.zeros(N_PARAMS), audit_path=audit_path, name="random-rank")
        self.rng = np.random.default_rng([seed, 7919])

    def rank(self, sim, cands, t):
        return [cands[i] for i in self.rng.permutation(len(cands))]
