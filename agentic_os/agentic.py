"""Agentic AI Operating System: policy-governed LLM control plane.

Components (paper section "Core Components"):
  StateMemoryManager  shared state snapshot + short-term memory of failures
  LLMOrchestrator     asks the LLM to rank candidate actions (never executes)
  PolicyEngine        safe-action constraints (agentic_os.policy), blocks unsafe actions
  ToolInterface       the only component that executes actions (sim.start)
  RecoveryAgent       decides retry vs abandon for failed tasks (retry budget)
  Monitor/audit log   every decision, proposal, rejection and execution is logged

Control loop at each decision epoch t (Algorithm 1):
  1. snapshot state s_t; build feasible candidate set A_t (queued tasks that fit)
  2. LLM ranks a bounded subset of A_t                      -> proposed order
  3. each proposal is checked by the policy engine          -> approve / reject
  4. approved actions execute through the tool interface (after the LLM's
     measured decision latency)
  5. deterministic backfill: remaining capacity is filled in a fixed safe order
     (retry > starving > priority > age), also policy-checked, so the plane is
     work-conserving even when the LLM output is empty or invalid
`use_llm=False` skips step 2 (ablation "agentic-nollm").
"""
from __future__ import annotations

import json
from collections import deque

SYSTEM_PROMPT = """You are the LLM orchestrator in the control plane of an enterprise compute cluster.
You only RANK queued tasks. A deterministic policy engine validates every choice and a tool layer executes approved ones.
Objectives, most important first:
1. Recovery: tasks with retry>0 failed earlier; restart them quickly.
2. Starvation: tasks whose waited_s exceeds {starve}s must run before others.
3. Priority: a higher prio number is more important (>=120 production, 100-119 batch, <100 free tier).
4. Fairness: do not let one user exceed {cap}% of cluster CPU (column user_cpu%).
5. Efficiency: pack CPU and memory well; when other factors tie, prefer smaller requests.
Reply ONLY with JSON of the form {{"order": ["c3", "c1", "c2"]}} listing candidate ids best-first. Omit tasks that should wait."""


class StateMemoryManager:
    def __init__(self, failure_window_s: float = 600.0):
        self.failures: deque = deque()
        self.failure_window_s = failure_window_s

    def record_failure(self, task, t):
        self.failures.append((t, task.user))

    def snapshot(self, sim, t) -> dict:
        while self.failures and self.failures[0][0] < t - self.failure_window_s:
            self.failures.popleft()
        cl = sim.cluster
        return {
            "t": t,
            "free_cpu_pct": 100 * (1 - cl.used_cpu / cl.cap_cpu),
            "free_mem_pct": 100 * (1 - cl.used_mem / cl.cap_mem),
            "queued": len(sim.queue),
            "running": sum(x.state == "running" for x in sim.tasks),
            "recent_failures": len(self.failures),
        }


class RecoveryAgent:
    def __init__(self, max_retries: int):
        self.max_retries = max_retries

    def on_failure(self, task) -> str:
        return "retry" if task.retries < self.max_retries else "abandon"


class LLMOrchestrator:
    def __init__(self, client, policy, max_candidates: int, runtime_hint: bool):
        self.client = client
        self.max_candidates = max_candidates
        self.runtime_hint = runtime_hint
        self.system = SYSTEM_PROMPT.format(starve=int(policy.starvation_s),
                                           cap=int(100 * policy.user_cpu_cap))

    def build_prompt(self, snap, cands, sim) -> str:
        cl = sim.cluster
        head = (f"t={snap['t']:.0f}s free_cpu={snap['free_cpu_pct']:.1f}% free_mem={snap['free_mem_pct']:.1f}% "
                f"queued={snap['queued']} running={snap['running']} "
                f"failures_last_10min={snap['recent_failures']}\n")
        cols = "id prio class cpu% mem% waited_s retry user_cpu%" + (" runtime_s" if self.runtime_hint else "")
        lines = []
        for i, x in enumerate(cands, 1):
            row = (f"c{i} {x.priority} {x.sched_class} {100 * x.cpu / cl.cap_cpu:.2f} "
                   f"{100 * x.mem / cl.cap_mem:.2f} {snap['t'] - x.enqueue_time:.0f} {x.retries} "
                   f"{100 * cl.user_cpu.get(x.user, 0.0) / cl.cap_cpu:.1f}")
            if self.runtime_hint:
                row += f" {x.duration:.0f}"
            lines.append(row)
        return head + "Candidates:\n" + cols + "\n" + "\n".join(lines)

    def rank(self, snap, cands, sim):
        prompt = self.build_prompt(snap, cands, sim)
        content, latency, cached = self.client.chat(self.system, prompt)
        idmap = {f"c{i}": x for i, x in enumerate(cands, 1)}
        order, invalid, parse_ok = [], 0, True
        try:
            obj = json.loads(content)
            raw = obj.get("order", []) if isinstance(obj, dict) else obj
            if not isinstance(raw, list):
                raise ValueError
        except (ValueError, AttributeError):
            raw, parse_ok = [], False
        seen = set()
        for r in raw:
            k = str(r).strip().lower()
            if k.isdigit():
                k = "c" + k
            if k in idmap and k not in seen:
                seen.add(k)
                order.append(idmap[k])
            else:
                invalid += 1
        return order, latency, {"cached": cached, "parse_ok": parse_ok, "invalid_ids": invalid,
                                "prompt": prompt, "response": content}


class AgenticScheduler:
    def __init__(self, cfg: dict, policy, client=None, use_llm: bool = True, audit_path: str | None = None):
        llm = cfg["llm"]
        self.name = "agentic" if use_llm else "agentic-nollm"
        self.use_llm = use_llm
        self.memory = StateMemoryManager()
        self.recovery = RecoveryAgent(policy.max_retries)
        self.policy = policy
        self.fixed_latency = None if llm["decision_latency"] == "measured" else float(llm["decision_latency"])
        self.orch = LLMOrchestrator(client, policy, llm["max_candidates"], llm["runtime_hint"]) if use_llm else None
        self.audit = open(audit_path, "w", encoding="utf-8") if audit_path else None
        self.stats = {"llm_calls": 0, "llm_cache_hits": 0, "llm_parse_failures": 0, "llm_invalid_ids": 0,
                      "proposals": 0, "proposals_rejected": 0, "proposals_deferred_capacity": 0,
                      "abandoned_by_policy": 0}
        self.rejected_by_rule: dict[str, int] = {}

    # --- recovery -------------------------------------------------------
    def observe_failure(self, sim, task, t):
        self.memory.record_failure(task, t)

    def on_failure(self, sim, task, t):
        decision = self.recovery.on_failure(task)
        if decision == "abandon":
            self.stats["abandoned_by_policy"] += 1
        self._log({"t": t, "type": "recovery", "task": task.tid, "retries": task.retries, "decision": decision})
        return decision

    # --- scheduling -----------------------------------------------------
    def _safe_order(self, queue, t):
        return sorted(queue, key=lambda x: (x.retries == 0, not self.policy.is_starving(x, t),
                                            -x.priority, x.enqueue_time))

    def _candidates(self, sim, t):
        fit = [x for x in sim.queue if sim.cluster.fits(x)]
        fit.sort(key=lambda x: (x.retries == 0, not self.policy.is_starving(x, t), x.enqueue_time))
        return fit[: self.orch.max_candidates] if self.orch else fit

    def decide(self, sim, t):
        cands = self._candidates(sim, t)
        if not cands:
            return 0.0
        latency, proposed, info = 0.0, [], None
        if self.use_llm and len(cands) > 1:
            snap = self.memory.snapshot(sim, t)
            proposed, latency, info = self.orch.rank(snap, cands, sim)
            self.stats["llm_calls"] += 1
            self.stats["llm_cache_hits"] += info["cached"]
            self.stats["llm_parse_failures"] += not info["parse_ok"]
            self.stats["llm_invalid_ids"] += info["invalid_ids"]
            if self.fixed_latency is not None:
                latency = self.fixed_latency

        executed, rejected = [], []
        for x in proposed:  # policy gate on LLM proposals
            self.stats["proposals"] += 1
            v = self.policy.check(sim, x, t) if x.state == "queued" else ["not_queued"]
            if v == ["P1_capacity"]:  # fits alone, but capacity already used by earlier picks
                self.stats["proposals_deferred_capacity"] += 1
                continue
            if v:
                self.stats["proposals_rejected"] += 1
                for r in v:
                    self.rejected_by_rule[r] = self.rejected_by_rule.get(r, 0) + 1
                rejected.append((x.tid, v))
                continue
            sim.start(x, t, latency, "llm")  # tool interface
            executed.append((x.tid, "llm"))
        for x in self._safe_order(list(sim.queue), t):  # deterministic safe backfill
            if x.state == "queued" and sim.cluster.fits(x) and not self.policy.check(sim, x, t):
                sim.start(x, t, latency, "backfill")
                executed.append((x.tid, "backfill"))
        self._log({"t": t, "type": "decision", "n_candidates": len(cands), "latency_s": latency,
                   "llm": None if info is None else {k: info[k] for k in ("cached", "parse_ok", "invalid_ids", "response")},
                   "rejected": rejected, "executed": executed})
        return latency

    def _log(self, rec):
        if self.audit:
            self.audit.write(json.dumps(rec) + "\n")

    def close(self):
        if self.audit:
            self.audit.close()

    def extra_metrics(self) -> dict:
        s = dict(self.stats)
        s["proposal_rejection_pct"] = 100 * s["proposals_rejected"] / max(s["proposals"], 1)
        for r, n in self.rejected_by_rule.items():
            s[f"rejected_{r}"] = n
        return s
