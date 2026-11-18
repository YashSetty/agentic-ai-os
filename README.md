# Agentic AI OS — trace-replay prototype

Reference implementation for the paper *"Agentic AI Operating Systems for Enterprise
Autonomy: A Policy-Governed LLM Control Plane for Workload Scheduling and Failure Recovery"*.

An LLM (local, via Ollama) **ranks** scheduling actions; a deterministic **policy engine**
approves or rejects them; a **tool interface** is the only component that executes them.
It is compared, under identical trace replay, with FIFO and priority schedulers.

## Layout

| File | Paper concept |
|---|---|
| `agentic_os/trace.py` | Dataset description, preprocessing, trace-window manifest, capacity calibration |
| `agentic_os/simulator.py` | Trace-replay cluster simulator (shared by all strategies), failure model |
| `agentic_os/policy.py` | Policy engine / safe-action constraints P1–P4 |
| `agentic_os/agentic.py` | State & Memory Manager, LLM Orchestrator, Recovery Agent, tool interface, audit log (Algorithm 1) |
| `agentic_os/llm.py` | Ollama client with response cache (reproducible replay) |
| `agentic_os/schedulers.py` | FIFO and priority baselines |
| `agentic_os/metrics.py` | Latency, utilisation, recovery time, violation rate, Jain fairness |
| `run_experiment.py` | Matched experiment runner, CSV outputs, bootstrap CIs, run manifest |
| `config.json` | All parameters (windows, load levels, seeds, policy thresholds, LLM settings) |

## Setup

```bash
pip install -r requirements.txt kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('derrickmwiti/google-2019-cluster-sample'))"
ollama pull llama3        # local LLM, served at http://localhost:11434
```

If the dataset is not at the default kagglehub cache path, set `dataset_path` in `config.json`.

## Run

```bash
python run_experiment.py --quick --audit          # smoke test: 1 window, 1 h, ~3 min
python run_experiment.py --config config.medium.json --out results/medium --audit --keep-awake   # paper run
python analyze.py results/medium                  # Table 1, paired comparisons, figures -> results/medium/report/
python run_experiment.py --strategies fifo,priority,agentic-nollm   # no LLM needed, seconds
python run_experiment.py --audit                  # all 3 load levels (0.7 / 0.9 / 1.1)
```

The run is resumable (rerun the same command; finished runs in `per_run.jsonl` are skipped).
A fresh paper run took about 7.5 h of LLM time on an RTX 3060 Laptop GPU. If the computer
sleeps during a call, the call is discarded and re-measured; `scripts/rerun_contaminated.py`
removes any agentic run whose audit log still shows a call above 60 s so that a resume redoes it.

### Reproducing the paper's numbers

`results/medium/` holds everything the paper reports. Three levels of reproduction:

1. **Tables and figures only** (no dataset, no LLM): `python analyze.py results/medium`
2. **Exact replay of all 32 runs** (dataset + Ollama running with `llama3`, takes minutes):
   the LLM response cache `llm_cache.jsonl` stores every prompt's response and measured latency,
   so a replay makes no new LLM calls and reproduces `per_run.jsonl` exactly.
   ```bash
   mkdir -p results/replay && cp results/medium/llm_cache.jsonl results/replay/
   python run_experiment.py --config config.medium.json --out results/replay
   ```
3. **Fresh run** (new LLM calls): use an empty output directory. Decision latencies depend on
   your hardware, and LLM outputs may differ slightly across Ollama/driver versions.

### Strategies

- `fifo` — strict first-come-first-served, no backfilling.
- `priority` — highest Borg priority first, greedy backfill.
- `agentic` — LLM-ranked actions → policy gate → tool execution → deterministic safe backfill.
- `agentic-nollm` — ablation: same control plane with the LLM removed (isolates the LLM's contribution).

### Outputs (`results/<run>/`)

`windows.csv` (trace-window manifest), `per_run.csv` (all metrics per window/load/seed/strategy),
`summary.csv` and `summary_by_load.csv` (means with 95% bootstrap CIs), `run_manifest.json`
(config, dataset SHA-256, LLM model digest, versions), `llm_cache.jsonl` (every prompt's
response and measured latency), `audit/*.jsonl` (every decision, rejection and execution).

## Method details and assumptions

**Trace → tasks.** One task per `(collection_id, instance_index)`; arrival = event `time`;
runtime = the row's usage window (`end_time − start_time`) clipped to [1, 3600] s; CPU/memory =
`resource_request`; failure = event FAIL / EVICT / LOST / KILL. The Kaggle sample joins events
to usage windows, so runtime is a proxy, not true task lifetime — state this as a threat to validity.

**Windows and load.** Per cell, the middle 6-hour window (after day 1) with ≥150 tasks.
Cluster capacity is set so offered work / (capacity × window) equals the load level
(0.7 / 0.9 / 1.1 = low / medium / high), never below the largest single request.

**Failures.** FAIL/EVICT/LOST tasks fail on their first attempt after a seeded fraction
U(0.1, 0.9) of runtime; retries fail again with p = 0.2. KILL = user termination, no recovery.
Detection after a 5 s heartbeat. All strategies get identical failure realisations per seed.

**Policy rules** (blocked for the agentic OS, audited for all strategies):
P1 capacity, P2 retry budget (≤3 retries), P3 anti-starvation (tasks waiting >1800 s go first),
P4 user share (a user may not exceed 50 % of CPU while another user's task fits).

**Metrics.** Latency = finish − submit over completed tasks; utilisation = time-averaged
allocated CPU and memory share (mean of the two); recovery = failure → restart;
violation rate = executed starts violating ≥1 rule / all starts; fairness = Jain's index over
users of mean(runtime / latency). The LLM's own decision time (measured wall-clock) is
charged to the simulated clock, so LLM overhead counts against the agentic OS.

**LLM.** `llama3:latest` (8B, Q4_0) via Ollama, temperature 0, seed 42, JSON output mode,
max 20 candidates per prompt, no runtime hints (baselines don't see runtimes either).
Cached responses make reruns bit-identical.

## Results (medium load, `results/medium/`)

Mean [95% bootstrap CI] over 8 six-hour windows (one per Borg cell), load 0.9, seed 1.

| Metric | FIFO | Priority | Agentic OS (no LLM) | Agentic OS (LLM) |
|---|---|---|---|---|
| Avg. latency (s) | 7423 [2958, 13073] | 4249 [1847, 7012] | 4589 [1922, 7321] | 5802 [3362, 8268] |
| P95 latency (s) | 14125 [6408, 23602] | 12408 [4598, 22106] | 13795 [5181, 23780] | 16120 [7999, 25882] |
| Utilisation (%) | 40.0 [28.1, 51.9] | 43.1 [30.6, 55.0] | 40.7 [29.6, 51.7] | 44.9 [36.1, 52.9] |
| Recovery time (s) | 3538 [513, 8111] | 2368 [110, 6685] | 420 [44, 988] | 1282 [269, 2815] |
| Policy violations (%) | 20.72 [6.51, 38.33] | 7.79 [0.77, 20.66] | 0.00 | 0.00 |
| Fairness (Jain) | 0.626 [0.510, 0.734] | 0.725 [0.645, 0.813] | 0.723 [0.636, 0.815] | 0.687 [0.615, 0.767] |

The policy-governed control plane removes executed policy violations and shortens failure
recovery; the LLM (9,773 calls, 5.4 s mean decision time, 50.5 % of proposals rejected by the
policy gate) does not improve latency over the same control plane without it. Paired
comparisons: `results/medium/report/paired.md`; LLM statistics: `report/llm_stats.md`.

## Data and licence

Code and the files in `results/` are released under the MIT licence (`LICENSE`). The trace is
not redistributed: it is the Kaggle extract
[derrickmwiti/google-2019-cluster-sample](https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample)
of Google's ClusterData2019; `run_manifest.json` records the SHA-256 of the file used. Llama 3 is
subject to the Meta Llama 3 Community License. Cite via `CITATION.cff`.

## Not yet implemented

- MAS-RL (matched reinforcement-learning comparator) described in the paper.
- Multi-node placement / fragmentation (the cluster is one pooled resource).
