Mean [95% bootstrap CI] over 8 runs (window x load; Google 2019 Cluster Sample, load [np.float64(0.9)]).

| Metric | FIFO | Priority heuristic | Agentic OS (no LLM) | Agentic OS (LLM) |
|---|---|---|---|---|
| Avg. task completion latency (s) | 7423.3 [2958.1, 13072.5] | 4248.9 [1846.6, 7011.5] | 4589.4 [1922.1, 7320.9] | 5801.6 [3362.3, 8268.3] |
| P95 task completion latency (s) | 14125.3 [6407.6, 23602.4] | 12408.1 [4597.8, 22106.4] | 13794.6 [5181.4, 23780.4] | 16120.0 [7998.5, 25882.2] |
| Resource utilisation (%) | 40.0 [28.1, 51.9] | 43.1 [30.6, 55.0] | 40.7 [29.6, 51.7] | 44.9 [36.1, 52.9] |
| Failure recovery time (s) | 3537.9 [513.0, 8111.1] | 2367.9 [109.9, 6684.7] | 419.5 [43.7, 988.3] | 1281.7 [269.3, 2815.4] |
| Policy violation rate (%) | 20.72 [6.51, 38.33] | 7.79 [0.77, 20.66] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Scheduling fairness (Jain) | 0.626 [0.510, 0.734] | 0.725 [0.645, 0.813] | 0.723 [0.636, 0.815] | 0.687 [0.615, 0.767] |
