Paired by window. diff = A - B (negative latency/recovery = A better). Bootstrap 95% CI of the mean difference; wins = windows where A is better.

| A vs B | Metric | Mean diff [95% CI] | A wins |
|---|---|---|---|
| Agentic OS (LLM) vs Agentic OS (no LLM) | avg_latency_s | 1212.22 [245.13, 2461.74] | 1/8 |
| Agentic OS (LLM) vs Agentic OS (no LLM) | avg_recovery_s | 862.24 [138.59, 1857.06] | 0/8 |
| Agentic OS (LLM) vs Agentic OS (no LLM) | util_pct | 4.20 [0.28, 9.25] | 6/8 |
| Agentic OS (LLM) vs Agentic OS (no LLM) | violation_rate_pct | 0.00 [0.00, 0.00] | 0/8 |
| Agentic OS (LLM) vs Agentic OS (no LLM) | fairness_jain | -0.04 [-0.10, 0.01] | 3/8 |
| Agentic OS (LLM) vs Priority heuristic | avg_latency_s | 1552.79 [470.29, 2819.73] | 1/8 |
| Agentic OS (LLM) vs Priority heuristic | avg_recovery_s | -1086.11 [-4033.77, 633.33] | 3/8 |
| Agentic OS (LLM) vs Priority heuristic | util_pct | 1.77 [-4.72, 8.34] | 5/8 |
| Agentic OS (LLM) vs Priority heuristic | violation_rate_pct | -7.79 [-20.66, -0.77] | 7/8 |
| Agentic OS (LLM) vs Priority heuristic | fairness_jain | -0.04 [-0.11, 0.01] | 2/8 |
| Agentic OS (LLM) vs FIFO | avg_latency_s | -1621.66 [-6042.25, 1624.57] | 3/8 |
| Agentic OS (LLM) vs FIFO | avg_recovery_s | -2256.14 [-5372.35, -3.04] | 6/8 |
| Agentic OS (LLM) vs FIFO | util_pct | 4.88 [-2.27, 12.13] | 6/8 |
| Agentic OS (LLM) vs FIFO | violation_rate_pct | -20.72 [-38.33, -6.51] | 7/8 |
| Agentic OS (LLM) vs FIFO | fairness_jain | 0.06 [-0.04, 0.16] | 7/8 |
| Agentic OS (no LLM) vs Priority heuristic | avg_latency_s | 340.57 [-558.06, 1358.22] | 5/8 |
| Agentic OS (no LLM) vs Priority heuristic | avg_recovery_s | -1948.35 [-5707.70, -13.80] | 6/8 |
| Agentic OS (no LLM) vs Priority heuristic | util_pct | -2.43 [-6.31, -0.10] | 2/8 |
| Agentic OS (no LLM) vs Priority heuristic | violation_rate_pct | -7.79 [-20.66, -0.77] | 7/8 |
| Agentic OS (no LLM) vs Priority heuristic | fairness_jain | -0.00 [-0.02, 0.01] | 2/8 |
| Agentic OS (no LLM) vs FIFO | avg_latency_s | -2833.88 [-7021.29, -196.58] | 6/8 |
| Agentic OS (no LLM) vs FIFO | avg_recovery_s | -3118.38 [-7131.08, -465.95] | 7/8 |
| Agentic OS (no LLM) vs FIFO | util_pct | 0.67 [-4.38, 5.44] | 4/8 |
| Agentic OS (no LLM) vs FIFO | violation_rate_pct | -20.72 [-38.33, -6.51] | 7/8 |
| Agentic OS (no LLM) vs FIFO | fairness_jain | 0.10 [0.03, 0.17] | 7/8 |
