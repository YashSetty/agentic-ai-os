"""Paired comparisons for the policy-matched baselines and ablations (paper Section 6.4).
Usage: python analyze_grid.py results/grid/per_run.csv
Mean per load, and paired differences A - B with 95% window-cluster bootstrap CIs."""
import sys, numpy as np, pandas as pd
df=pd.read_csv(sys.argv[1])
M=['avg_latency_s','util_pct','avg_recovery_s','violation_rate_pct','fairness_jain']
S=['fifo','priority','fifo-gated','priority-gated','agentic-nollm','nollm-noreco','nollm-nogate']
rng=np.random.default_rng(0)
def boot_cluster(d, by='window_id', n=4000):
    g=d.groupby(by)['diff'].mean() if False else None
    wins=d['window_id'].unique(); arr={w:d[d.window_id==w]['diff'].values for w in wins}
    means=[]
    for _ in range(n):
        pick=rng.choice(wins,len(wins),replace=True)
        means.append(np.concatenate([arr[w] for w in pick]).mean())
    return np.percentile(means,[2.5,97.5])
out=[]
print('== Means by load (8 windows x 3 seeds = 24 runs per cell)')
for load,g in df.groupby('load'):
    t=g.groupby('strategy')[M].mean().reindex(S)
    print(f'\nload {load}'); print(t.round(3).to_string())
pairs=[('agentic-nollm','priority-gated'),('agentic-nollm','fifo-gated'),('priority-gated','priority'),
       ('nollm-noreco','agentic-nollm'),('nollm-nogate','agentic-nollm')]
print('\n== Paired differences A - B, mean [95% window-cluster bootstrap CI]')
for load,g in df.groupby('load'):
    print(f'\nload {load}')
    for a,b in pairs:
        x=g[g.strategy==a].set_index(['window_id','seed'])[M]; y=g[g.strategy==b].set_index(['window_id','seed'])[M]
        d=(x-y).reset_index()
        cells=[]
        for m in M:
            dd=d[['window_id',m]].rename(columns={m:'diff'})
            lo,hi=boot_cluster(dd); cells.append(f'{m.split("_")[1] if m!="avg_latency_s" else "lat"}={dd["diff"].mean():+.1f}[{lo:+.1f},{hi:+.1f}]')
        print(f'  {a} - {b}: '+'  '.join(cells))
