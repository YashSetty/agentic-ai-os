"""Remove agentic rows whose LLM calls spanned a system sleep, so a resume re-runs them.
Usage: python scripts_rerun_contaminated.py results/medium [threshold_s]"""
import glob, json, os, sys
d = sys.argv[1]; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
bad = set()
for f in glob.glob(os.path.join(d, "audit", "*_agentic.jsonl")):
    for line in open(f, encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("type") == "decision" and rec.get("latency_s", 0) > thr:
            bad.add(os.path.basename(f).split("_L")[0]); break
p = os.path.join(d, "per_run.jsonl")
rows = [json.loads(l) for l in open(p, encoding="utf-8")]
keep = [r for r in rows if not (r["strategy"] == "agentic" and r["window_id"] in bad)]
with open(p, "w", encoding="utf-8") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in keep)
print("contaminated windows:", sorted(bad), "| removed rows:", len(rows) - len(keep))
