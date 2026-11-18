"""Per-topic summary from replies.jsonl (which already carries the matching p_side_1 and mass).
Usage: python summarise.py ../replies.jsonl summary.csv"""
import csv, json, statistics as st, sys
reps = [json.loads(l) for l in open(sys.argv[1])]
rows = []
for t in sorted({r["topic"] for r in reps}) + ["all"]:
    g = [r for r in reps if t == "all" or r["topic"] == t]
    ref = [r for r in g if r["label"] == "refused"]; ans = [r for r in g if r["label"].startswith("answered")]
    p = [r["p_side_1_ab_readout"] for r in ref]
    rows.append({"topic": t, "replies": len(g), "refused": len(ref), "answered": len(ans),
                 "answered_side_1_pct": round(100 * sum(r["answered_side"] == "side_1" for r in ans) / len(ans), 1) if ans else "",
                 "refused_mean_p_side_1_pct": round(100 * st.mean(p), 1) if p else "",
                 "refused_p_gt_0.9_pct": round(100 * sum(x > 0.9 for x in p) / len(p), 1) if p else ""})
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
