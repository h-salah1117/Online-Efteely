"""Pick MAX_DISTANCE from data instead of guessing.

For every golden-set question we retrieve once with no threshold (max_distance=1.0),
then simulate each candidate threshold offline: a fatwa survives if its closest chunk
is within the threshold. Because results are sorted by distance, filtering afterwards
gives exactly what search_fatwas would return with that threshold.

Trade-off we measure per threshold:
  - no_answer_ok : share of unanswerable questions that get ZERO fatwas (want high)
  - hit@5        : share of answerable questions whose accepted fatwa is still returned (want high)

Usage:  python eval/distance_analysis.py
Output: eval/distance_results.csv, eval/threshold_sweep.csv (+ printed tables)
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag_core import load_rag, search_fatwas, MAX_DISTANCE
from eval.utils import link_to_id

THRESHOLDS = np.round(np.arange(0.30, 0.51, 0.02), 2)

df = pd.read_csv(os.path.join(ROOT, "eval", "golden_set.csv"))
vs = load_rag()

rows = []
for row in df.itertuples():
    fatwas = search_fatwas(vs, row.question, max_distance=1.0)
    gold = row.source_id.split("|") if isinstance(row.source_id, str) else []
    ranked = [(link_to_id(f["link"]), f["distance"]) for f in fatwas]
    gold_dists = [d for fid, d in ranked if fid in gold]
    rows.append({
        "id": row.id,
        "question_type": row.question_type,
        "answerable": row.question_type != "no_answer",
        "top1_distance": ranked[0][1] if ranked else np.nan,
        # distance of the best accepted fatwa, if it is in the top-5 at all
        "gold_distance": min(gold_dists) if gold_dists else np.nan,
        "ranked": ranked,
        "gold": gold,
    })

res = pd.DataFrame(rows)

print("Top-1 distance by question type:")
print(res.groupby("question_type")["top1_distance"].describe()[["min", "50%", "max"]].round(3))
print("\nPer question:")
print(res[["id", "question_type", "top1_distance", "gold_distance"]].round(3).to_string(index=False))

sweep = []
ans = res[res.answerable]
neg = res[~res.answerable]
for t in THRESHOLDS:
    kept = lambda r: [fid for fid, d in r.ranked if d <= t]
    hit5 = np.mean([any(fid in r.gold for fid in kept(r)) for r in ans.itertuples()])
    no_ans_ok = np.mean([len(kept(r)) == 0 for r in neg.itertuples()])
    # answerable questions that would get no fatwa at all ("I don't know" on a question we can answer)
    false_reject = np.mean([len(kept(r)) == 0 for r in ans.itertuples()])
    sweep.append({"threshold": t, "hit@5": round(hit5, 2), "no_answer_ok": round(no_ans_ok, 2),
                  "answerable_rejected": round(false_reject, 2)})

sweep = pd.DataFrame(sweep)
print(f"\nThreshold sweep (current MAX_DISTANCE = {MAX_DISTANCE}):")
print(sweep.to_string(index=False))

res.drop(columns=["ranked", "gold"]).to_csv(os.path.join(ROOT, "eval", "distance_results.csv"),
                                             index=False, encoding="utf-8-sig")
sweep.to_csv(os.path.join(ROOT, "eval", "threshold_sweep.csv"), index=False)
print("\nSaved -> eval/distance_results.csv, eval/threshold_sweep.csv")
