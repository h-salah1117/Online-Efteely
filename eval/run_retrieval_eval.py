import os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag_core import load_rag, search_fatwas
from eval.utils import link_to_id

df = pd.read_csv(os.path.join(ROOT, "eval", "golden_set.csv"))

vs = load_rag()

def split_ids(value):
    return value.split("|") if isinstance(value, str) else []


def hit_metrics(retrieved, gold, suffix):
    """hit@1 / hit@5 / MRR against a set of acceptable fatwa ids."""
    first_hit = next((i for i, r in enumerate(retrieved, 1) if r in gold), None)
    return {
        f"hit@1{suffix}": int(bool(retrieved) and retrieved[0] in gold),
        f"hit@5{suffix}": int(first_hit is not None),
        f"mrr{suffix}":   1 / first_hit if first_hit else 0,
    }


rows = []
for row in df.itertuples():
    fatwas    = search_fatwas(vs, row.question)
    retrieved = [link_to_id(f["link"]) for f in fatwas]   # من الأقرب للأبعد
    result = {
        "id": row.id,
        "question_type": row.question_type,
        "retrieved": "|".join(r or "?" for r in retrieved),
        "gold": row.source_id if isinstance(row.source_id, str) else "",
    }

    if row.question_type == "no_answer":
        result["no_answer_ok"] = int(len(fatwas) == 0)
    else:
        original = split_ids(row.source_id_original)   # the single labelled fatwa(s)
        accepted = split_ids(row.source_id)            # + alternatives accepted after manual review

        result.update(hit_metrics(retrieved, original, "_strict"))
        result.update(hit_metrics(retrieved, accepted, ""))
        # recall uses the original ids: alternatives are "any of", not "all required"
        result["recall@5"] = len(set(retrieved) & set(original)) / len(original)

    rows.append(result)

res = pd.DataFrame(rows)
metrics = ["hit@1_strict", "hit@5_strict", "mrr_strict",
           "hit@1", "hit@5", "mrr", "recall@5", "no_answer_ok"]

pd.set_option("display.width", 200)
print("strict = original label only | lenient (no suffix) = + accepted alternatives\n")
print(res.groupby("question_type")[metrics].mean().round(2))
print("\nOverall:")
print(res[metrics].mean().round(2))

out = os.path.join(ROOT, "eval", "retrieval_results.csv")
res.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\nSaved -> {out}")
