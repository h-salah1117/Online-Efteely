import os, re, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag_core import load_rag, search_fatwas

df = pd.read_csv(os.path.join(ROOT, "eval", "golden_set.csv"))

vs = load_rag()

PATTERNS = {
    "islamweb": r"islamweb\.net/.*?/fatwa/(\d+)",
    "binbaz":   r"binbaz\.org\.sa/fatwas/(\d+)",
    "islamqa":  r"islamqa\.info/.*?/answers/(\d+)",
}

def link_to_id(link):
    for site, pattern in PATTERNS.items():
        m = re.search(pattern, link or "")
        if m:
            return f"{site}_{m.group(1)}"
    return None

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
        gold = row.source_id.split("|")
        first_hit = next((i for i, r in enumerate(retrieved, 1) if r in gold), None)

        result["hit@1"]    = int(bool(retrieved) and retrieved[0] in gold)
        result["hit@5"]    = int(first_hit is not None)
        result["mrr"]      = 1 / first_hit if first_hit else 0
        result["recall@5"] = len(set(retrieved) & set(gold)) / len(gold)

    rows.append(result)

res = pd.DataFrame(rows)
metrics = ["hit@1", "hit@5", "mrr", "recall@5", "no_answer_ok"]

print(res.groupby("question_type")[metrics].mean().round(2))
print("\nOverall:")
print(res[metrics].mean().round(2))

out = os.path.join(ROOT, "eval", "retrieval_results.csv")
res.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\nSaved → {out}")
