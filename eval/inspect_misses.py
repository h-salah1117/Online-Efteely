"""Dump what retrieval returned for selected golden-set questions, for manual review.

Usage:  python eval/inspect_misses.py [q006 q008 ...]
Output: eval/misses_review.md
"""
import os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag_core import load_rag, search_fatwas, RETRIEVE_K, MAX_DISTANCE
from eval.utils import link_to_id

DEFAULT_IDS = ["q006", "q008", "q009", "q012"]
SNIPPET_CHARS = 300

ids = sys.argv[1:] or DEFAULT_IDS
df = pd.read_csv(os.path.join(ROOT, "eval", "golden_set.csv")).set_index("id")
vs = load_rag()

lines = [f"# Retrieval misses review\n", f"RETRIEVE_K={RETRIEVE_K}, MAX_DISTANCE={MAX_DISTANCE}\n"]

for qid in ids:
    row = df.loc[qid]
    gold = str(row.source_id).split("|") if isinstance(row.source_id, str) else []

    lines += [
        f"\n---\n\n## {qid} ({row.question_type}, {row.category})\n",
        f"**Question:** {row.question}\n",
        f"**Gold:** {', '.join(gold) or '-'}\n",
        f"**Ground truth:** {row.ground_truth}\n",
        "\n### Top fatwas (what search_fatwas returns)\n",
    ]

    fatwas = search_fatwas(vs, row.question)
    if not fatwas:
        lines.append("_No fatwas returned (all chunks beyond MAX_DISTANCE)._\n")
    for rank, f in enumerate(fatwas, 1):
        fid = link_to_id(f["link"]) or "?"
        mark = " ✅ GOLD" if fid in gold else ""
        snippet = f["chunks"][0][:SNIPPET_CHARS].replace("\n", " ")
        lines += [
            f"\n**#{rank} — {fid}{mark}**  \n",
            f"Title: {f['title']}  \n",
            f"Link: {f['link']}  \n",
            f"> {snippet}...\n",
        ]

    # Raw chunk ranking before grouping/threshold: shows if gold was close but filtered out
    lines.append(f"\n### Raw top-{RETRIEVE_K} chunks (distance, before MAX_DISTANCE filter)\n\n")
    lines.append("| rank | id | distance | kept |\n|---|---|---|---|\n")
    for rank, (doc, dist) in enumerate(vs.similarity_search_with_score(row.question, k=RETRIEVE_K), 1):
        link = doc.metadata.get("link") or doc.metadata.get("source") or ""
        fid = link_to_id(link) or "?"
        mark = " ✅" if fid in gold else ""
        kept = "yes" if dist <= MAX_DISTANCE else "no"
        lines.append(f"| {rank} | {fid}{mark} | {dist:.3f} | {kept} |\n")

out = os.path.join(ROOT, "eval", "misses_review.md")
with open(out, "w", encoding="utf-8") as fh:
    fh.writelines(lines)
print(f"Saved -> {out}")
