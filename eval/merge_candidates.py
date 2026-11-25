"""Append reviewed candidates to golden_set.csv.

Only rows with review_status == "keep" are added (edit question / ground_truth in the
sheet first if a row needs fixing, then mark it "keep"). Safe to re-run: questions that
are already in the golden set are skipped.

Usage:  python eval/merge_candidates.py [--dry-run]
"""
import os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "eval", "golden_set.csv")
CANDS = os.path.join(ROOT, "eval", "golden_candidates.csv")

golden = pd.read_csv(GOLDEN)
cands = pd.read_csv(CANDS)
keep = cands[cands.review_status.astype(str).str.strip().str.lower() == "keep"]
keep = keep[~keep.question.isin(golden.question)]

next_id = int(golden.id.str[1:].astype(int).max()) + 1
new = pd.DataFrame({
    "id": [f"q{next_id + i:03d}" for i in range(len(keep))],
    "question": keep.question.values,
    "ground_truth": keep.ground_truth.fillna("").values,
    "source_id": keep.source_id.fillna("").replace("", pd.NA).values,
    "source_id_original": keep.source_id.fillna("").replace("", pd.NA).values,
    "category": keep.category.values,
    "difficulty": keep.difficulty.values,
    "question_type": keep.question_type.values,
    "notes": ("from " + keep.cand_id + ("; " + keep.review_notes.fillna("")).where(
        keep.review_notes.notna(), "")).values,
})

print(f"Adding {len(new)} rows:")
print(new.question_type.value_counts().to_string())
print("\nGolden set after merge:")
print(pd.concat([golden, new]).question_type.value_counts().to_string())

if "--dry-run" not in sys.argv:
    pd.concat([golden, new], ignore_index=True)[golden.columns].to_csv(GOLDEN, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {GOLDEN}")
