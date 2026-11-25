# Efteely evaluation

> **Status:** golden set **v1 has 79 questions**. Retrieval is measured on all 79. The LLM-judge answer eval has only run on the first 20 (v0): Groq's free-tier limits stopped the full run, and `run_answer_eval.py` resumes from its cache.

## Golden set: `golden_set.csv`

| type | n | what it tests |
|---|---|---|
| `paraphrase` | 29 | Egyptian-dialect questions, the main use case |
| `no_answer` | 20 | nothing in the corpus answers it (16 modern fiqh-shaped "hard negatives", 4 out-of-scope) |
| `direct` | 14 | short MSA questions, reworded from the fatwa |
| `multi_doc` | 10 | one situation that needs two fatwas |
| `title_match` | 6 | near-copies of a fatwa title, kept as an upper bound |

| column | meaning |
|---|---|
| `ground_truth` | the ruling, quoted verbatim from the labelled fatwa |
| `source_id_original` | the fatwa(s) the question was written from (`site_id`, `\|`-separated) |
| `source_id` | `source_id_original` + other fatwas that give **the same ruling to the same question**, accepted after manual review (only done for q008 and q012 so far) |
| `notes` | review decisions and scholarly differences the judge must respect |

Two metrics come from this: **strict** uses `source_id_original` only, **lenient** includes the accepted alternatives. The corpus has 50K+ fatwas and many of them answer the same question, so strict undercounts. For the 59 new questions lenient equals strict (their alternatives haven't been reviewed yet), so lenient is a lower bound too.

### How v1 was built (`build_golden_candidates.py` → review → `merge_candidates.py`)
1. **Drafting.** Random fatwas (seed 42), 50% IslamWeb, 25% Bin Baz, 25% IslamQA, only fatwas of 3 chunks or fewer so each question has one clear answer. `gpt-oss-120b` (Gemini as fallback when Groq is rate-limited) writes one question per fatwa without reusing its wording, plus a verbatim ground-truth quote. no_answer questions are hand-written.
2. **Automatic checks.** The quote must really appear in the fatwa. The question's similarity to the fatwa title is computed. A **different model family** (`qwen3.8-27b`) confirms that the source answers the question; for no_answer, it confirms that none of the 5 closest fatwas answers it. Whether the current retriever finds the answer is **not** used as a filter, because that would bias the set toward easy questions.
3. **Manual review.** 97 drafts → 59 kept (decisions and reasons are in `golden_candidates.csv`). The most common reasons to drop: a ruling missing its key condition, two unrelated topics glued into one question, and "no_answer" questions that the corpus does answer (astronauts' prayer, riding during tawaf).

## Scripts

| script | what it does |
|---|---|
| `run_retrieval_eval.py` | hit@1, hit@5, MRR (strict and lenient), recall@5, no_answer rejection → `retrieval_results.csv` |
| `distance_analysis.py` | sweeps `MAX_DISTANCE` offline → `distance_results.csv`, `threshold_sweep.csv` |
| `run_answer_eval.py` | the **same pipeline as the app** (router → retrieval → answer) + LLM judge → `answer_results.csv`; resumable through `answers.jsonl` / `judgments.jsonl` |
| `inspect_misses.py q006 …` | prints what was retrieved for given questions, plus raw chunk distances, for manual review |
| `build_golden_candidates.py`, `merge_candidates.py` | draft new questions for review / append the reviewed ones |

Run them from the repo root. Keys come from the environment or `.streamlit/secrets.toml` (`GROQ_API_KEY`, optional `GEMINI_API_KEY`).

## 1. Retrieval (v1, 79 questions, raw question → `search_fatwas`, `MAX_DISTANCE = 0.40`)

| type | n | hit@1 | hit@5 | MRR | recall@5 |
|---|---|---|---|---|---|
| paraphrase (dialect) | 29 | 52% → 55% | 76% → **79%** | 0.60 → 0.64 | |
| direct (MSA) | 14 | 29% → 36% | 71% → **79%** | 0.50 → 0.57 | |
| multi_doc | 10 | 70% | **90%** | 0.78 | 63% |
| title_match | 6 | 83% | 100% | 0.92 | |
| **answerable overall** | **59** | **53% → 56%** | **80% → 83%** | **0.64 → 0.68** | 75% |
| no_answer, correctly rejected | 20 | | **16/20 (80%)** | | |

*strict → lenient.* The 95% margin of error on 83% with n=59 is about ±10 points.

**Misses (hit@5 = 0):** 10 of 59. Seven are in dialect or reworded MSA, for example q006 (عدة المختلعة) and q009 (قضاء رمضان vs النذر), where the embedding matched the general topic and missed the specific concept. These are the before/after cases for hybrid BM25 and a reranker.

## 2. Choosing `MAX_DISTANCE`

Top-1 cosine distance by type (min / median / max): title_match 0.19 / 0.23 / 0.25, multi_doc 0.20 / 0.26 / 0.33, paraphrase 0.22 / 0.29 / 0.41, direct 0.17 / 0.29 / 0.47, **no_answer 0.34 / 0.44 / 0.62**.

| threshold | hit@5 | no_answer rejected | answerable left with zero fatwas |
|---|---|---|---|
| 0.36 | 75% | 85% | 14% |
| 0.38 | 81% | 85% | 7% |
| **0.40** | **83%** | **80%** | **3%** |
| 0.44 | 85% | 45% | 2% |
| 0.50 (old) | 85% | 20% | 0% |

0.40 was first picked on 20 questions (only 3 of them no_answer). On 79 questions it is still the best trade-off: it gives up 2 points of hit@5 to reject 4× more no_answer questions than 0.5. The distributions do overlap: 4 no_answer questions (VR glasses, consciousness upload, AI image platforms, zakat in USDT) sit closer to the corpus than the hardest answerable ones. **A distance threshold alone can't separate them**, and that's the case for adding a reranker or an LLM relevance check.

## 3. Answers (end-to-end, LLM judge; v0, first 20 questions only)

- Pipeline: the router (`gpt-oss-20b`) rewrites the question → `search_fatwas` → the answer model (`gpt-oss-120b`, temperature 0.1), with the same prompts as the app (shared in `rag_core.py`).
- Judge: `qwen/qwen3.8-27b`, a different family from the answer model, temperature 0, JSON output. It sees the question, the reference ruling, the reviewer notes, and the fatwas the answer cites.

| type | n | correctness | faithfulness | citations | pass* |
|---|---|---|---|---|---|
| title_match | 6 | 5.00 | 5.00 | 5.00 | 100% |
| direct | 2 | 3.00 | 5.00 | 3.00 | 50% |
| paraphrase (dialect) | 6 | 4.67 | 5.00 | 5.00 | 100% |
| multi_doc | 3 | 4.67 | 4.67 | 5.00 | 100% |
| **answerable overall** | 17 | **4.59** | **4.94** | **4.76** | **94%** |
| no_answer, abstained without guessing | 3 | | | | **3/3** |

\* pass = correctness ≥ 4 and faithfulness ≥ 4.

- **The router's query rewrite helps dialect questions.** It turned q009 into MSA and retrieved the right fatwa, which the raw question had missed.
- **A retrieval miss is no longer a wrong answer.** With the 0.40 threshold, q006 gets no fatwas and the bot says it found none instead of answering from unrelated fatwas.
- **Faithfulness is close to 5.** The model stays within the given fatwas and cites them.

## Limitations

- The answer eval covers 20 of the 79 questions. One question moves a type's score by 17–50 points.
- The judge was spot-checked by hand on 6 answers, with no formal human-agreement study. Treat the 4.x scores as upper bounds.
- The golden set was drafted by an LLM and reviewed by one person. 6 of the questions are near-copies of fatwa titles, kept and reported as an upper bound.
- Alternative correct fatwas were reviewed for only 2 questions, so lenient hit@5 underestimates.
