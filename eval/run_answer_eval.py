"""End-to-end answer evaluation with an LLM judge.

Stage 1 (generate): runs the SAME pipeline as app.py for every golden-set question
    router (rewrite / CHAT vs SEARCH) -> search_fatwas -> format_context -> answer LLM
    and caches the answers in eval/answers.jsonl. Judgments are cached in eval/judgments.jsonl.
    Both caches make the run resumable (Groq free tier has a daily token limit); --regen starts over.

Stage 2 (judge): a different model family (JUDGE_MODEL) scores each answer, temperature 0, JSON:
    answerable questions -> correctness / faithfulness / citations, each 1-5
    no_answer questions  -> abstained (said "no fatwa found" without guessing a ruling)

Usage:  python eval/run_answer_eval.py [--regen]
Output: eval/answer_results.csv (+ printed summary)
"""
import os, sys, re, json, time
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from rag_core import (
    load_rag, search_fatwas, format_context,
    build_answer_prompt, build_router_prompt, parse_router_output,
    ANSWER_MODEL, ROUTER_MODEL,
)
from eval.utils import link_to_id, get_groq_key, with_retry, DailyLimitError

JUDGE_MODEL   = "qwen/qwen3.8-27b"   # must NOT be the answer model's family (gpt-oss)
CITED_CHARS   = 2500                 # per cited fatwa shown to the judge (Groq free tier: 8K tokens/min)
UNCITED_CHARS = 700                  # per fatwa when the answer cites nothing
ANSWERS_PATH  = os.path.join(ROOT, "eval", "answers.jsonl")
RESULTS_PATH  = os.path.join(ROOT, "eval", "answer_results.csv")
JUDGMENTS_PATH = os.path.join(ROOT, "eval", "judgments.jsonl")   # cache, so runs can resume


# ── Stage 1: generate answers with the app's pipeline ────────────────────────
def load_jsonl(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in map(json.loads, fh) if r}


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate(df, key, cached):
    """Answer every question not in the cache. Each answer is saved as soon as it exists,
    so a run stopped by Groq's daily limit resumes where it left off."""
    todo = df[~df.id.isin(cached)]
    if todo.empty:
        return cached
    vs = load_rag()
    llm = ChatGroq(model=ANSWER_MODEL, temperature=0.1, groq_api_key=key, max_retries=0)
    router = ChatGroq(model=ROUTER_MODEL, temperature=0, groq_api_key=key, max_retries=0).with_fallbacks([llm])
    chain = build_answer_prompt() | llm | StrOutputParser()

    for row in todo.itertuples():
        print(f"[generate] {row.id}")
        router_out = with_retry(lambda: router.invoke(build_router_prompt("لا يوجد.", row.question)).content)
        use_search, query = parse_router_output(router_out, row.question)
        fatwas = search_fatwas(vs, query) if use_search else []
        context = format_context(fatwas) if use_search else "لا يوجد سياق فقهي محدد لهذه الرسالة."
        answer = with_retry(lambda: chain.invoke({"context": context, "question": row.question, "history": []}))
        rec = {
            "id": row.id,
            "router_output": router_out,
            "search_query": query,
            "fatwas": [{"id": link_to_id(f["link"]), "title": f["title"], "distance": f["distance"],
                        "text": "\n...\n".join(f["chunks"])} for f in fatwas],
            "answer": answer,
        }
        append_jsonl(ANSWERS_PATH, rec)
        cached[row.id] = rec
    return cached


# ── Stage 2: judge ───────────────────────────────────────────────────────────
JUDGE_ANSWERABLE = """You are a strict evaluator of an Islamic-fiqh RAG assistant. All content is Arabic.

QUESTION:
{question}

REFERENCE ANSWER (ground truth from the labelled fatwa):
{ground_truth}

REVIEWER NOTES (may mention valid scholarly differences):
{notes}

FATWAS GIVEN TO THE ASSISTANT (numbered as the assistant saw them):
{sources}

ASSISTANT ANSWER:
{answer}

Score each criterion from 1 (worst) to 5 (best):
- correctness: does the answer give the same ruling as the reference? A different ruling that is
  explicitly a recognised scholarly view supported by the given fatwas is NOT wrong (score 4+).
  5 = same ruling, 3 = partly / hedged / missing the key condition, 1 = contradicts it or no ruling.
- faithfulness: is every claim in the answer supported by the given fatwas? 5 = everything supported,
  3 = some unsupported additions, 1 = mostly invented. Judge ONLY against the fatwas shown.
- citations: are [n] markers present and does each point to a fatwa that supports the sentence?
  5 = all correct, 3 = some missing or wrong, 1 = none or all wrong.

Return ONLY a JSON object:
{{"correctness": int, "faithfulness": int, "citations": int, "reason": "<one short English sentence>"}}"""

JUDGE_NO_ANSWER = """You are a strict evaluator of an Islamic-fiqh RAG assistant. All content is Arabic.
The question below has NO answer in the fatwa database. The correct behaviour is to say clearly that
no fatwa was found and refer the user to scholars, WITHOUT stating or guessing a ruling.

QUESTION:
{question}

ASSISTANT ANSWER:
{answer}

Return ONLY a JSON object:
{{"abstained": true|false, "guessed_ruling": true|false, "reason": "<one short English sentence>"}}"""


def sources_for_judge(rec):
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", rec["answer"])}
    parts = []
    for i, f in enumerate(rec["fatwas"], 1):
        if cited and i not in cited:
            parts.append(f"[{i}] {f['title']}  (not cited)")
            continue
        limit = CITED_CHARS if cited else UNCITED_CHARS
        parts.append(f"[{i}] {f['title']}\n{f['text'][:limit]}")
    return "\n\n---\n\n".join(parts) or "(no fatwas were retrieved)"


def judge(client, prompt):
    def call():
        r = client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = r.choices[0].message.content
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        return json.loads(text)
    return with_retry(call)


def judge_row(client, row, rec):
    gold = row.source_id.split("|") if isinstance(row.source_id, str) else []
    out = {
        "id": row.id, "question_type": row.question_type,
        "search_query": rec["search_query"],
        "e2e_hit@5": int(any(f["id"] in gold for f in rec["fatwas"])) if gold else None,
        "n_fatwas": len(rec["fatwas"]),
        "answer": rec["answer"],
    }
    if row.question_type == "no_answer":
        j = judge(client, JUDGE_NO_ANSWER.format(question=row.question, answer=rec["answer"]))
        out["abstained"] = int(bool(j.get("abstained")) and not j.get("guessed_ruling"))
    else:
        j = judge(client, JUDGE_ANSWERABLE.format(
            question=row.question, ground_truth=row.ground_truth,
            notes=row.notes if isinstance(row.notes, str) and row.notes else "-",
            sources=sources_for_judge(rec), answer=rec["answer"]))
        for k in ("correctness", "faithfulness", "citations"):
            out[k] = int(j[k])
        out["pass"] = int(out["correctness"] >= 4 and out["faithfulness"] >= 4)
    out["judge_reason"] = j.get("reason", "")
    return out


def main():
    df = pd.read_csv(os.path.join(ROOT, "eval", "golden_set.csv"))
    key = get_groq_key()

    if "--regen" in sys.argv:
        for path in (ANSWERS_PATH, JUDGMENTS_PATH):
            if os.path.exists(path):
                os.remove(path)
    answers = load_jsonl(ANSWERS_PATH)
    judgments = load_jsonl(JUDGMENTS_PATH)
    # a judgment is only valid for the exact answer it scored
    judgments = {i: j for i, j in judgments.items() if i in answers and j.get("answer") == answers[i]["answer"]}

    client = Groq(api_key=key, max_retries=0)   # with_retry handles 429s
    try:
        answers = generate(df, key, answers)
        for row in df[~df.id.isin(judgments)].itertuples():
            print(f"[judge] {row.id}")
            judgments[row.id] = judge_row(client, row, answers[row.id])
            append_jsonl(JUDGMENTS_PATH, judgments[row.id])
            time.sleep(2)   # stay under the judge's tokens-per-minute limit
    except DailyLimitError as e:
        done = sum(i in judgments for i in df.id)
        print(f"\nStopped: {e}\nProgress is saved ({done}/{len(df)} judged). Rerun the same command later to resume.")
        sys.exit(2)

    res = pd.DataFrame([judgments[i] for i in df.id])
    res.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    metrics = ["correctness", "faithfulness", "citations", "pass", "e2e_hit@5", "abstained"]
    print("\nAnswer quality by question type (scores 1-5, pass/hit/abstained = share):")
    print(res.groupby("question_type")[metrics].mean().round(2).to_string())
    print("\nOverall:")
    print(res[metrics].mean().round(2).to_string())
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
