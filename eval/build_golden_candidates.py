"""Draft new golden-set questions for human review (golden set v0 -> v1, ~80 questions).

Nothing here is added to golden_set.csv automatically. The output is a review sheet:
    eval/golden_candidates.csv  -> a human sets review_status = keep / fix / drop
    then:  python eval/merge_candidates.py   appends the "keep" rows to golden_set.csv

How each type is drafted
  paraphrase / direct : a random fatwa -> the GENERATOR writes one question a real user
                        would ask (Egyptian dialect / MSA) without copying the original
                        wording, plus a verbatim ground_truth quote from the answer.
  multi_doc           : a random fatwa + a nearby fatwa on a different point -> one
                        compound question that needs both.
  no_answer           : hand-written list below (modern, fiqh-shaped "hard negatives"
                        plus a few out-of-scope questions).

How candidates are checked (automatic, before the human pass)
  - quote_found      : ground_truth really appears in the source fatwa (no invented rulings)
  - title_similarity : how close the question is to the fatwa's own title (high = too easy)
  - validator_ok     : a DIFFERENT model family (VALIDATOR) confirms the source answers
                       the question / for no_answer, that none of the top-5 retrieved
                       fatwas answers it
  Retrieval success is deliberately NOT used as a filter: that would bias the set
  toward questions the current retriever already gets right.

Usage:  python eval/build_golden_candidates.py
        python eval/build_golden_candidates.py --types multi_doc --n 16 --seed 7 --append
"""
import os, sys, re, json, random, difflib
from collections import defaultdict
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chromadb
from groq import Groq
from rag_core import load_rag, search_fatwas, CHROMA_SUBDIR
from eval.utils import link_to_id, get_groq_key, with_retry, gemini_json

GENERATOR = "openai/gpt-oss-120b"
VALIDATOR = "qwen/qwen3.8-27b"      # different family from the generator
SEED = 42
TARGETS = {"paraphrase": 24, "direct": 12, "multi_doc": 7, "no_answer": 17}
OVERSAMPLE = 1.5                    # draft more than needed; unsuitable fatwas get dropped
MAX_FATWA_CHARS = 3500
SITE_SHARE = {"islamweb": 0.5, "binbaz": 0.25, "islamqa": 0.25}
CATEGORIES = ["Aqeedah", "Tahara", "Salah", "Janazah", "Zakat", "Siyam", "Hajj", "Family",
              "Muamalat", "Food", "Dress", "Adab", "Contemporary", "Other"]
OUT = os.path.join(ROOT, "eval", "golden_candidates.csv")

NO_ANSWER_CANDIDATES = [
    ("هل يجوز الصلاة وأنا لابس نظارة الواقع الافتراضي VR على عيني؟", "Salah"),
    ("ما حكم شراء أرض أو عقار افتراضي في الميتافيرس؟", "Muamalat"),
    ("ينفع أتجوز شخصية ذكاء اصطناعي أو روبوت؟", "Contemporary"),
    ("ما حكم تحميل وعي الإنسان على الكمبيوتر بعد وفاته؟", "Contemporary"),
    ("هل يجوز بيع صورة شخص متوفى على شكل NFT؟", "Muamalat"),
    ("حكم استخدام الديب فيك لتقليد صوت شيخ مشهور في فيديوهات دعوية؟", "Contemporary"),
    ("هل يجوز التبرع ببياناتي الجينية لشركات تحليل الحمض النووي زي 23andMe؟", "Contemporary"),
    ("ما حكم الشراء بنظام اشتري دلوقتي وادفع بعدين في تطبيقات زي ڤاليو وتابي؟", "Muamalat"),
    ("هل يجوز للصائم يستخدم الفيب أو السيجارة الإلكترونية من غير نيكوتين؟", "Siyam"),
    ("ما حكم تجميد البويضات للبنت اللي لسه ما اتجوزتش؟", "Family"),
    ("كيف يصلي رائد الفضاء المسلم في محطة الفضاء الدولية ويحدد القبلة؟", "Salah"),
    ("هل يجوز الطواف حول الكعبة بالسكوتر الكهربائي الشخصي؟", "Hajj"),
    ("ما حكم العمل في تدريب نماذج الذكاء الاصطناعي على محتوى فيه أغاني وموسيقى؟", "Contemporary"),
    ("هل يجوز لعب ألعاب الواقع الافتراضي جوه المسجد؟", "Adab"),
    ("حكم فلاتر تيك توك اللي بتغيّر شكل الوش في الفيديوهات؟", "Contemporary"),
    ("ما حكم الربح من تعدين العملات الرقمية باستخدام كهرباء الشغل؟", "Muamalat"),
    ("هل يجوز دفع زكاة المال بعملة USDT المستقرة؟", "Zakat"),
    ("ما حكم الاشتراك في منصات الذكاء الاصطناعي اللي بتعمل صور لأشخاص من غير إذنهم؟", "Contemporary"),
    ("هل يجوز تشغيل القرآن بصوت مُولّد بالذكاء الاصطناعي لقارئ متوفى؟", "Contemporary"),
    ("ما حكم الصلاة في عربية ذاتية القيادة وهي ماشية؟", "Salah"),
    ("إزاي أتعلم البرمجة بلغة بايثون من الصفر؟", "Out-of-scope"),
    ("مين كسب كأس العالم 2022؟", "Out-of-scope"),
    ("إيه أحسن طريقة أعمل بيها كيكة شوكولاتة في البيت؟", "Out-of-scope"),
    ("ما هي أعراض نقص فيتامين د؟", "Out-of-scope"),
]

GEN_SINGLE = """You are building an evaluation set for an Arabic Islamic-fiqh RAG chatbot used by Egyptians.
Below is ONE fatwa (the user's original question, then the scholar's answer).

FATWA:
{fatwa}

Task: write ONE question that a real user could ask and that THIS fatwa answers clearly.
Style: {style}
Rules:
- Ask about the general ruling the fatwa gives, not the asker's private story details.
- Do NOT copy the original question's wording: reword it the way a different person would ask.
- If the fatwa is unclear, only about a very personal situation, mainly refers to other fatwas,
  or has no clear ruling, set "suitable": false.
- "ground_truth": copy 1-2 sentences VERBATIM from the scholar's answer that state the ruling
  (exact characters, max 300 chars).
- "category": one of {categories}.
- "difficulty": "easy" (ruling stated directly) or "medium" (needs a condition or a detail).

Return ONLY JSON:
{{"suitable": bool, "reason": "<short English>", "question": "<Arabic>", "ground_truth": "<verbatim Arabic>", "category": "...", "difficulty": "easy|medium"}}"""

STYLE = {
    "paraphrase": "Egyptian colloquial Arabic (عامية مصرية), like a WhatsApp message, no diacritics. "
                  "It may add a short realistic context (e.g. 'أنا', 'مراتي', 'في الشغل').",
    "direct": "Short Modern Standard Arabic question (فصحى), one sentence, different wording from the original.",
}

GEN_MULTI = """You are building an evaluation set for an Arabic Islamic-fiqh RAG chatbot used by Egyptians.
Below are TWO related fatwas, A and B.

FATWA A:
{a}

FATWA B:
{b}

Task: if it is natural, write ONE compound question that needs BOTH fatwas: part 1 answered by A,
part 2 answered by B, and neither fatwa alone answers both parts.
Both parts must come from ONE realistic situation of ONE person (e.g. the same act of worship,
the same marriage, the same transaction). Never glue two unrelated topics with "and also".
If no such question exists (they say the same thing, or the topics do not belong to one
situation), set "suitable": false. Being strict here is better than a forced question.
Write it in Egyptian colloquial Arabic or simple MSA, as a real user would, without copying the
original wording.
"ground_truth": two short VERBATIM quotes, the ruling from A, then " | ", then the ruling from B.
Each quote must include the key condition of that ruling, if there is one.
"category": one of {categories}.

Return ONLY JSON:
{{"suitable": bool, "reason": "<short English>", "question": "<Arabic>", "ground_truth": "<quote A> | <quote B>", "category": "..."}}"""

VAL_SINGLE = """Arabic Islamic-fiqh evaluation set check. Be strict.

QUESTION: {question}
PROPOSED REFERENCE RULING: {ground_truth}

SOURCE FATWA(S):
{fatwa}

1. Does the source (all of the sources together, if several) clearly answer the question?
2. Is the proposed reference ruling what the source actually says?
3. Is the question natural and unambiguous?
Return ONLY JSON: {{"ok": bool, "reason": "<one short English sentence>"}}"""

VAL_NO_ANSWER = """Arabic Islamic-fiqh evaluation set check. The QUESTION is meant to have NO answer in a
fatwa database. Below are the 5 closest fatwas from that database. Be strict: if ANY of them gives a
ruling that answers the question (even under a general principle that clearly covers this case), it
is answerable.

QUESTION: {question}

CLOSEST FATWAS:
{fatwas}

Return ONLY JSON: {{"answered_by_any": bool, "which": "<numbers or empty>", "reason": "<one short English sentence>"}}"""


def norm(t):
    """Normalise Arabic for fuzzy comparisons (diacritics, alef/yaa/taa variants, spaces)."""
    t = re.sub(r"[ً-ْـ]", "", t or "")
    t = re.sub("[إأآا]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def quote_found(quote, text):
    parts = [q for q in quote.split("|") if q.strip()]
    return all(norm(q) in norm(text) or
               difflib.SequenceMatcher(None, norm(q), norm(text)).find_longest_match().size >= 0.8 * len(norm(q))
               for q in parts)


def similarity(a, b):
    return round(difflib.SequenceMatcher(None, norm(a), norm(b)).ratio(), 2)


def ask(client, model, prompt):
    """Groq first; if it stays rate-limited or the connection drops, fall back to Gemini.
    The returned dict carries "_model": the model that actually answered."""
    temperature = 0.3 if model == GENERATOR else 0

    def call():
        r = client.chat.completions.create(model=model, temperature=temperature,
                                           response_format={"type": "json_object"},
                                           messages=[{"role": "user", "content": prompt}])
        text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content, flags=re.S).strip()
        return json.loads(text)
    try:
        out = with_retry(call, tries=3)
        out["_model"] = model
        return out
    except Exception as e:
        print(f"   groq failed ({e.__class__.__name__}), falling back to Gemini")
    try:
        out, used = gemini_json(prompt, temperature)
        out["_model"] = used
        return out
    except Exception as e:   # one bad item must not kill the run
        print(f"   skipped: {e.__class__.__name__}: {str(e)[:120]}")
        return None


def load_corpus():
    """link -> {"title", "chunks"} for every fatwa in the index (metadata only, no embeddings)."""
    col_info = chromadb.PersistentClient(path=CHROMA_SUBDIR).list_collections()[0]
    col = chromadb.PersistentClient(path=CHROMA_SUBDIR).get_collection(getattr(col_info, "name", col_info))
    fatwas, n, batch = defaultdict(lambda: {"title": "", "chunks": {}}), col.count(), 5000
    for off in range(0, n, batch):
        r = col.get(limit=batch, offset=off, include=["metadatas", "documents"])
        for m, d in zip(r["metadatas"], r["documents"]):
            f = fatwas[m["link"]]
            f["title"] = m.get("title", "")
            f["chunks"][m.get("chunk", 0)] = d
    return fatwas


def fatwa_text(f):
    return "\n".join(f["chunks"][k] for k in sorted(f["chunks"]))[:MAX_FATWA_CHARS]


def sample_links(fatwas, n, exclude, rng):
    """Stratified by site; short fatwas (<= 3 chunks) so one question has one clear answer."""
    by_site = defaultdict(list)
    for link, f in fatwas.items():
        fid = link_to_id(link)
        if fid and fid not in exclude and len(f["chunks"]) <= 3:
            by_site[fid.split("_")[0]].append(link)
    out = []
    for site, share in SITE_SHARE.items():
        pool = sorted(by_site[site])
        out += rng.sample(pool, min(len(pool), round(n * share)))
    rng.shuffle(out)
    return out


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=list(TARGETS), help="which question types to draft")
    ap.add_argument("--n", type=int, default=None, help="override how many fatwas to draft per type")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--append", action="store_true", help="add to the existing candidates sheet")
    return ap.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    client = Groq(api_key=get_groq_key(), max_retries=0)
    golden = pd.read_csv(os.path.join(ROOT, "eval", "golden_set.csv"))
    used = {i for s in golden.source_id.dropna() for i in s.split("|")}
    previous = pd.read_csv(OUT) if args.append and os.path.exists(OUT) else None
    if previous is not None:
        used |= {i for s in previous.source_id.dropna() for i in str(s).split("|")}

    print("Loading corpus metadata ...")
    fatwas = load_corpus()
    print(f"  {len(fatwas)} fatwas")
    vs = load_rag()

    n_single = {t: (args.n or int(TARGETS[t] * OVERSAMPLE)) if t in args.types else 0
                for t in ("paraphrase", "direct")}
    # compound questions fail more often, so draft twice as many
    n_multi = (args.n or int(TARGETS["multi_doc"] * 2)) if "multi_doc" in args.types else 0
    links = sample_links(fatwas, sum(n_single.values()) + n_multi, used, rng)
    plan = (["paraphrase"] * n_single["paraphrase"] + ["direct"] * n_single["direct"]
            + ["multi_doc"] * n_multi)

    rows = []
    for qtype, link in zip(plan, links):
        f, fid = fatwas[link], link_to_id(link)
        print(f"[{qtype}] {fid}")
        if qtype != "multi_doc":
            g = ask(client, GENERATOR, GEN_SINGLE.format(fatwa=fatwa_text(f), style=STYLE[qtype],
                                                         categories=CATEGORIES))
            if not g:
                continue
            row = {"question_type": qtype, "source_id": fid, "source_title": f["title"],
                   "source_link": link, "source_text": fatwa_text(f),
                   "generator_model": g.pop("_model"), **g}
        else:
            # nearest fatwa on a different point: close enough to be related, not a duplicate
            partner = None
            for doc, d in vs.similarity_search_with_score(f["title"], k=20):
                pl = doc.metadata.get("link", "")
                if pl != link and 0.18 < d < 0.40 and link_to_id(pl) not in used \
                        and similarity(fatwas[pl]["title"], f["title"]) < 0.6:
                    partner = pl
                    break
            if not partner:
                continue
            g = ask(client, GENERATOR, GEN_MULTI.format(a=fatwa_text(f), b=fatwa_text(fatwas[partner]),
                                                        categories=CATEGORIES))
            if not g:
                continue
            row = {"question_type": "multi_doc", "source_id": f"{fid}|{link_to_id(partner)}",
                   "source_title": f"{f['title']} | {fatwas[partner]['title']}",
                   "source_link": f"{link} | {partner}",
                   "source_text": fatwa_text(f) + "\n\n=== FATWA B ===\n\n" + fatwa_text(fatwas[partner]),
                   "difficulty": "hard", "generator_model": g.pop("_model"), **g}
        if not row.get("suitable"):
            print(f"   unsuitable: {row.get('reason')}")
            continue
        row["quote_found"] = quote_found(row.get("ground_truth", ""), row["source_text"])
        row["title_similarity"] = similarity(row["question"], row["source_title"])
        v = ask(client, VALIDATOR, VAL_SINGLE.format(question=row["question"],
                                                     ground_truth=row["ground_truth"],
                                                     fatwa=row["source_text"][:5000]))
        row["validator_ok"], row["validator_reason"] = (v or {}).get("ok"), (v or {}).get("reason", "")
        row["validator_model"] = (v or {}).get("_model", "")
        rows.append(row)
        used.update(row["source_id"].split("|"))

    for question, category in (NO_ANSWER_CANDIDATES if "no_answer" in args.types else []):
        print(f"[no_answer] {question[:40]}")
        top = search_fatwas(vs, question, max_distance=1.0)
        listing = "\n\n".join(f"[{i}] {f['title']}\n{f['chunks'][0][:1200]}" for i, f in enumerate(top, 1))
        v = ask(client, VALIDATOR, VAL_NO_ANSWER.format(question=question, fatwas=listing)) or {}
        rows.append({"question_type": "no_answer", "question": question, "ground_truth": "",
                     "source_id": "", "category": category,
                     "difficulty": "easy" if category == "Out-of-scope" else "medium",
                     "top1_distance": round(top[0]["distance"], 3) if top else None,
                     "closest_fatwas": " | ".join(f"{link_to_id(f['link'])}: {f['title'][:60]}" for f in top[:3]),
                     "validator_ok": not v.get("answered_by_any", True),
                     "validator_reason": v.get("reason", ""), "validator_model": v.get("_model", ""),
                     "reason": ""})

    df = pd.DataFrame(rows)
    start = 1 if previous is None else int(previous.cand_id.str[1:].astype(int).max()) + 1
    df.insert(0, "cand_id", [f"c{i:03d}" for i in range(start, start + len(df))])
    df["review_status"], df["review_notes"] = "", ""
    cols = ["cand_id", "question_type", "question", "ground_truth", "source_id", "category", "difficulty",
            "quote_found", "title_similarity", "validator_ok", "validator_reason",
            "generator_model", "validator_model", "top1_distance",
            "closest_fatwas", "source_title", "source_link", "reason", "review_status", "review_notes",
            "source_text"]
    df = df[[c for c in cols if c in df.columns]]
    if previous is not None:
        df = pd.concat([previous, df], ignore_index=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("\nDrafted (before human review):")
    print(df.groupby("question_type").agg(n=("cand_id", "size"),
                                          validator_ok=("validator_ok", "sum")).to_string())
    print(f"Targets: {TARGETS}\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
