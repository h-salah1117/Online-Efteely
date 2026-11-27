from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import snapshot_download
import os
from langsmith import traceable



# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", "/tmp/chroma_db")
CHROMA_SUBDIR   = os.path.join(CHROMA_PATH, "chroma_db")

# Bump when the index on HF is rebuilt, so a stale /tmp copy is re-downloaded
INDEX_VERSION   = "bge-m3-chunks-v1"
DOWNLOAD_MARKER = os.path.join(CHROMA_PATH, f".download_complete_{INDEX_VERSION}")

# Must match the indexing notebook
EMBEDDING_MODEL = "BAAI/bge-m3"

ANSWER_MODEL    = "openai/gpt-oss-120b"
ROUTER_MODEL    = "openai/gpt-oss-20b"     # small & fast; falls back to ANSWER_MODEL on error
                                           # (llama-3.1-8b-instant was removed from Groq: 404)

RETRIEVE_K      = 15     # chunks fetched from Chroma
TOP_FATWAS      = 5      # distinct fatwas passed to the LLM
MAX_DISTANCE    = 0.40   # cosine distance; chunks farther than this are ignored.
                         # Chosen with eval/distance_analysis.py: same hit@5 as 0.5, but rejects
                         # all no_answer questions (margin is thin: re-check when the golden set grows)

# ── Load RAG ─────────────────────────────────────────────────────────────────
def load_rag():
    if not os.path.exists(DOWNLOAD_MARKER):
        os.makedirs(CHROMA_PATH, exist_ok=True)
        snapshot_download(
            repo_id="H-Salah/online-efteely-chroma",
            repo_type="dataset",
            local_dir=CHROMA_PATH,
            allow_patterns=["chroma_db/*"],
            token=os.getenv("HF_TOKEN", None)
        )
        with open(DOWNLOAD_MARKER, "w") as f:
            f.write("done")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        persist_directory=CHROMA_SUBDIR,
        embedding_function=embeddings
    )



@traceable(name="search_fatwas", run_type="retriever")
def search_fatwas(vectorstore, query, max_distance=None):
    """Return up to TOP_FATWAS fatwas, each with its retrieved chunks merged.

    The index stores several chunks per fatwa, so results are grouped by link
    (keeping best-match order) and chunks beyond max_distance (default
    MAX_DISTANCE) are dropped. Each fatwa carries "distance": the cosine
    distance of its closest chunk, which the eval uses to tune the threshold.
    """
    if max_distance is None:
        max_distance = MAX_DISTANCE
    results = vectorstore.similarity_search_with_score(query, k=RETRIEVE_K)
    fatwas = {}
    for doc, distance in results:
        if distance > max_distance:
            continue
        link = (doc.metadata.get("link") or doc.metadata.get("source") or "").strip()
        key  = link or doc.page_content[:100]
        if key not in fatwas:
            if len(fatwas) == TOP_FATWAS:
                continue
            # results come sorted by distance, so the first chunk is the closest
            fatwas[key] = {"title": doc.metadata.get("title", ""), "link": link,
                           "distance": float(distance), "chunks": []}
        fatwas[key]["chunks"].append(doc.page_content)
    return list(fatwas.values())


def format_context(fatwas):
    if not fatwas:
        return "لا توجد فتاوى ذات صلة بهذا السؤال."
    blocks = []
    for i, f in enumerate(fatwas, 1):
        blocks.append(f"[{i}] {f['title']}\n" + "\n...\n".join(f["chunks"]))
    return "\n\n---\n\n".join(blocks)


# ── Prompts (shared by app.py and eval/) ─────────────────────────────────────
ANSWER_SYSTEM_PROMPT = (
    'أنت "إفتيلي"، مساعد شرعي ودود ومتخصص.\n'
    "قواعد الإجابة:\n"
    "- أجب فقط بناءً على الفتاوى المرفقة أدناه، ولا تضف أحكامًا أو أدلة من عندك.\n"
    "- ضع رقم الفتوى بين قوسين مربعين مثل [1] بعد كل معلومة مأخوذة منها.\n"
    "- إذا لم تجد في الفتاوى ما يجيب عن السؤال، فقل بوضوح إنك لم تجد فتوى في هذه المسألة، "
    "وانصح السائل بالرجوع إلى أهل العلم، ولا تخمّن.\n"
    "- إذا كانت الرسالة تحية أو شكرًا أو كلامًا عامًا، فرد بلطف واختصار.\n"
    "- أجب بنفس لغة السائل وأسلوبه (فصحى أو عامية).\n\n"
    "الفتاوى المتاحة:\n{context}"
)


def build_answer_prompt():
    """Prompt with {context}, {history} (list of messages) and {question}."""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    return ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])


def build_router_prompt(history_text, message):
    """Intent detection + standalone-question rewrite in a single call."""
    return (
        f"Conversation history:\n{history_text}\n\n"
        f"Last user message: {message}\n\n"
        "If the last message is a religious/fiqh question, reply exactly:\n"
        "SEARCH: <the question rewritten in Arabic as a standalone question, using the history to resolve references>\n"
        "Otherwise (greeting, thanks, small talk), reply exactly: CHAT"
    )


def parse_router_output(router_result, message):
    """Return (use_search, search_query). Malformed output defaults to search."""
    text = (router_result or "").strip()
    if text.upper().startswith("SEARCH"):
        return True, text.split(":", 1)[-1].strip() or message
    if text.upper().startswith("CHAT"):
        return False, None
    return True, message
