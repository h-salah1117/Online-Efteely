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
ROUTER_MODEL    = "llama-3.1-8b-instant"   # small & fast; falls back to ANSWER_MODEL on error

RETRIEVE_K      = 15     # chunks fetched from Chroma
TOP_FATWAS      = 5      # distinct fatwas passed to the LLM
MAX_DISTANCE    = 0.5    # cosine distance; chunks farther than this are ignored (tune with the notebook's sanity check)

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
def search_fatwas(vectorstore, query):
    """Return up to TOP_FATWAS fatwas, each with its retrieved chunks merged.

    The index stores several chunks per fatwa, so results are grouped by link
    (keeping best-match order) and chunks beyond MAX_DISTANCE are dropped.
    """
    results = vectorstore.similarity_search_with_score(query, k=RETRIEVE_K)
    fatwas = {}
    for doc, distance in results:
        if distance > MAX_DISTANCE:
            continue
        link = (doc.metadata.get("link") or doc.metadata.get("source") or "").strip()
        key  = link or doc.page_content[:100]
        if key not in fatwas:
            if len(fatwas) == TOP_FATWAS:
                continue
            fatwas[key] = {"title": doc.metadata.get("title", ""), "link": link, "chunks": []}
        fatwas[key]["chunks"].append(doc.page_content)
    return list(fatwas.values())


def format_context(fatwas):
    if not fatwas:
        return "لا توجد فتاوى ذات صلة بهذا السؤال."
    blocks = []
    for i, f in enumerate(fatwas, 1):
        blocks.append(f"[{i}] {f['title']}\n" + "\n...\n".join(f["chunks"]))
    return "\n\n---\n\n".join(blocks)
