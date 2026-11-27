---
title: Online Efteely
emoji: 🕌
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: false
license: mit
short_description: RAG chatbot for Islamic questions with cited fatwas
---

# 🕌 Online Efteely

**A production RAG chatbot that answers Islamic (fiqh) questions from 50,000+ real fatwas, and cites the exact fatwas it used.**

<a href="https://huggingface.co/spaces/H-Salah/online-efteely"><img src="https://img.shields.io/badge/Try_it_live-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Try it live on Hugging Face"></a>
<a href="https://www.kaggle.com/datasets/hazemmosalah/50k-islamic-fatwa-q-and-a-dataset-arabic"><img src="https://img.shields.io/badge/Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Dataset on Kaggle"></a>

<img src="assets/demo.gif" width="720" alt="Efteely answering a question with cited sources">

## The problem

People search for religious rulings and get either generic chatbot answers with no source, or long forum threads. A general LLM will happily invent a ruling. Efteely only answers from real, published fatwas, shows the source for every claim, and says clearly when it found nothing instead of guessing.

## What it does

- **Answers from real sources only.** The model is instructed to use the retrieved fatwas and nothing else.
- **Cites every claim.** Answers carry `[1]`, `[2]` markers that match a numbered list of source links under the answer.
- **Understands follow-ups.** A small router model rewrites "and what about travelling?" into a standalone question before searching.
- **Skips search for small talk.** Greetings and thanks are answered directly, without a retrieval call.
- **Refuses to guess.** If no fatwa is close enough, it says so and points the user to a scholar.

## How it works

```mermaid
flowchart LR
    U[User question] --> R{Router LLM<br/>gpt-oss-20b}
    H[(Last 6 turns<br/>of chat history)] --> R
    R -- "CHAT" --> G
    R -- "SEARCH: standalone question" --> E[bge-m3<br/>embedding]
    E --> V[(ChromaDB<br/>fatwa chunks)]
    V --> F[Top 15 chunks<br/>→ drop distance > 0.40<br/>→ group into top 5 fatwas]
    F --> G[Answer LLM<br/>gpt-oss-120b via Groq]
    G --> A[Streamed answer<br/>+ numbered sources]
```

**Indexing (offline, once):** [`notebook/OFCB.ipynb`](notebook/OFCB.ipynb) splits each fatwa into ~1,200-character chunks (150 overlap) with the question prepended to every chunk, embeds them with `BAAI/bge-m3` on a Colab T4, and uploads the Chroma index to a [Hugging Face dataset](https://huggingface.co/datasets/H-Salah/online-efteely-chroma).

**Serving:** on startup the app downloads that index once, then handles each message in two LLM calls: one to the router, one for the answer.

## Tech stack

| Layer | Tools |
|:--|:--|
| Data | 50K+ fatwas scraped from Bin Baz, IslamQA and IslamWeb (BeautifulSoup), cleaned with pandas |
| Embeddings | `BAAI/bge-m3` (sentence-transformers), normalized, cosine distance |
| Vector store | ChromaDB, hosted as a Hugging Face dataset |
| LLMs | Groq: `gpt-oss-120b` (answers), `gpt-oss-20b` (routing, with fallback to the answer model) |
| Orchestration | LangChain (prompt templates, streaming, fallbacks) |
| Observability | LangSmith tracing of retrieval and LLM calls |
| App & deploy | Streamlit on Hugging Face Spaces, synced from GitHub by a GitHub Action |

## Run it locally

```bash
git clone https://github.com/h-salah1117/Online-Efteely.git
cd Online-Efteely
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-key"        # required
# HF_TOKEN = "..."                    # only if the index dataset is private
# LANGSMITH_API_KEY = "..."           # optional tracing
# LANGSMITH_TRACING = "true"
# LANGSMITH_PROJECT = "online-efteely"
```

```bash
streamlit run app.py
```

The first start downloads the Chroma index and the bge-m3 model, so it takes a few minutes.

The index is downloaded to `/tmp/chroma_db` by default. If you already have it locally, set `CHROMA_PATH` to the folder that contains `chroma_db/` and create an empty `.download_complete_bge-m3-chunks-v1` file in it to skip the download.

## Key settings (in `rag_core.py`)

| Setting | Value | Meaning |
|:--|:--|:--|
| `RETRIEVE_K` | 15 | chunks fetched per question |
| `TOP_FATWAS` | 5 | distinct fatwas passed to the LLM |
| `MAX_DISTANCE` | 0.40 | chunks farther than this are ignored (picked with `eval/distance_analysis.py`) |
| `INDEX_VERSION` | `bge-m3-chunks-v1` | bump to force a re-download after re-indexing |

## Project structure

```
├── app.py                         # Streamlit UI: chat, router, answer, sources
├── rag_core.py                    # Index download, embeddings, retrieval, prompts, settings
├── notebook/OFCB.ipynb            # Indexing pipeline (chunk → embed → upload)
├── eval/                          # Golden set, retrieval + LLM-judge answer eval (see eval/README.md)
├── requirements.txt
├── .streamlit/config.toml
└── .github/workflows/
    └── sync-to-hf-space.yml       # Deploys every push on main to the HF Space
```

## How it evolved

| | v1: [Efteely Offline](https://github.com/h-salah1117/Efteely-Islamic-Chatbot) | v2: Online Efteely |
|:--|:--|:--|
| Embeddings | CAMeL-Lab Arabic BERT, whole questions | bge-m3, chunked question + answer |
| Retrieval | Cosine top-5 over question embeddings | Chroma top-15 chunks, distance filter, grouped per fatwa |
| LLM | Gemma 3 (local, Ollama) | gpt-oss-120b on Groq |
| Conversation | Single question | History + standalone-question rewriting |
| Sources | Link to the top fatwa | Inline `[n]` citations for every fatwa used |
| Deployment | Local Flask app | Public Streamlit app, auto-deployed |

## Evaluation

Golden set v1: **79 questions**: 29 in Egyptian dialect, 14 in MSA, 10 that need two fatwas, 6 near-copies of fatwa titles, and 20 with no answer in the corpus. Questions were drafted by an LLM from random fatwas, checked by a second model family, and reviewed by hand. Full method, scripts and per-question results are in [eval/README.md](eval/README.md).

| | metric | result |
|:--|:--|:--|
| **Retrieval** (79 q) | hit@5, all 59 answerable questions | **83%** (strict labels: 80%) |
| | hit@5, Egyptian-dialect questions (29) | **79%** |
| | MRR | **0.68** |
| | out-of-corpus questions correctly rejected | **16/20** (4/20 at the old `MAX_DISTANCE=0.5`) |
| **Answers** (LLM judge, first 20 q) | correctness / faithfulness / citations (1–5) | **4.59 / 4.94 / 4.76** |
| | pass rate (correctness ≥ 4 and faithfulness ≥ 4) | **94%** |
| | says "no fatwa found" instead of guessing | **3/3** |

What the eval changed:
- `MAX_DISTANCE` went from 0.5 to **0.40**, chosen from a threshold sweep and re-checked on the larger set. It costs 2 points of hit@5 and rejects 4× more out-of-corpus questions.
- It caught a dead router model (`llama-3.1-8b-instant` was removed from Groq, so every request was silently falling back to the big model). It is replaced with `gpt-oss-20b`.
- It showed where dense retrieval breaks: dialect questions where the embedding matches the general topic but misses the specific point. It also showed that a distance threshold alone can't fully separate "no answer" from "hard answer".

⚠️ Early numbers, not a benchmark: hit@5 has a margin of about ±10 points at this size, and the answer eval still has to run on all 79 questions. Next: hybrid BM25 + a reranker, compared against dense retrieval on the same questions.

## Disclaimer

Efteely is for educational and technical purposes. Its answers are not religious rulings. Always confirm with a qualified scholar.

## License

MIT
