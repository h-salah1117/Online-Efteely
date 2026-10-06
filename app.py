import streamlit as st
import os
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="إفتيلي", page_icon="🕌", layout="centered")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@300;400;600;700&display=swap');

:root {
    --green-deep:   #0d2b1f;
    --green-mid:    #14442e;
    --green-accent: #1e7a4a;
    --green-light:  #2aad6a;
    --gold:         #c9a84c;
    --gold-light:   #e5c97e;
    --cream:        #f7f3eb;
    --text-main:    #1a1a1a;
    --text-muted:   #5a6b62;
    --border:       rgba(200,168,76,0.25);
    --shadow:       0 4px 24px rgba(13,43,31,0.10);
    --radius:       16px;
}

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif !important;
    direction: rtl;
}

.stApp {
    background: var(--cream);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 760px;
    padding: 2rem 1.5rem 6rem;
}

.efteely-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.efteely-logo {
    font-family: 'Amiri', serif;
    font-size: 3rem;
    color: var(--green-mid);
    line-height: 1;
    margin-bottom: 0.25rem;
    letter-spacing: 2px;
}
.efteely-subtitle {
    font-size: 0.95rem;
    color: var(--text-muted);
    font-weight: 300;
    margin-top: 0.6rem;
}
.efteely-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 0.75rem auto 0;
}

.disclaimer {
    background: linear-gradient(135deg, #fff8e6 0%, #fef3d0 100%);
    border: 1px solid var(--gold);
    border-radius: var(--radius);
    padding: 0.75rem 1.25rem;
    margin-bottom: 1.5rem;
    font-size: 0.85rem;
    color: #7a5c00;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessage"] {
    border-radius: var(--radius) !important;
    margin-bottom: 0.75rem !important;
    padding: 1rem 1.25rem !important;
    box-shadow: var(--shadow) !important;
    border: 1px solid var(--border) !important;
    animation: fadeUp 0.3s ease both;
    background: #ffffff !important;
}

[data-testid="stChatMessage"] *,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: var(--text-main) !important;
    background: transparent !important;
}

div[data-testid="stChatMessage"]:has(> div > [data-testid="chatAvatarIcon-user"]) {
    background: #f0faf5 !important;
    border-left: 4px solid var(--green-accent) !important;
}

div[data-testid="stChatMessage"]:has(> div > [data-testid="chatAvatarIcon-assistant"]) {
    background: #fffdf7 !important;
    border-left: 4px solid var(--gold) !important;
}

[data-testid="chatAvatarIcon-user"] svg,
[data-testid="chatAvatarIcon-user"] {
    background: var(--green-accent) !important;
    color: #fff !important;
    fill: #fff !important;
}
[data-testid="chatAvatarIcon-assistant"] svg,
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--gold) !important;
    color: #fff !important;
    fill: #fff !important;
}

[data-testid="stChatInput"] {
    border-radius: 50px !important;
    border: 2px solid var(--green-accent) !important;
    background: #ffffff !important;
    box-shadow: 0 4px 20px rgba(30,122,74,0.12) !important;
    padding: 0.5rem 1.25rem !important;
    font-family: 'Cairo', sans-serif !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 4px 24px rgba(201,168,76,0.2) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Cairo', sans-serif !important;
    font-size: 1rem !important;
    color: var(--text-main) !important;
    -webkit-text-fill-color: var(--text-main) !important;
    caret-color: var(--green-accent) !important;
    direction: rtl !important;
}
/* Keep the input light even when the browser/OS is in dark mode */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    opacity: 1 !important;
}
[data-testid="stChatInput"] button svg {
    fill: var(--green-accent) !important;
    color: var(--green-accent) !important;
}
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {
    background: var(--cream) !important;
}

[data-testid="stSpinner"] {
    color: var(--green-accent) !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: #fafaf8 !important;
    margin-top: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
}

[data-testid="stExpander"] a {
    color: var(--green-accent) !important;
    text-decoration: none !important;
    font-size: 0.875rem;
    padding: 4px 0;
    display: inline-block;
    border-bottom: 1px dashed var(--border);
    transition: color 0.2s ease;
}
[data-testid="stExpander"] a:hover {
    color: var(--gold) !important;
}

[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border: none !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--green-accent); border-radius: 2px; }
</style>

<!-- Header -->
<div class="efteely-header">
    <div class="efteely-logo">🕌 إفتيلي</div>
    <div class="efteely-subtitle">مساعدك الفقهي الذكي — اسأل بكل ثقة</div>
    <div class="efteely-divider"></div>
</div>

<!-- Disclaimer -->
<div class="disclaimer">
    ⚠️ هذا البوت لأغراض تعليمية فقط · لا تعتمد عليه في مسائلك الدينية دون الرجوع لأهل العلم
</div>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_PATH     = "/tmp/chroma_db"
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
@st.cache_resource(show_spinner=False)
def load_rag():
    if not os.path.exists(DOWNLOAD_MARKER):
        os.makedirs(CHROMA_PATH, exist_ok=True)
        snapshot_download(
            repo_id="H-Salah/online-efteely-chroma",
            repo_type="dataset",
            local_dir=CHROMA_PATH,
            allow_patterns=["chroma_db/*"],
            token=st.secrets.get("HF_TOKEN", None)
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

try:
    with st.spinner("جاري تحميل محرك البحث الفقهي…"):
        vectorstore = load_rag()
except Exception as e:
    st.error(f"❌ خطأ في تشغيل النظام: {e}")
    st.stop()


def search_fatwas(query):
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

# ── LLM ──────────────────────────────────────────────────────────────────────
# FIX 1: validate secret exists before crashing with an unclear KeyError
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ GROQ_API_KEY مش موجود في الـ secrets — تأكد إنك أضفته في إعدادات المساحة")
    st.stop()

llm = ChatGroq(
    model=ANSWER_MODEL,
    temperature=0.1,
    groq_api_key=st.secrets["GROQ_API_KEY"]
)
router_llm = ChatGroq(
    model=ROUTER_MODEL,
    temperature=0,
    groq_api_key=st.secrets["GROQ_API_KEY"]
).with_fallbacks([llm])

# FIX 2: build prompt chain once at startup, not on every request
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     'أنت "إفتيلي"، مساعد شرعي ودود ومتخصص.\n'
     "قواعد الإجابة:\n"
     "- أجب فقط بناءً على الفتاوى المرفقة أدناه، ولا تضف أحكامًا أو أدلة من عندك.\n"
     "- ضع رقم الفتوى بين قوسين مربعين مثل [1] بعد كل معلومة مأخوذة منها.\n"
     "- إذا لم تجد في الفتاوى ما يجيب عن السؤال، فقل بوضوح إنك لم تجد فتوى في هذه المسألة، "
     "وانصح السائل بالرجوع إلى أهل العلم، ولا تخمّن.\n"
     "- إذا كانت الرسالة تحية أو شكرًا أو كلامًا عامًا، فرد بلطف واختصار.\n"
     "- أجب بنفس لغة السائل وأسلوبه (فصحى أو عامية).\n\n"
     "الفتاوى المتاحة:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
chain = prompt_template | llm | StrOutputParser()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.toast("✅ إفتيلي جاهز للرد على استفساراتكم", icon="🕌")

# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle new input ──────────────────────────────────────────────────────────
if prompt := st.chat_input("اكتب سؤالك هنا…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Conversation history (last 6 turns), excluding the current message
    # so it isn't repeated alongside {question}
    past = st.session_state.messages[:-1][-6:]
    history_messages = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in past
    ]
    full_history = "\n".join(
        f"{'المستخدم' if m['role'] == 'user' else 'إفتيلي'}: {m['content']}" for m in past
    ) or "لا يوجد."

    with st.chat_message("assistant"):
        # Intent detection + standalone question rewrite in a single call,
        # so retrieval embeds only the question instead of the whole history
        router_prompt = (
            f"Conversation history:\n{full_history}\n\n"
            f"Last user message: {prompt}\n\n"
            "If the last message is a religious/fiqh question, reply exactly:\n"
            "SEARCH: <the question rewritten in Arabic as a standalone question, using the history to resolve references>\n"
            "Otherwise (greeting, thanks, small talk), reply exactly: CHAT"
        )
        router_result = router_llm.invoke(router_prompt).content.strip()

        fatwas = []

        # Fallback: if the router output is malformed, anything that looks
        # like a question defaults to search
        if router_result.upper().startswith("SEARCH"):
            use_search   = True
            search_query = router_result.split(":", 1)[-1].strip() or prompt
        elif router_result.upper().startswith("CHAT"):
            use_search   = False
        else:
            use_search   = len(prompt.split()) > 3
            search_query = prompt

        if use_search:
            with st.spinner("جاري مراجعة الفتاوى…"):
                fatwas  = search_fatwas(search_query)
                context = format_context(fatwas)
        else:
            context = "لا يوجد سياق فقهي محدد لهذه الرسالة."

        response = st.write_stream(chain.stream({
            "context":  context,
            "question": prompt,
            "history":  history_messages
        }))

        # Sources expander — numbered to match the [n] citations in the answer
        if fatwas:
            with st.expander("📚 المصادر والمراجع"):
                for i, f in enumerate(fatwas, 1):
                    title = f["title"] or "رابط الفتوى"
                    if f["link"]:
                        st.markdown(f"[{i}] [{title} ↗]({f['link']})")
                    else:
                        st.markdown(f"[{i}] {title}")

    # FIX 4: removed st.rerun() — Streamlit reruns automatically after each interaction
    st.session_state.messages.append({"role": "assistant", "content": response})
