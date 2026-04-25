import streamlit as st
import os
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
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
    direction: rtl !important;
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
DOWNLOAD_MARKER = os.path.join(CHROMA_PATH, ".download_complete")

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
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_SUBDIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})

try:
    with st.spinner("جاري تحميل محرك البحث الفقهي…"):
        retriever = load_rag()
except Exception as e:
    st.error(f"❌ خطأ في تشغيل النظام: {e}")
    st.stop()

# ── LLM ──────────────────────────────────────────────────────────────────────
# FIX 1: validate secret exists before crashing with an unclear KeyError
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ GROQ_API_KEY مش موجود في الـ secrets — تأكد إنك أضفته في إعدادات المساحة")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=st.secrets["GROQ_API_KEY"]
)

# FIX 2: build prompt chain once at startup, not on every request
prompt_template = PromptTemplate.from_template(
    'أنت "إفتيلي"، خبير شرعي ودود ومتخصص. أجب بناءً على تاريخ المحادثة والفتاوى المتاحة فقط.\n'
    'تاريخ المحادثة:\n{chat_history}\n\n'
    'الفتاوى المتاحة:\n{context}\n\n'
    'السؤال الحالي: {question}\n\n'
    'إجابة إفتيلي:'
)
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

    # Build conversation history (last 6 turns)
    history_lines = []
    for m in st.session_state.messages[-6:]:
        role = "المستخدم" if m["role"] == "user" else "إفتيلي"
        history_lines.append(f"{role}: {m['content']}")
    full_history = "\n".join(history_lines)

    with st.chat_message("assistant"):
        # Intent detection
        intent_query = (
            f"Based on the conversation history:\n{full_history}\n"
            "Is the last message a specific religious question? Answer 'search' or 'chat'."
        )
        intent_result = llm.invoke(intent_query).content.strip().lower()

        context = ""
        docs    = []

        # FIX 3: improved fallback — short greetings still go through chat path,
        # but anything that looks like a question defaults to search
        use_search = "search" in intent_result or (
            "chat" not in intent_result and len(prompt.split()) > 3
        )

        if use_search:
            search_query = f"{full_history}\nQuestion: {prompt}"
            with st.spinner("جاري مراجعة الفتاوى…"):
                docs    = retriever.invoke(search_query)
                context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        else:
            context = "لا يوجد سياق فقهي محدد لهذه الرسالة."

        response = chain.invoke({
            "context":      context,
            "question":     prompt,
            "chat_history": full_history
        })

        st.markdown(response)

        # Sources expander
        if use_search and docs:
            with st.expander("📚 المصادر والمراجع"):
                urls = set()
                for doc in docs:
                    u = doc.metadata.get("link", doc.metadata.get("source", "")).strip()
                    if u and u not in urls:
                        st.markdown(f"- [رابط الفتوى ↗]({u})")
                        urls.add(u)

    # FIX 4: removed st.rerun() — Streamlit reruns automatically after each interaction
    st.session_state.messages.append({"role": "assistant", "content": response})