import streamlit as st
import os
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="إفتيلي", page_icon="🕌", layout="centered")

# =====================
# Custom CSS - matching offline UI
# =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

* {
    font-family: 'Tajawal', sans-serif !important;
    direction: rtl;
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 0 !important; max-width: 800px;}

/* ---- HEADER ---- */
.header-box {
    background: linear-gradient(135deg, #1a7a3c 0%, #25a550 60%, #2ecc71 100%);
    border-radius: 18px 18px 0 0;
    padding: 48px 32px 36px 32px;
    text-align: center;
    margin-bottom: 0;
    box-shadow: 0 4px 24px rgba(30,120,60,0.18);
}
.header-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #e8f5e9;
    letter-spacing: 2px;
    margin-bottom: 10px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.18);
}
.header-subtitle {
    font-size: 1.15rem;
    color: #c8e6c9;
    font-weight: 400;
}

/* ---- WARNING BOX ---- */
.warning-box {
    background: linear-gradient(90deg, #ff1a1a 0%, #cc0000 100%);
    color: white;
    border-radius: 10px;
    padding: 18px 24px;
    margin: 18px 0 10px 0;
    font-size: 1.05rem;
    font-weight: 700;
    text-align: center;
    line-height: 1.7;
    box-shadow: 0 4px 18px rgba(200,0,0,0.25);
    border: 2px solid #ff4444;
    animation: pulse-border 2s infinite;
}
@keyframes pulse-border {
    0%, 100% { box-shadow: 0 4px 18px rgba(200,0,0,0.25); }
    50% { box-shadow: 0 4px 32px rgba(200,0,0,0.55); }
}

/* ---- CHAT AREA ---- */
.chat-container {
    background: #f7f7f7;
    border-radius: 0 0 18px 18px;
    padding: 28px 24px 24px 24px;
    min-height: 200px;
}

/* ---- MESSAGES ---- */
.stChatMessage {
    border-radius: 14px !important;
    margin-bottom: 12px !important;
}
[data-testid="stChatMessageContent"] {
    font-size: 1.05rem !important;
    line-height: 1.8 !important;
}

/* ---- INPUT BOX ---- */
.stChatInputContainer {
    background: white !important;
    border-radius: 12px !important;
    border: 2px solid #25a550 !important;
    padding: 4px !important;
    margin-top: 12px !important;
}
.stChatInputContainer textarea {
    font-family: 'Tajawal', sans-serif !important;
    font-size: 1.05rem !important;
    direction: rtl !important;
}

/* ---- SEND BUTTON ---- */
.stChatInputContainer button {
    background: #25a550 !important;
    border-radius: 10px !important;
    color: white !important;
}
.stChatInputContainer button:hover {
    background: #1a7a3c !important;
}

/* ---- SOURCE EXPANDER ---- */
.streamlit-expanderHeader {
    font-family: 'Tajawal', sans-serif !important;
    color: #25a550 !important;
    font-weight: 700 !important;
    direction: rtl !important;
}

/* ---- SPINNER / INFO ---- */
.stSpinner > div {
    border-top-color: #25a550 !important;
}
.stAlert {
    border-radius: 10px !important;
    direction: rtl !important;
    font-family: 'Tajawal', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# =====================
# Header
# =====================
st.markdown("""
<div class="header-box">
    <div class="header-title">🕌 Efteely أفتيلي</div>
    <div class="header-subtitle">الإجابة على الأسئلة الشرعية باستخدام الذكاء الاصطناعي</div>
</div>
""", unsafe_allow_html=True)

# =====================
# Warning
# =====================
st.markdown("""
<div class="warning-box">
    ⚠️ تحذير هام ⚠️<br>
    هذا المشروع تم لغرض التعلم فقط، فلا يجب الاعتماد على أي من هذه الفتاوى في أمورك الدينية لأن الموديل قد يخطئ
</div>
""", unsafe_allow_html=True)

# =====================
# Backend
# =====================
CHROMA_PATH = "/tmp/chroma_db"
CHROMA_SUBDIR = os.path.join(CHROMA_PATH, "chroma_db")
DOWNLOAD_MARKER = os.path.join(CHROMA_PATH, ".download_complete")

@st.cache_resource(show_spinner=True)
def load_rag():
    if not os.path.exists(DOWNLOAD_MARKER):
        os.makedirs(CHROMA_PATH, exist_ok=True)
        with st.spinner("جاري تحميل قاعدة الفتاوى... هياخد شوية وقت في أول مرة"):
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

    count = vectorstore._collection.count()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    st.info(f"✅ تم تحميل {count} فتوى")
    return retriever

try:
    retriever = load_rag()
except Exception as e:
    st.error(f"❌ خطأ في تحميل قاعدة البيانات: {e}")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.25,
    groq_api_key=st.secrets["GROQ_API_KEY"]
)

prompt_template = PromptTemplate.from_template("""أنت مفتي وخبير شرعي موثوق. أجب على سؤال المستخدم بالعربية الفصحى الواضحة.
استخدم فقط الفتاوى المقدمة في السياق. كن موجزاً ومحترماً.
إذا لم يكن السؤال في السياق، قل ذلك بأدب.

السياق:
{context}

السؤال: {question}

الإجابة:""")

# =====================
# Chat
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 المصادر"):
                for i, url in enumerate(msg["sources"]):
                    st.markdown(f"**{i+1}.** [🔗 رابط الفتوى]({url})")

if prompt := st.chat_input("اكتب سؤالك الشرعي هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في الفتاوى..."):
            docs = retriever.invoke(prompt)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])

            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke({"context": context, "question": prompt})

            st.markdown(response)

            # Collect unique sources
            seen = set()
            sources = []
            for doc in docs:
                link = doc.metadata.get("link", "").strip()
                source = doc.metadata.get("source", "").strip()
                url = link or source
                if url and url not in seen:
                    seen.add(url)
                    sources.append(url)

            if sources:
                with st.expander("📚 المصادر"):
                    for i, url in enumerate(sources):
                        st.markdown(f"**{i+1}.** [🔗 رابط الفتوى]({url})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })