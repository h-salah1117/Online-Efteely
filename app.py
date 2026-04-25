import streamlit as st
import os
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="إفتيلي", page_icon="🕌", layout="centered")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="st-"] {
        font-family: 'Cairo', sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 2rem;
        background-color: #0d6b3f;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 5px;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1 style="color: white !important;">🕌 إفتيلي</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">مساعدك الذكي للإجابة على الأسئلة الشرعية</p>
    </div>
    """, unsafe_allow_html=True)

st.warning("⚠️ البوت معمول لغرض تعليمي فقط لا تعتمد عليه في أمورك الدينية❌")

CHROMA_PATH = "/tmp/chroma_db"
CHROMA_SUBDIR = os.path.join(CHROMA_PATH, "chroma_db")
DOWNLOAD_MARKER = os.path.join(CHROMA_PATH, ".download_complete")

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
    vectorstore = Chroma(persist_directory=CHROMA_SUBDIR, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

try:
    with st.spinner("جاري التحميل..."):
        retriever = load_rag()
except Exception as e:
    st.error(f"❌ خطأ: {e}")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=st.secrets["GROQ_API_KEY"]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history_text = "\n".join([f"{'المستخدم' if m['role']=='user' else 'إفتيلي'}: {m['content']}" for m in st.session_state.messages[-5:]])

    with st.chat_message("assistant"):
        intent_query = f"Conversation:\n{history_text}\nLast message: {prompt}\nIs this a specific religious question? Answer 'search' or 'chat' only."
        intent = llm.invoke(intent_query).content.strip().lower()

        context = ""
        docs = []
        if "search" in intent:
            with st.spinner("جاري مراجعة المصادر..."):
                docs = retriever.invoke(f"{history_text}\n{prompt}")
                context = "\n\n---\n\n".join([d.page_content for d in docs])

        prompt_template = PromptTemplate.from_template("""أنت "إفتيلي"، خبير شرعي ومساعد ذكي.
        أجب بلباقة بناءً على تاريخ المحادثة والفتاوى المتاحة.
        
        التاريخ: {chat_history}
        السياق: {context}
        السؤال: {question}
        الإجابة:""")

        response = (prompt_template | llm | StrOutputParser()).invoke({
            "context": context,
            "question": prompt,
            "chat_history": history_text
        })

        st.markdown(response)
        
        if docs and "search" in intent:
            with st.expander("📚 المصادر المعتمدة"):
                urls = {d.metadata.get("link", d.metadata.get("source", "")) for d in docs}
                for u in urls:
                    if u: st.markdown(f"- [رابط الفتوى]({u})")

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()