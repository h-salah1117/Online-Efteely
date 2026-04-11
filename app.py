import streamlit as st
import os
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Efteely أفتيلي",
    page_icon="🕌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ====================== CUSTOM CSS (محسنة ومنظفة) ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');

* { font-family: 'Cairo', sans-serif !important; direction: rtl; }

.stApp {
    background: linear-gradient(135deg, #0a4d2e 0%, #0d6b3f 50%, #0f8a4f 100%);
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden !important; }

/* Main Container */
.main .block-container {
    max-width: 850px;
    margin: 15px auto;
    padding: 0 !important;
}

/* Header */
.header-box {
    background: linear-gradient(135deg, #1a7a3c 0%, #25a550 60%, #2ecc71 100%);
    border-radius: 20px 20px 0 0;
    padding: 55px 30px 40px 30px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.header-title {
    font-size: 2.9rem;
    font-weight: 900;
    color: white;
    margin-bottom: 8px;
    text-shadow: 0 3px 15px rgba(0,0,0,0.3);
}

.header-subtitle {
    font-size: 1.25rem;
    color: #c8e6c9;
    font-weight: 500;
}

/* Big Red Warning */
.warning-box {
    background: linear-gradient(135deg, #d32f2f, #b71c1c);
    color: white;
    border-radius: 16px;
    padding: 22px 25px;
    margin: 20px 0 25px 0;
    font-size: 1.15rem;
    font-weight: 700;
    line-height: 1.75;
    text-align: center;
    box-shadow: 0 6px 20px rgba(211, 47, 47, 0.4);
    border: 3px solid #ff5252;
}

/* Chat Area */
.chat-area {
    background: #ffffff;
    border-radius: 0 0 20px 20px;
    padding: 30px 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

/* Input Box */
.stChatInputContainer {
    background: white !important;
    border: 2px solid #25a550 !important;
    border-radius: 15px !important;
    padding: 5px !important;
}

.stChatInputContainer textarea {
    font-size: 1.1rem !important;
    padding: 15px 20px !important;
}

/* Send Button */
.stChatInputContainer button {
    background: #25a550 !important;
    color: white !important;
    border-radius: 12px !important;
}

.stChatInputContainer button:hover {
    background: #1a7a3c !important;
}
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("""
<div class="header-box">
    <div class="header-title">🕌 Efteely أفتيلي</div>
    <div class="header-subtitle">الإجابة على الأسئلة الشرعية باستخدام الذكاء الاصطناعي</div>
</div>
""", unsafe_allow_html=True)

# ====================== LARGE RED WARNING ======================
st.markdown("""
<div class="warning-box">
    ⚠️ <strong>تحذير هام جداً</strong> ⚠️<br><br>
    هذا المشروع تم لأغراض التعلم فقط.<br>
    فلا يجب الاعتماد على أي من هذه الفتاوى في أمورك الدينية<br>
    لأن الموديل قد يخطئ أو يعطي معلومات غير دقيقة.<br>
    <strong>يرجى استشارة عالم دين موثوق في كل الأمور الشرعية.</strong>
</div>
""", unsafe_allow_html=True)

# ====================== RAG LOADING ======================
CHROMA_PATH = "./chroma_db"

@st.cache_resource(show_spinner="جاري تحميل قاعدة الفتاوى...")
def load_rag():
    if not os.path.exists(CHROMA_PATH) or len(os.listdir(CHROMA_PATH)) < 5:
        snapshot_download(
            repo_id="H-Salah/online-efteely-chroma",
            repo_type="dataset",
            local_dir=CHROMA_PATH,
            allow_patterns=["*"]
        )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = load_rag()

# ====================== LLM ======================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.25,
    groq_api_key=st.secrets.get("GROQ_API_KEY")
)

prompt_template = PromptTemplate.from_template("""
أنت مفتي إسلامي موثوق. أجب على السؤال بالعربية بوضوح واختصار باستخدام الفتاوى المقدمة فقط.
كن محترماً ودقيقاً.

السياق:
{context}

السؤال: {question}

الإجابة:
""")

# ====================== CHAT INTERFACE ======================
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
            
            # Sources
            with st.expander("📚 عرض المصادر والفتاوى الأصلية"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**{i}.** {doc.metadata.get('source', 'غير معروف')}")
                    if doc.metadata.get("link"):
                        st.markdown(f"[🔗 رابط الفتوى]({doc.metadata['link']})")
                    st.write(doc.page_content)
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown('</div>', unsafe_allow_html=True)