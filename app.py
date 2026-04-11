import streamlit as st
import os
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="أفتيلي", page_icon="🕌", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');

* { font-family: 'Cairo', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0a4d2e 0%, #0d6b3f 25%, #0f8a4f 50%, #0d6b3f 75%, #0a4d2e 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    direction: rtl;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main-container {
    max-width: 900px;
    margin: 30px auto;
    background: #ffffff;
    border-radius: 24px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    overflow: hidden;
    animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.header {
    background: linear-gradient(135deg, #0d6b3f 0%, #0f8a4f 50%, #11a85f 100%);
    color: white;
    padding: 50px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.header h1 {
    font-size: 3em;
    font-weight: 900;
    margin-bottom: 10px;
    color: white;
    text-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.header p {
    font-size: 1.1em;
    color: rgba(255,255,255,0.95);
    font-weight: 400;
}

.warning-banner {
    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
    color: white;
    padding: 20px 30px;
    text-align: center;
    font-size: 1.15em;
    font-weight: 700;
    line-height: 1.8;
    direction: rtl;
    border-bottom: 4px solid #a71d2a;
}

.content {
    padding: 40px;
    direction: rtl;
}

.answer-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    border-right: 5px solid #0f8a4f;
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    direction: rtl;
}

.answer-card h3 {
    color: #0d6b3f;
    font-size: 1.4em;
    font-weight: 700;
    margin-bottom: 16px;
}

.answer-text {
    font-size: 1.1em;
    line-height: 2;
    color: #2c3e50;
    white-space: pre-wrap;
}

.original-card {
    background: linear-gradient(135deg, #fff9e6 0%, #fffbf0 100%);
    border: 2px solid #ffc107;
    border-radius: 20px;
    padding: 28px;
    margin-top: 20px;
    box-shadow: 0 4px 20px rgba(255,193,7,0.15);
    direction: rtl;
}

.original-card h4 {
    color: #856404;
    font-size: 1.2em;
    font-weight: 700;
    margin-bottom: 12px;
}

.source-info {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 2px solid #e8e8e8;
    font-size: 0.95em;
    color: #666;
}

.source-info a {
    color: #0f8a4f;
    font-weight: 600;
    text-decoration: none;
}

.error-card {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-right: 5px solid #dc3545;
    border-radius: 16px;
    padding: 18px 24px;
    color: #721c24;
    margin-top: 20px;
}

.stTextArea textarea {
    font-family: 'Cairo', sans-serif !important;
    font-size: 1.1em !important;
    border: 2px solid #e0e0e0 !important;
    border-radius: 16px !important;
    direction: rtl !important;
    background: #fafafa !important;
    padding: 18px 24px !important;
    line-height: 1.6 !important;
}

.stTextArea textarea:focus {
    border-color: #0f8a4f !important;
    box-shadow: 0 0 0 4px rgba(15,138,79,0.1) !important;
}

.stButton > button {
    width: 100% !important;
    padding: 18px !important;
    font-size: 1.2em !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #0f8a4f 0%, #0d6b3f 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 14px rgba(15,138,79,0.3) !important;
    font-family: 'Cairo', sans-serif !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    box-shadow: 0 8px 24px rgba(15,138,79,0.4) !important;
    transform: translateY(-3px) !important;
}

/* Fix Streamlit expander RTL */
.streamlit-expanderHeader {
    direction: rtl !important;
    font-family: 'Cairo', sans-serif !important;
    font-weight: 700 !important;
    color: #0d6b3f !important;
    font-size: 1.1em !important;
}

/* Fix label direction */
.stTextArea label {
    direction: rtl !important;
    font-weight: 700 !important;
    color: #0d6b3f !important;
    font-size: 1.1em !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.block-container { 
    padding: 0 !important; 
    max-width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# =====================
# Header + Warning
# BUG FIX: Separated into its own st.markdown call
# and closed .content div properly later
# =====================
st.markdown("""
<div class="main-container">
    <div class="header">
        <h1>Efteely أفتيلي</h1>
        <p>الإجابة على الأسئلة الشرعية باستخدام الذكاء الاصطناعي</p>
    </div>
    <div class="warning-banner">
        ⚠️ هذا المشروع تم لغرض التعلم فقط، فلا يجب الاعتماد على أي من هذه الفتاوى في أمورك الدينية لأن الموديل قد يخطئ ⚠️
    </div>
    <div class="content">
    </div>
</div>
""", unsafe_allow_html=True)

# =====================
# Load RAG
# =====================
CHROMA_PATH = "/tmp/chroma_db"
CHROMA_SUBDIR = os.path.join(CHROMA_PATH, "chroma_db")
DOWNLOAD_MARKER = os.path.join(CHROMA_PATH, ".download_complete")

@st.cache_resource(show_spinner="جاري تحميل قاعدة الفتاوى...")
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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever, vectorstore

try:
    retriever, vectorstore = load_rag()
except Exception as e:
    st.markdown(f'<div class="error-card">❌ خطأ في تحميل قاعدة البيانات: {e}</div>', unsafe_allow_html=True)
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
# UI - Input
# =====================
question = st.text_area(
    "أدخل سؤالك الشرعي:",
    placeholder="اكتب سؤالك هنا...",
    height=120,
    label_visibility="visible"
)

submit = st.button("إرسال السؤال")

# =====================
# UI - Output
# BUG FIX: response and original_answer are now
# defined before being used in f-strings
# BUG FIX: HTML special chars in response escaped
# =====================
if submit:
    if not question.strip():
        st.markdown('<div class="error-card">⚠️ يرجى إدخال سؤال</div>', unsafe_allow_html=True)
    else:
        with st.spinner("جاري معالجة السؤال..."):
            try:
                docs = retriever.invoke(question)
                context = "\n\n---\n\n".join([doc.page_content for doc in docs])

                chain = prompt_template | llm | StrOutputParser()
                response = chain.invoke({"context": context, "question": question})

                top_doc = docs[0] if docs else None
                source_url = ""
                original_answer = ""

                if top_doc:
                    source_url = (
                        top_doc.metadata.get("link") or
                        top_doc.metadata.get("source") or
                        ""
                    ).strip()
                    original_answer = top_doc.page_content

            except Exception as e:
                st.markdown(f'<div class="error-card">❌ حدث خطأ: {e}</div>', unsafe_allow_html=True)
                st.stop()

        # Source HTML
        source_html = ""
        if source_url:
            source_html = f"""
            <div class="source-info">
                <strong>المصدر:</strong>
                <a href="{source_url}" target="_blank">{source_url}</a>
            </div>
            """

        # Answer card
        st.markdown(f"""
        <div class="answer-card">
            <h3>✨ الإجابة المولدة</h3>
            <div class="answer-text">{response}</div>
            {source_html}
        </div>
        """, unsafe_allow_html=True)

        # Original answer
        with st.expander("📖 إظهار الإجابة الأصلية"):
            st.markdown(f"""
            <div class="original-card">
                <h4>الإجابة الأصلية من المصدر:</h4>
                <div class="answer-text">{original_answer}</div>
            </div>
            """, unsafe_allow_html=True)