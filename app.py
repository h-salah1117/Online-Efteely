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

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');

* { font-family: 'Cairo', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0a4d2e 0%, #0d6b3f 25%, #0f8a4f 50%, #0d6b3f 75%, #0a4d2e 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main .block-container {
    max-width: 1000px;
    margin: 20px auto;
    padding: 0 !important;
}

.header {
    background: linear-gradient(135deg, #0d6b3f 0%, #0f8a4f 50%, #11a85f 100%);
    color: white;
    padding: 60px 40px;
    text-align: center;
    border-radius: 24px 24px 0 0;
    margin-bottom: 0;
}

.header h1 {
    font-size: 3.2em;
    font-weight: 900;
    margin-bottom: 12px;
    text-shadow: 0 4px 12px rgba(0,0,0,0.3);
    background: linear-gradient(135deg, #ffffff 0%, #e8f5e9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.warning-banner {
    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
    color: white;
    padding: 25px 30px;
    text-align: center;
    font-size: 1.25em;
    font-weight: 700;
    line-height: 1.8;
    border-radius: 16px;
    margin: 20px 0;
    box-shadow: 0 4px 20px rgba(220, 53, 69, 0.4);
    border-right: 6px solid #b71c1c;
}

.content {
    background: white;
    padding: 50px 40px;
    border-radius: 0 0 24px 24px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
}

.stTextArea textarea {
    font-size: 1.15em !important;
    padding: 20px !important;
    border: 2px solid #e0e0e0 !important;
    border-radius: 16px !important;
    direction: rtl !important;
    min-height: 130px !important;
}

.stButton button {
    width: 100% !important;
    padding: 18px !important;
    font-size: 1.25em !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #0f8a4f 0%, #0d6b3f 100%) !important;
    color: white !important;
    border-radius: 16px !important;
    margin-top: 15px;
    box-shadow: 0 4px 14px rgba(15,138,79,0.3) !important;
}

.answer-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    border-right: 6px solid #0f8a4f;
    border-radius: 20px;
    padding: 32px;
    margin: 25px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

.original-card {
    background: linear-gradient(135deg, #fff9e6 0%, #fffbf0 100%);
    border: 2px solid #ffc107;
    border-radius: 20px;
    padding: 28px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("""
<div class="header">
    <h1>Efteely أفتيلي</h1>
    <p>الإجابة على الأسئلة الشرعية باستخدام الذكاء الاصطناعي</p>
</div>
""", unsafe_allow_html=True)

# ====================== LARGE RED WARNING ======================
st.markdown("""
<div class="warning-banner">
    ⚠️⚠️⚠️ هذا المشروع تم لأغراض التعلم فقط ⚠️⚠️⚠️<br>
    فلا يجب الاعتماد على أي من هذه الفتاوى في أمورك الدينية<br>
    لأن الموديل قد يخطئ أو يعطي معلومات غير دقيقة<br>
    يرجى استشارة عالم دين موثوق في كل الأمور الشرعية
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
    return vectorstore.as_retriever(search_kwargs={"k": 6})

retriever = load_rag()

# ====================== LLM & PROMPT ======================
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

# ====================== USER INPUT ======================
question = st.text_area("أدخل سؤالك الشرعي:", 
                       placeholder="مثال: ما حكم صيام يوم الجمعة؟", 
                       height=130)

if st.button("إرسال السؤال", type="primary"):
    if not question.strip():
        st.error("يرجى كتابة السؤال")
    else:
        with st.spinner("جاري معالجة السؤال..."):
            docs = retriever.invoke(question)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke({"context": context, "question": question})
            
            # Display Answer
            st.markdown(f"""
            <div class="answer-card">
                <h3>✨ الإجابة المولدة</h3>
                <div style="font-size: 1.1em; line-height: 2;">{response}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sources
            with st.expander("📚 إظهار الفتاوى الأصلية والمصادر"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**{i}.** {doc.metadata.get('source', 'غير معروف')}")
                    if doc.metadata.get("link"):
                        st.markdown(f"[رابط المصدر]({doc.metadata['link']})")
                    st.write(doc.page_content)
                    st.divider()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 0.95em;'>Efteely - مشروع تعليمي فقط</p>", unsafe_allow_html=True)