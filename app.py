import streamlit as st
import os
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ====================== CONFIG ======================
st.set_page_config(page_title="إفتيلي", page_icon="🕌", layout="centered")

st.title("🕌 إفتيلي - Islamic Chatbot")
st.caption("اسأل أي سؤال فقهي وهجاوبك من فتاوى موثوقة")

# Use persistent storage on HF Spaces
CHROMA_PATH = "/data/chroma_db"

# ====================== LOAD RAG ======================
@st.cache_resource(show_spinner=False)
def load_rag():
    # Download only if not exists in persistent storage
    if not os.path.exists(CHROMA_PATH) or len(os.listdir(CHROMA_PATH)) < 5:
        with st.spinner("جاري تحميل قاعدة الفتاوى لأول مرة (988MB)..."):
            snapshot_download(
                repo_id="H-Salah/online-efteely-chroma",
                repo_type="dataset",
                local_dir=CHROMA_PATH,
                allow_patterns=["*"]
            )
        st.success("✅ تم تحميل قاعدة البيانات!")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    st.info(f"✅ Loaded {vectorstore._collection.count()} documents")
    return vectorstore.as_retriever(search_kwargs={"k": 6})

retriever = load_rag()

# ====================== LLM ======================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.25,
    groq_api_key=st.secrets.get("GROQ_API_KEY")
)

# ====================== PROMPT ======================
prompt_template = PromptTemplate.from_template("""
You are a trusted Islamic scholar. 
Answer in clear Arabic using ONLY the provided fatwas.
Be concise, respectful, and accurate.

Context:
{context}

Question: {question}

Answer:
""")

# ====================== CHAT ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اكتب سؤالك هنا..."):
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
            
            with st.expander("📚 المصادر"):
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source", "غير معروف")
                    st.write(f"{i}. {source}")

    st.session_state.messages.append({"role": "assistant", "content": response})