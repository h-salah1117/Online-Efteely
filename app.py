import streamlit as st
import os
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="إفتيلي", page_icon="🕌", layout="centered")

st.title("🕌 إفتيلي - Islamic Chatbot")
st.caption("اسأل أي سؤال فقهي")

CHROMA_PATH = "./chroma_db"   # نستخدم ./ بدل /data عشان يكون أكثر استقراراً

@st.cache_resource(show_spinner=True)
def load_rag():
    if not os.path.exists(CHROMA_PATH) or len(os.listdir(CHROMA_PATH)) < 5:
        with st.spinner("جاري تحميل قاعدة الفتاوى (988MB)... هياخد شوية وقت في أول مرة"):
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

    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    st.info(f"✅ Vectorstore loaded with {vectorstore._collection.count()} documents")
    return retriever

retriever = load_rag()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.25,
    groq_api_key=st.secrets.get("GROQ_API_KEY")
)

prompt_template = PromptTemplate.from_template("""
You are a trusted Islamic scholar. Answer in clear Arabic.
Use ONLY the provided fatwas. Be concise and respectful.

Context:
{context}

Question: {question}

Answer:
""")

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
        with st.spinner("جاري البحث..."):
            docs = retriever.invoke(prompt)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke({"context": context, "question": prompt})
            
            st.markdown(response)
            
            with st.expander("المصادر"):
                for doc in docs:
                    st.write(doc.metadata.get("source", "غير معروف"))

    st.session_state.messages.append({"role": "assistant", "content": response})