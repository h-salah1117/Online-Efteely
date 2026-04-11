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

# Use /tmp for writable storage on HuggingFace Spaces
CHROMA_PATH = "/tmp/chroma_db"
DOWNLOAD_MARKER = "/tmp/chroma_db/.download_complete"

@st.cache_resource(show_spinner=True)
def load_rag():
    # Check if already fully downloaded using a marker file
    if not os.path.exists(DOWNLOAD_MARKER):
        os.makedirs(CHROMA_PATH, exist_ok=True)
        with st.spinner("جاري تحميل قاعدة الفتاوى (988MB)... هياخد شوية وقت في أول مرة"):
            snapshot_download(
                repo_id="H-Salah/online-efteely-chroma",
                repo_type="dataset",
                local_dir=CHROMA_PATH,
                allow_patterns=["*"],
                token=st.secrets.get("HF_TOKEN", None)  # Add HF_TOKEN to secrets
            )
        # Write marker only after successful download
        with open(DOWNLOAD_MARKER, "w") as f:
            f.write("done")
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    count = vectorstore._collection.count()
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

prompt_template = PromptTemplate.from_template("""You are a trusted Islamic scholar. Answer in clear Arabic.
Use ONLY the provided fatwas. Be concise and respectful.

Context:
{context}

Question: {question}

Answer:""")

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

            

            with st.expander("📚 المصادر"):
                # DEBUG - just for one doc
                st.write("🔍 Metadata keys:", docs[0].metadata)

    st.session_state.messages.append({"role": "assistant", "content": response})