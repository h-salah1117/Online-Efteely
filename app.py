import streamlit as st
import os
import shutil
from huggingface_hub import snapshot_download, login

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ====================== HF SPACES CACHE FIX ======================
os.environ["HF_HOME"] = "/data/.huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/.huggingface"
os.environ["HF_HUB_CACHE"] = "/data/.huggingface"

# optional but recommended (set in Space secrets)
if "HF_TOKEN" in st.secrets:
    login(token=st.secrets["HF_TOKEN"])


# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="إفتيلي - Islamic Chatbot",
    page_icon="🕌",
    layout="centered"
)

st.title("🕌 إفتيلي - Islamic Chatbot")
st.caption("اسأل أي سؤال فقهي وهجاوبك من فتاوى موثوقة")


# ====================== EMBEDDINGS (CACHED) ======================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="/data/.huggingface"
    )


# ====================== DOWNLOAD CHROMA DB ======================
@st.cache_resource
def download_and_load_chroma():
    chroma_path = "/data/chroma_db"

    if not os.path.exists(chroma_path) or len(os.listdir(chroma_path)) < 3:
        st.info("جاري تحميل قاعدة البيانات لأول مرة... (ده هياخد دقايق)")
        with st.spinner("Downloading chroma_db from Hugging Face (817MB)..."):
            snapshot_download(
                repo_id="H-Salah/online-efteely-chroma",
                repo_type="dataset",
                local_dir=chroma_path,
                allow_patterns=["*"]
            )
        st.success("✅ تم تحميل قاعدة البيانات بنجاح!")

    embeddings = load_embeddings()

    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})


retriever = download_and_load_chroma()


# ====================== LLM ======================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.25,
    groq_api_key=st.secrets.get("GROQ_API_KEY")
)


# ====================== PROMPT ======================
prompt_template = PromptTemplate.from_template("""
You are a trusted Islamic scholar.  
Answer in clear Arabic (Fusha or simple Egyptian dialect when suitable). 
Use ONLY the provided fatwas. Be concise, respectful, and accurate. 
If the context doesn't have a clear answer, say so politely. 

Context:
{context}

Question:
{question}

Answer:
""")


# ====================== CHAT HISTORY ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ====================== USER INPUT ======================
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في الفتاوى..."):
            docs = retriever.invoke(prompt)

            context = "\n\n---\n\n".join(
                [doc.page_content for doc in docs]
            )

            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke({
                "context": context,
                "question": prompt
            })

            st.markdown(response)

            # sources
            with st.expander("📚 المصادر"):
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source", "غير معروف")
                    link = doc.metadata.get("link", "")

                    st.write(f"{i}. {source}")
                    if link:
                        st.write(f"🔗 {link}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )