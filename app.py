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
        st.success("✅ تم تحميل قاعدة البيانات!")

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
                seen = set()
                for i, doc in enumerate(docs):
                    link = doc.metadata.get("link", "").strip()
                    source = doc.metadata.get("source", "").strip()
                    url = link or source

                    if not url or url in seen:
                        continue
                    seen.add(url)

                    st.markdown(f"**{i+1}.** [🔗 رابط الفتوى]({url})")

    st.session_state.messages.append({"role": "assistant", "content": response})