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
st.warning("⚠️ البوت معمول لغرض تعليمي فقط لا تعتمد عليه في أمرك الدينية❌")

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
    with st.spinner("جاري تشغيل محرك البحث..."):
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
    st.toast("✅ جاهز للرد على استفساراتكم", icon="🕌")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history_lines = []
    for m in st.session_state.messages[-6:]:
        role = "المستخدم" if m["role"] == "user" else "إفتيلي"
        history_lines.append(f"{role}: {m['content']}")
    full_history = "\n".join(history_lines)

    with st.chat_message("assistant"):
        intent_query = f"Based on the conversation history:\n{full_history}\nIs the last message a specific religious question? Answer 'search' or 'chat'."
        intent_result = llm.invoke(intent_query).content.strip().lower()

        context = ""
        docs = [] 
        
        if "search" in intent_result:
            search_query = f"{full_history}\nQuestion: {prompt}"
            with st.spinner("جاري مراجعة الفتاوى..."):
                docs = retriever.invoke(search_query)
                context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        else:
            context = "لا يوجد سياق فقهي محدد لهذه الرسالة."

        prompt_template = PromptTemplate.from_template("""أنت "إفتيلي"، خبير شرعي ودود. أجب بناءً على تاريخ المحادثة والفتاوى المتاحة فقط.
        تاريخ المحادثة:
        {chat_history}
        الفتاوى المتاحة:
        {context}
        السؤال الحالي: {question}
        إجابة إفتيلي:""")

        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "question": prompt,
            "chat_history": full_history
        })

        st.markdown(response)
        
        if "search" in intent_result and docs:
            with st.expander("📚 المصادر"):
                urls = set()
                for doc in docs:
                    u = doc.metadata.get("link", doc.metadata.get("source", "")).strip()
                    if u and u not in urls:
                        st.markdown(f"- [رابط الفتوى]({u})")
                        urls.add(u)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()