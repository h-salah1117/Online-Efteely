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
st.caption(" (البوت معمول لغرض تعليمي فقط لا تعتمد عليه في أمرك الدينية❌)اسأل أي سؤال فقهي")

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

    vectorstore = Chroma(
        persist_directory=CHROMA_SUBDIR,
        embedding_function=embeddings
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 5})

try:
    with st.spinner("جاري تشغيل محرك البحث..."):
        retriever = load_rag()
except Exception as e:
    st.error(f"❌ خطأ في تحميل قاعدة البيانات: {e}")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.25,
    groq_api_key=st.secrets["GROQ_API_KEY"]
)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.toast("✅ تم تحميل قاعدة البيانات وجاهز للرد!", icon="📚")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history_text = ""
    for msg in st.session_state.messages[-6:-1]: 
        role = "المستخدم" if msg["role"] == "user" else "إفتيلي"
        chat_history_text += f"{role}: {msg['content']}\n"

    with st.chat_message("assistant"):
        check_prompt = f"Is the following user message a specific Islamic jurisprudence question that needs a fatwa search or just a greeting/general talk? Respond with 'search' or 'chat'. Message: {prompt}"
        intent = llm.invoke(check_prompt).content.strip().lower()

        context = ""
        docs = [] 

        if "search" in intent:
            with st.spinner("جاري البحث في قاعدة الفتاوى..."):
                docs = retriever.invoke(prompt)
                context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        else:
            context = "لا يوجد سياق فقهي محدد لهذه الرسالة."

        prompt_template = PromptTemplate.from_template("""أنت "إفتيلي"، خبير شرعي ومساعد ذكي بأسلوب ودود.
        1. إذا كانت رسالة المستخدم تحية أو كلام عام، رد بلباقة وادعه لسؤالك.
        2. ابدأ بالإجابة مباشرة دون مقدمات مكررة.
        3. استخدم السياق فقط للأسئلة الفقهية الصريحة.
        4. تعامل كإنسان عادي ولا تذكر أنك بوت.

        سياق المحادثة السابقة:
        {chat_history}

        الفتاوى المستخرجة:
        {context}

        سؤال المستخدم: {question}

        إجابة إفتيلي:""")

        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({
            "context": context, 
            "question": prompt, 
            "chat_history": chat_history_text
        })
        
        st.markdown(response)

        if docs:
            with st.expander("📚 المصادر"):
                seen = set()
                for i, doc in enumerate(docs):
                    url = doc.metadata.get("link", doc.metadata.get("source", "")).strip()
                    if url and url not in seen:
                        st.markdown(f"**{i+1}.** [🔗 رابط الفتوى]({url})")
                        seen.add(url)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()