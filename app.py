import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Page configuration
st.set_page_config(
    page_title="Efteely - Islamic Chatbot",
    page_icon="🕌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🕌 إفتيلي - Islamic Chatbot")
st.caption("اسأل أي سؤال فقهي وهجاوبك من فتاوى موثوقة")

# Load RAG components with caching (important for performance)
@st.cache_resource
def load_rag():
    # Use the SAME embedding model that was used during indexing
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",   # Must match your backup
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load the chroma_db from your backup
    if not os.path.exists("./chroma_db"):
        st.error("chroma_db folder not found! Please make sure you extracted chroma_db_backup.zip")
        st.stop()
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = load_rag()

# Initialize Groq LLM (fast + good free tier)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.25,
    groq_api_key=st.secrets.get("GROQ_API_KEY")
)

# Improved prompt for Islamic content
prompt_template = PromptTemplate.from_template("""
You are a knowledgeable and trustworthy Islamic scholar.
Answer the user's question in clear Arabic (use Fusha or simple Egyptian dialect when suitable).
Use ONLY the provided fatwas as reference. Do not add external knowledge.
Be concise, respectful, and accurate.
If the provided context does not contain a direct answer, politely say so and suggest consulting a qualified scholar.

Context fatwas:
{context}

Question: {question}

Answer:
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("اكتب سؤالك هنا... مثلاً: حكم صيام يوم الجمعة؟"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في الفتاوى وتوليد الإجابة..."):
            
            # Retrieve relevant fatwas
            docs = retriever.invoke(prompt)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
            # Generate answer
            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke({"context": context, "question": prompt})
            
            st.markdown(response)
            
            # Show sources in expandable section
            with st.expander("📚 المصادر المستخدمة"):
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source", "Unknown Source")
                    link = doc.metadata.get("link", "")
                    st.markdown(f"**{i}.** {source}")
                    if link:
                        st.markdown(f"   🔗 [رابط الفتوى]({link})")

    st.session_state.messages.append({"role": "assistant", "content": response})