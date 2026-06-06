import sys
# ==================== SQLITE FIX FOR STREAMLIT CLOUD ====================
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# =====================================================================

import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Company Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# LOAD ENV VARIABLES
# =========================================================
load_dotenv()

# =========================================================
# CHROMADB INITIALIZATION
# =========================================================
@st.cache_resource
def init_chromadb():
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(
        name="company_docs",
        embedding_function=embedding_func
    )
    return collection

# =========================================================
# OPENROUTER LLM INITIALIZATION
# =========================================================
@st.cache_resource
def init_llm():
    return ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

# =========================================================
# INITIALIZE COMPONENTS
# =========================================================
try:
    collection = init_chromadb()
    llm = init_llm()

    print(f"✅ ChromaDB collection 'company_docs' has {collection.count()} documents")

    # =========================================================
    # TITLE
    # =========================================================
    st.title("🤖 Company Knowledge Assistant")
    st.markdown("Ask me anything about company policies!")

    # =========================================================
    # SESSION STATE
    # =========================================================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # =========================================================
    # SIDEBAR
    # =========================================================
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This AI assistant can answer questions about:
        - Vacation policies
        - Remote work guidelines
        - Parental leave
        - Benefits information

        **Powered by:**
        - OpenRouter (GPT-3.5)
        - ChromaDB vector search
        - Semantic RAG
        """)
        st.divider()
        st.metric("Documents Indexed", collection.count())
        st.metric("Messages in Chat", len(st.session_state.messages))
        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # =========================================================
    # WELCOME MESSAGE
    # =========================================================
    if len(st.session_state.messages) == 0:
        welcome = """
        Hi! I'm your company knowledge assistant.

        I can help you find information about:
        - Vacation and time off policies
        - Remote work guidelines
        - Parental leave benefits
        - And more!

        Just ask me a question to get started.
        """
        with st.chat_message("assistant"):
            st.write(welcome)

    # =========================================================
    # DISPLAY CHAT HISTORY
    # =========================================================
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # =========================================================
    # RAG FUNCTION
    # =========================================================
    def get_rag_response(question, n_results=3):
        try:
            results = collection.query(
                query_texts=[question],
                n_results=n_results
            )
            contexts = results["documents"][0] if results.get("documents") else []
            if not contexts:
                return "Sorry, I couldn't find relevant information in the company documents."

            context_text = "\n\n".join(contexts)

            system_prompt = """
You are a helpful company knowledge assistant.
Answer ONLY using the provided context.
If the answer is not present in the context, say that you don't know.
Do not make up any information.
"""

            user_prompt = f"""
Context:
{context_text}

Question:
{question}

Answer based only on the context above.
"""

            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

    # =========================================================
    # CHAT INPUT
    # =========================================================
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = get_rag_response(prompt)
                st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("""
Make sure:
- You have created the **chroma_db** folder with all .bin and .sqlite3 files inside
- OPENROUTER_API_KEY is added in .streamlit/secrets.toml
- You are using Python 3.10 on Streamlit Cloud
""")
    st.stop()
