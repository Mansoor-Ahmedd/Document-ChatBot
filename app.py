import sys
# ==================== CRITICAL FIX FOR STREAMLIT CLOUD ====================
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
# CHROMADB INITIALIZATION - FIXED FOR CLOUD
# =========================================================
@st.cache_resource(show_spinner="Loading knowledge base...")
def init_chromadb():
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        
        embedding_func = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Use get_or_create_collection (more reliable on cloud)
        collection = client.get_or_create_collection(
            name="company_docs",
            embedding_function=embedding_func
        )
        
        count = collection.count()
        st.sidebar.success(f"✅ Loaded {count} document chunks")
        return collection
        
    except Exception as e:
        st.error(f"ChromaDB Error: {str(e)}")
        st.stop()

# =========================================================
# LLM INITIALIZATION
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
# MAIN APP
# =========================================================
try:
    collection = init_chromadb()
    llm = init_llm()

    st.title("🤖 Company Knowledge Assistant")
    st.markdown("**Ask me anything about company policies!**")

    # Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This assistant uses your company documents to answer questions about:
        - Vacation & Leave Policies
        - Remote Work Guidelines  
        - Benefits & Parental Leave
        - HR Policies
        """)
        st.divider()
        st.metric("Documents Indexed", collection.count())
        st.metric("Chat Messages", len(st.session_state.messages))
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Welcome Message
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.write("""
            Hi! 👋 I'm your Company Knowledge Assistant. 
            
            Ask me anything about company policies and I'll answer using the uploaded documents.
            """)

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # RAG Function
    def get_rag_response(question, n_results=4):
        try:
            results = collection.query(query_texts=[question], n_results=n_results)
            contexts = results["documents"][0] if results.get("documents") else []
            
            if not contexts:
                return "Sorry, I couldn't find relevant information in the company documents."

            context_text = "\n\n".join(contexts)

            system_prompt = """
You are a helpful, accurate company knowledge assistant.
Answer ONLY using the provided context. 
If the information is not in the context, say "I don't have that information."
Do not make up answers.
"""

            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:")
            ])
            return response.content

        except Exception as e:
            return f"Error: {str(e)}"

    # Chat Input
    if prompt := st.chat_input("Ask a question about company policies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching company documents..."):
                response = get_rag_response(prompt)
                st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"Application Error: {str(e)}")
    st.info("Check that `runtime.txt` has `python-3.11.9` and ChromaDB folder is correct.")
