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
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # List all collections to debug
        collections = client.list_collections()
        st.sidebar.info(f"Found collections: {[col.name for col in collections]}")
        
        embedding_func = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        collection = client.get_collection(
            name="company_docs",
            embedding_function=embedding_func
        )
        
        return collection
        
    except Exception as e:
        st.error(f"ChromaDB Loading Error: {str(e)}")
        st.info("Trying to list available collections...")
        
        try:
            client = chromadb.PersistentClient(path="./chroma_db")
            collections = client.list_collections()
            st.error(f"Available collections: {[col.name for col in collections]}")
        except:
            pass
            
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
    st.markdown("Ask me anything about company policies!")

    # Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This assistant answers questions based on your company documents:
        - HR Policies
        - Benefits
        - Vacation & Leave
        - Remote Work Guidelines
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
        welcome = """
        Hi! 👋 I'm your **Company Knowledge Assistant**.

        I can help you with information about:
        - Vacation and time off policies
        - Remote work guidelines  
        - Parental leave benefits
        - Other company policies

        Just type your question below!
        """
        with st.chat_message("assistant"):
            st.write(welcome)

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # RAG Function
    def get_rag_response(question, n_results=4):
        try:
            results = collection.query(
                query_texts=[question],
                n_results=n_results
            )
            contexts = results["documents"][0] if results.get("documents") else []
            
            if not contexts:
                return "Sorry, I couldn't find any relevant information in the company documents."

            context_text = "\n\n".join(contexts)

            system_prompt = """
You are a helpful and accurate company knowledge assistant.
Answer ONLY using the information provided in the context.
If the answer is not in the context, say "I don't have information about that."
Do not make up any answers.
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
            return f"Error processing your question: {str(e)}"

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
    st.info("""
    **Troubleshooting:**
    - Make sure `chroma_db` folder exists with all files
    - Check that `.streamlit/secrets.toml` has `OPENROUTER_API_KEY`
    - Python version should be 3.11
    """)
