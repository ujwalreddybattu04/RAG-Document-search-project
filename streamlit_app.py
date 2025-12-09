"""Streamlit UI for Agentic RAG System - Gemini ReAct Version"""
import uuid
import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------
# SESSION STATE INIT
# -----------------------------
def init_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []


# -----------------------------
# INITIALIZE RAG SYSTEM (CACHED)
# -----------------------------
@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Load Gemini LLM
        llm = Config.get_llm()

        # Initialize components
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()

        # Process default URLs
        urls = Config.DEFAULT_URLS
        documents = doc_processor.process_urls(urls)

        # Create vector store
        vector_store.create_vectorstore(documents)

        # Build LangGraph workflow
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()

        return graph_builder, len(documents)

    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")
        return None, 0


# -----------------------------
# MAIN APPLICATION
# -----------------------------
def main():
    init_session_state()

    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents below 👇")

    # First-time initialization
    if not st.session_state.initialized:
        with st.spinner("Loading RAG system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")

    st.markdown("---")

    # -----------------------------
    # QUESTION INPUT
    # -----------------------------
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., Explain the agent workflow"
        )
        submit = st.form_submit_button("🔍 Search")

    # -----------------------------
    # PROCESS QUESTION
    # -----------------------------
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Generating answer with OpenAI..."):
                start_time = time.time()

                result = st.session_state.rag_system.run(question)

                elapsed_time = time.time() - start_time

                # Save query history
                st.session_state.history.append({
                    "question": question,
                    "answer": result["answer"],
                    "time": elapsed_time
                })

                # -----------------------------
                # DISPLAY ANSWER
                # -----------------------------
                st.markdown("### 🧠 Answer")
                st.markdown(f"**{result['answer']}**")

                # -----------------------------
                # SHOW RETRIEVED DOCS
                # -----------------------------
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result["retrieved_docs"], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:400] + "...",
                            height=120,
                            disabled=True
                        )

                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    # -----------------------------
    # SHOW HISTORY
    # -----------------------------
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")

        for item in reversed(st.session_state.history[-3:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer'][:200]}...")
            st.caption(f"Time: {item['time']:.2f}s")
            st.markdown("")


if __name__ == "__main__":
    main()
