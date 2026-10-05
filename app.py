import streamlit as st
import os
import tempfile
from model_pipeline import process_pdf_and_create_vectorstore, query_rag_system

st.set_page_config(
    page_title="Enterprise Document RAG System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Enterprise Document RAG Assistant")
st.markdown("### AI-Powered Policy & Document Query System")

# Sidebar - Document Ingestion
st.sidebar.header("Document Management")
uploaded_file = st.sidebar.file_uploader("Upload Policy PDF", type=["pdf"])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file is not None:
    with st.sidebar.spinner("Processing & Indexing Document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        vectorstore, num_pages, num_chunks = process_pdf_and_create_vectorstore(tmp_path)
        st.session_state.vectorstore = vectorstore
        os.remove(tmp_path)
        
        st.sidebar.success(f"Indexed {num_pages} Pages ({num_chunks} Text Chunks)")

# Main Query Interface
query = st.text_input("Enter your query regarding internal enterprise policies:")

if st.button("Search & Answer") and query:
    if st.session_state.vectorstore is None:
        st.error("Please upload a PDF document in the sidebar first!")
    else:
        with st.spinner("Retrieving Relevant Contexts..."):
            answer, docs = query_rag_system(query, st.session_state.vectorstore)
            
            st.subheader("Generated Answer")
            st.info(answer)
            
            st.subheader("Retrieved Reference Chunks")
            for i, doc in enumerate(docs):
                with st.expander(f"Reference Chunk {i+1} (Source: Page {doc.metadata.get('page', 0) + 1})"):
                    st.write(doc.page_content)