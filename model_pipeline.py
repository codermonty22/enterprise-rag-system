import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    """Initializes and returns the HuggingFace Embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def process_pdf_and_create_vectorstore(pdf_path: str):
    """Loads PDF, splits text into overlapping chunks, and stores embeddings in ChromaDB."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Text chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Store in Chroma Vector DB
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    return vectorstore, len(documents), len(chunks)

def query_rag_system(query: str, vectorstore, top_k: int = 3):
    """Retrieves top-k relevant context chunks for a given query and cleans whitespace."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)
    
    # 1. Join page contents
    raw_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 2. FIX: Replace single newlines and multiple spaces with a single space
    cleaned_context = re.sub(r'\s+', ' ', raw_context).strip()
    
    # 3. Formulate response
    response = (
        f"Based on internal policy documents, here is the relevant guidance:\n\n"
        f"{cleaned_context[:500]}..."
    )
    return response, retrieved_docs