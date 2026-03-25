# backend\mcp_server\vectorstore.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- PATH SETUP ----------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PDF_PATH = os.path.join(PROJECT_ROOT, "NovaCart_Ecommerce.pdf")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "rag_data_store")

# ---------------- SINGLETON OBJECTS ----------------

_embeddings = None
_vectorstore = None
_retriever = None


# ---------------- EMBEDDINGS ----------------

def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embeddings


# ---------------- VECTORSTORE ----------------

def get_vectorstore():
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    os.makedirs(CHROMA_DIR, exist_ok=True)

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name="company_faq",
    )

    # Check if collection is empty
    existing_count = vectorstore._collection.count()

    if existing_count == 0:
        print("📄 Building vector store from PDF (one-time)...")

        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        docs = splitter.split_documents(documents)

        vectorstore.add_documents(docs)

        print(f"✅ Vector store built with {len(docs)} chunks")

    else:
        print(f"✅ Reusing existing vector store ({existing_count} chunks)")

    _vectorstore = vectorstore
    return vectorstore


# ---------------- RETRIEVER ----------------

def get_retriever():
    global _retriever

    if _retriever is not None:
        return _retriever

    vectorstore = get_vectorstore()

    _retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return _retriever