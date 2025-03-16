import os
import tempfile  # ✅ Fix for ChromaDB storage issues
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
import streamlit as st
from uuid import uuid4
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Securely Fetch API Key from Streamlit Secrets
api_key = st.secrets["GROQ_MODEL"]

# ✅ Use In-Memory ChromaDB OR Temporary Persistent Storage
chroma_client = chromadb.PersistentClient(path=tempfile.mkdtemp())  # ✅ Temporary Fix for Streamlit Cloud

# Constants
CHUNK_SIZE = 512
COLLECTION_NAME = "real_estate"

# ✅ Initialize Components
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7, max_tokens=500)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

def process_urls(urls):
    """Scrapes data from URLs and stores it in ChromaDB."""
    yield "Initializing Components..."
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    yield f"Extracted {len(data)} documents from URLs."

    if not data or all(not doc.page_content.strip() for doc in data):
        yield "❌ ERROR: No content extracted!"
        return

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=CHUNK_SIZE)
    docs = text_splitter.split_documents(data)
    yield f"Total documents after splitting: {len(docs)}"

    if len(docs) == 0:
        yield "❌ ERROR: No documents were split properly."
        return

    texts = [doc.page_content for doc in docs]
    ids = [str(uuid4()) for _ in range(len(docs))]
    metadatas = [{"source": urls[i % len(urls)]} for i in range(len(docs))]

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    yield f"✅ ChromaDB now contains {collection.count()} documents."

def generate_answer(query):
    """Retrieves relevant documents and generates an answer using the LLM."""
    retrieved_docs = collection.query(query_texts=[query], n_results=5)

    if not retrieved_docs["documents"]:
        return "I couldn't find relevant information.", []

    docs_text = retrieved_docs["documents"][0]
    sources = [meta["source"] for meta in retrieved_docs["metadatas"][0]]

    context = "\n\n".join(docs_text)
    prompt = f"Use the following information to answer the query:\n\n{context}\n\nQuestion: {query}"

    result = llm.invoke(prompt)
    answer = result["content"] if isinstance(result, dict) and "content" in result else result.content

    return answer, set(sources)
