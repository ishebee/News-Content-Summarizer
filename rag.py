import asyncio
import sys

# Ensure there's an active event loop in Streamlit
if sys.platform != "win32":
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

import os
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # Force using updated SQLite
except ImportError:
    print("pysqlite3-binary is missing. Install it using `pip install pysqlite3-binary`.")

from uuid import uuid4
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import streamlit as st
from chromadb.config import Settings


# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 512
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "real_estate"

# Disable Hugging Face tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load API Key for Groq (Works for both local & Streamlit Cloud)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing! Set it in `.env` (local) or `st.secrets` (Streamlit Cloud).")

# Initialize components
llm = None
chroma_client = None
collection = None
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def initialize_components():
    """Initializes the LLM and ChromaDB with an HTTP-based client."""
    global llm, chroma_client, collection

    # Initialize LLM if not already done
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=500, api_key=GROQ_API_KEY)

    # Use ChromaDB HTTP Client for Streamlit Cloud
    chroma_client = chromadb.HttpClient(host="chroma-server", port=8000)

    # Load embedding function
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Ensure the collection is created before querying
    existing_collections = chroma_client.list_collections()
    if COLLECTION_NAME not in [col.name for col in existing_collections]:
        collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
    else:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)



def process_urls(urls):
    """Scrapes data from URLs and stores it in ChromaDB."""
    initialize_components()
    yield "Initializing Components..."

    # Load web page data
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    yield f"Extracted {len(data)} documents from URLs."

    if not data or all(not doc.page_content.strip() for doc in data):
        yield "⚠️ ERROR: No content was extracted from the URLs!"
        return

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=CHUNK_SIZE)
    docs = text_splitter.split_documents(data)
    yield f"Total documents after splitting: {len(docs)}"

    if len(docs) == 0:
        yield "⚠️ ERROR: No documents were split properly."
        return

    # Prepare documents for embedding
    texts = [doc.page_content for doc in docs]
    ids = [str(uuid4()) for _ in range(len(docs))]
    metadatas = [{"source": urls[i % len(urls)]} for i in range(len(docs))]

    # Store embeddings in ChromaDB only if data exists
    if len(texts) > 0:
        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        yield f"✅ ChromaDB now contains {collection.count()} documents."
    else:
        yield "⚠️ No documents to add to ChromaDB."


def generate_answer(query):
    """Retrieves relevant documents and generates an answer using the LLM."""
    global collection

    if collection is None:
        initialize_components()

    # Check if collection has data before querying
    if collection.count() == 0:
        return "⚠️ No data found in ChromaDB. Please process URLs first.", []

    retrieved_docs = collection.query(query_texts=[query], n_results=5)

    if not retrieved_docs["documents"]:
        return "I couldn't find relevant information.", []

    docs_text = retrieved_docs["documents"][0]
    sources = [meta["source"] for meta in retrieved_docs["metadatas"][0]]

    # Format prompt for LLM
    context = "\n\n".join(docs_text)
    prompt = f"Use the following information to answer the query:\n\n{context}\n\nQuestion: {query}"

    # Generate response using LLM
    result = llm.invoke(prompt)

    if isinstance(result, dict) and "content" in result:
        answer = result["content"]
    elif hasattr(result, "content"):
        answer = result.content
    else:
        answer = str(result)

    return answer, set(sources)
