import streamlit as st
from uuid import uuid4
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# ✅ Fetch API key from Streamlit Secrets
api_key = st.secrets["GROQ_MODEL"]

# Constants
CHUNK_SIZE = 512
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

# Disable Hugging Face tokenizer parallelism warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize components
llm = None
chroma_client = None
collection = None
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def initialize_components():
    """Initializes the LLM and ChromaDB vector database."""
    global llm, chroma_client, collection

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7, max_tokens=500)

    if chroma_client is None:
        chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # ✅ Use sentence-transformers for embeddings
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

    # ✅ Remove Old Data Before Adding New URLs
    collection.delete(where={"source": {"$ne": ""}})  # Deletes all docs where 'source' is not empty

    print(" ChromaDB cleared. Ready for new data.")


def process_urls(urls):
    """Scrapes data from URLs and stores it in ChromaDB."""
    initialize_components()
    yield "Initializing Components..."

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    yield f"Extracted {len(data)} documents from URLs."

    if not data or all(not doc.page_content.strip() for doc in data):
        yield " ERROR: No content was extracted from the URLs!"
        return

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=CHUNK_SIZE)
    docs = text_splitter.split_documents(data)
    yield f"Total documents after splitting: {len(docs)}"

    if len(docs) == 0:
        yield " ERROR: No documents were split properly."
        return

    texts = [doc.page_content for doc in docs]
    ids = [str(uuid4()) for _ in range(len(docs))]

    # ✅ Store actual URLs in metadata instead of "Doc X"
    metadatas = [{"source": urls[i % len(urls)]} for i in range(len(docs))]  # Stores the correct URL

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    yield f" ChromaDB now contains {collection.count()} documents."


def generate_answer(query):
    """Retrieves relevant documents and generates an answer using the LLM."""
    global collection

    if collection is None:
        initialize_components()

    # Retrieve relevant documents
    retrieved_docs = collection.query(query_texts=[query], n_results=5)

    print("\n Retrieved Documents (Before LLM Query):")
    if not retrieved_docs["documents"]:
        print(" No relevant documents were retrieved!")
        return "I couldn't find relevant information.", []

    docs_text = retrieved_docs["documents"][0]  # Extract retrieved text
    sources = [meta["source"] for meta in retrieved_docs["metadatas"][0]]  # Extract exact URLs

    # Format the context for LLM
    context = "\n\n".join(docs_text)
    prompt = f"Use the following information to answer the query:\n\n{context}\n\nQuestion: {query}"

    # Call LLM
    result = llm.invoke(prompt)

    # ✅ Fix: Extract only the answer text
    if isinstance(result, dict) and "content" in result:
        answer = result["content"]
    elif hasattr(result, "content"):
        answer = result.content
    else:
        answer = str(result)

    return answer, set(sources)
