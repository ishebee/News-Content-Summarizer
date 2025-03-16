import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 512
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

# Ensure Streamlit Cloud compatibility
PERSISTENT_STORAGE = os.getenv("STREAMLIT_CLOUD") != "true"  # Disable persistent storage on Streamlit Cloud

# Disable Hugging Face tokenizer parallelism warning
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
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=500)

    if chroma_client is None:
        if PERSISTENT_STORAGE:
            chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        else:
            chroma_client = chromadb.Client()  # In-memory mode for Streamlit Cloud

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

    # Clear old data
    collection.delete(where={"source": {"$ne": ""}})


def process_urls(urls):
    """Scrapes data from URLs and stores it in ChromaDB."""
    initialize_components()
    yield "Initializing Components..."

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    yield f"Extracted {len(data)} documents from URLs."

    if not data or all(not doc.page_content.strip() for doc in data):
        yield "ERROR: No content was extracted from the URLs!"
        return

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=CHUNK_SIZE)
    docs = text_splitter.split_documents(data)
    yield f"Total documents after splitting: {len(docs)}"

    if len(docs) == 0:
        yield "ERROR: No documents were split properly."
        return

    texts = [doc.page_content for doc in docs]
    ids = [str(uuid4()) for _ in range(len(docs))]
    metadatas = [{"source": urls[i % len(urls)]} for i in range(len(docs))]

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    yield f"ChromaDB now contains {collection.count()} documents."


def generate_answer(query):
    """Retrieves relevant documents and generates an answer using the LLM."""
    global collection

    if collection is None:
        initialize_components()

    retrieved_docs = collection.query(query_texts=[query], n_results=5)

    if not retrieved_docs["documents"]:
        return "I couldn't find relevant information.", []

    docs_text = retrieved_docs["documents"][0]
    sources = [meta["source"] for meta in retrieved_docs["metadatas"][0]]

    context = "\n\n".join(docs_text)
    prompt = f"Use the following information to answer the query:\n\n{context}\n\nQuestion: {query}"

    result = llm.invoke(prompt)

    if isinstance(result, dict) and "content" in result:
        answer = result["content"]
    elif hasattr(result, "content"):
        answer = result.content
    else:
        answer = str(result)

    return answer, set(sources)
