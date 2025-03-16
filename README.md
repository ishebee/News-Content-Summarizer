# **News Content Summarizer**

## **📌 Overview**
This project is a **News Content Summarizer** built using **Streamlit, ChromaDB, and LangChain**. It scrapes news articles from URLs, stores them in a **vector database**, and allows users to query the stored content for meaningful answers.

## **🚀 Features**
- **Scrape & Store** news articles from user-input URLs.
- **Process and Chunk Text** into small segments for efficient retrieval.
- **Query News Data** using an LLM (Llama-3.3-70B via Groq API).
- **Retrieve Sources** and display actual URLs in the response.
- **Uses ChromaDB** for efficient vector-based storage and retrieval.

## **🛠️ Installation & Setup**
### **🔹 Prerequisites**
Ensure you have Python **3.9+** installed. Then install the required dependencies:
```bash
pip install streamlit langchain langchain-community langchain-groq chromadb transformers dotenv
```

### **🔹 Clone Repository**
```bash
git clone https://github.com/your-repository/news-content-summarizer.git
cd news-content-summarizer
```

### **🔹 Configure Environment Variables**
Create a **.env** file in the project directory and add your Groq API key:
```plaintext
GROQ_API_KEY=your_groq_api_key
```

## **🔄 Running the Application**
### **Step 1: Start the Streamlit App**
```bash
streamlit run main.py
```

### **Step 2: Using the App**
1. **Enter URLs** in the sidebar to fetch news articles.
2. **Click "Process URLs"** to scrape and store content.
3. **Ask a question** related to the news articles.
4. **Get summarized answers** along with source URLs.

## **📂 Project Structure**
```plaintext
📁 news-content-summarizer
│── 📄 main.py           # Streamlit frontend UI
│── 📄 rag.py            # Backend logic for retrieval & processing
│── 📄 .env              # Environment variables (API keys)
│── 📄 README.md         # Project documentation
│── 📂 resources/vectorstore  # ChromaDB persistent storage
```

## **💡 How It Works**
1. **Text Extraction**: Uses `UnstructuredURLLoader` to extract content from URLs.
2. **Chunking & Embedding**:
   - Breaks large texts into smaller segments using `RecursiveCharacterTextSplitter`.
   - Converts text into embeddings using `sentence-transformers`.
3. **Vector Storage**:
   - Stores chunked text and metadata (source URL) in **ChromaDB**.
4. **Retrieval & Answering**:
   - Queries ChromaDB for relevant content.
   - Sends the best-matching content to Llama-3.3-70B for a summary.
5. **Display Results**:
   - Shows the summarized answer.
   - Provides source links to verify information.

## **🔍 Example Usage**
```plaintext
📝 Question: What percentage did India grow in 2019-2020?

💡 Answer: India's GDP growth rate for 2019-2020 was 4.2% according to World Bank data.

🔗 Sources:
1. [https://indianexpress.com/article/explained/explained-economics/indians-over-taxed-union-budget-9807329/](https://indianexpress.com/article/explained/explained-economics/indians-over-taxed-union-budget-9807329/)
2. [https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html](https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html)
```

## **🛠️ Technologies Used**
- **Python** (3.9+)
- **Streamlit** - Web UI
- **ChromaDB** - Vector Storage
- **LangChain** - Document Processing & Retrieval
- **LLM (Llama-3.3-70B via Groq API)** - Answer Generation
- **Hugging Face Transformers** - Tokenization & Embeddings

## **⚡ Future Enhancements**
- Support for multiple LLM providers (OpenAI, DeepSeek, Mistral, etc.)
- Improved query handling with RAG optimization
- Expand source retrieval to PDFs & other document types
- Integration with a speech-to-text model for voice queries


## **🙌 Contributing**
Contributions are welcome! Feel free to fork, improve, and create a pull request.

---
🎯 **Built for AI-driven News Summarization & Retrieval! 🚀**

