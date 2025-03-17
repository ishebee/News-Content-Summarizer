import streamlit as st
st.set_option("server.runOnSave", False)
from rag import process_urls, generate_answer
import os
port = int(os.environ.get("PORT", 10000))


st.set_page_config(page_title="News Content Summarizer", layout="wide")

st.title("📢 News Content Summarizer")

st.sidebar.header("🔗 Enter URLs")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_url_button = st.sidebar.button("Process URLs")
placeholder = st.empty()

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url.strip()]
    if not urls:
        placeholder.error("⚠️ Please enter at least one valid URL.")
    else:
        status_bar = st.sidebar.progress(0)
        for i, status in enumerate(process_urls(urls)):
            placeholder.text(status)
            status_bar.progress((i + 1) * 20)  # Simulated progress
        status_bar.empty()

st.subheader("🔍 Ask a Question")
query = st.text_input("Enter your query:", "")

if query:
    try:
        answer, sources = generate_answer(query)
        st.success("✅ Answer Generated")

        st.markdown("### 💡 Answer:")
        st.write(answer)

        if sources:
            st.markdown("### 🔗 Sources:")
            for source in sources:
                st.markdown(f"- [{source}]({source})")

    except RuntimeError:
        placeholder.error("⚠️ You must process URLs first!")
