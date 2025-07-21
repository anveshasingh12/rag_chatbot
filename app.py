import streamlit as st
from rag_pipeline import load_qa_chain
import os
import shutil
from ingest import load_and_split_pdfs, build_faiss_index

st.set_page_config(page_title="📚 Smart RAG Chatbot", layout="wide")
st.title("📄 Multi-PDF Chatbot with Source & Summary")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    os.makedirs("uploaded_files", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("uploaded_files", file.name), "wb") as f:
            f.write(file.read())

    st.sidebar.success("PDFs uploaded. Now processing...")
    docs = load_and_split_pdfs("uploaded_files")
    build_faiss_index(docs)
    st.sidebar.success("Index built successfully!")

    st.session_state.qa = load_qa_chain()
else:
    st.info("Please upload PDFs to begin.")

query = st.text_input("💬 Ask something from the documents:")

if query and "qa" in st.session_state:
    with st.spinner("Thinking..."):
        result = st.session_state.qa(query)
        answer = result["result"]
        sources = result["source_documents"]

    st.markdown("### ✅ Answer:")
    st.write(answer)

    st.markdown("### 📚 Source Snippets:")
    for i, doc in enumerate(sources):
        st.write(f"**Doc {i+1}:** {doc.page_content[:300]}...")

    st.session_state.chat_history.append((query, answer))

if st.session_state.chat_history:
    with st.expander("🕒 Chat History"):
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
