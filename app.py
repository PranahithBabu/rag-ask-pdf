import streamlit as st
import os
import tempfile
from utils.pdf_loader import load_pdf
from utils.llm_chain import LLMChain
from utils.vector_store import VectorStore
from utils.config import PERSIST_DIR

st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("📄 PDF Question & Answer Bot")

# Session state initializations
if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = None

if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []

# Initialize vector store (so we can list already-indexed PDFs on startup)
vector_store = VectorStore(persist_dir=PERSIST_DIR)
persisted_filenames = vector_store.list_pdfs()

# initialize session_state list from persisted index (avoid duplicates)
for fn in persisted_filenames:
    if fn not in st.session_state.uploaded_pdfs:
        st.session_state.uploaded_pdfs.append(fn)

# Sidebar: Groq API key + list of persisted PDFs
with st.sidebar:
    api_key = st.text_input("Enter your Groq API key", type="password")

    st.markdown("---")
    st.markdown("### Indexed PDFs")
    if st.session_state.uploaded_pdfs:
        for name in st.session_state.uploaded_pdfs:
            st.write(f"- {name}")
    else:
        st.write("_No PDFs indexed yet_")

# Initialize LLMChain only after API key provided
if api_key:
    if st.session_state.llm_chain is None:
        try:
            st.session_state.llm_chain = LLMChain(api_key=api_key, persist_dir=PERSIST_DIR)
        except Exception as e:
            st.error(f"Failed to initialize LLM: {e}")

# File upload area in main page
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar before uploading.")
    else:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            texts = load_pdf(tmp_file_path)
            if not texts:
                st.warning("PDF contained no extractable text.")
            else:
                # Add to vector DB via LLMChain (which uses VectorStore)
                st.session_state.llm_chain.create_knowledge_base(texts, file_name=uploaded_file.name)
                # update sidebar list (session state)
                if uploaded_file.name not in st.session_state.uploaded_pdfs:
                    st.session_state.uploaded_pdfs.append(uploaded_file.name)
                st.success(f"✅ Indexed: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
        finally:
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

# Show question input only if there is at least one indexed document
if st.session_state.uploaded_pdfs:
    st.markdown("---")
    st.header("Ask a question (across all indexed PDFs)")
    question = st.text_input("Enter your question about the indexed PDFs:")
    if st.button("Ask") and question:
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar first.")
        else:
            try:
                answer = st.session_state.llm_chain.ask_question(question)
                st.write("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error while answering: {e}")
else:
    st.info("Upload at least one PDF to enable question answering.")
