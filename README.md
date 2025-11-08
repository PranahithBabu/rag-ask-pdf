# 🧠 RAG-Based PDF Chat Application

This is a Retrieval-Augmented Generation (RAG) application built with Streamlit, and ChromaDB that allows users to upload PDFs and chat with them using the Groq LLM API.
It extracts text from uploaded documents, stores embeddings in a vector database, and retrieves the most relevant chunks during question-answering.

## 🚀 Features

- 📄 Upload one or more PDF documents.

- ⚡ Extract and embed text using Sentence Transformers and ChromaDB.

- 💬 Ask questions in natural language about your PDFs.

- 🧹 Automatically handles text chunking and vector storage.

- 🔑 Secure API key input for Groq LLM.

- 🌐 Deployed on Streamlit Cloud (free and public access).

## 🔭 Usage

- Avoid reading long FAQ's.
- Get summaries of long books.
- Understand terms and conditions of businesses with simple questions.

## 🧩 Tech Stack
Layer | Technology
--- | ---
Frontend	| Streamlit
Backend | Python
Vector DB |	ChromaDB
Embeddings |	SentenceTransformers
LLM |	Groq API
File Parsing |	PyPDF2

## 🖥️ Live Demo

You can try the deployed version here:
👉 Streamlit App [Link](https://rag-ask-pdf.streamlit.app/)

## 🛠️ Local Setup Instructions

Follow these steps to run the app on your local machine:

1. Clone the repository

    ```
    git clone https://github.com/PranahithBabu/rag-ask-pdf.git
    cd rag-ask-pdf
    ```

2. Create and activate a virtual environment

    ```
    python -m venv venv
    venv\Scripts\activate
    ```

3. Install dependencies

    ```
    pip install -r requirements.txt
    ```

    <i>Remove content inside <b>chromadb</b> folder to avoid default PDF's.  </i>

4. You can get your Groq API key from https://console.groq.com/keys

5. Run the application ```streamlit run app.py```

6. Then open the displayed local URL (e.g., http://localhost:8501) in your browser.

## 🧭 Usage Guide

- Launch the app.

- Enter your Groq API key in the sidebar input box.

- Upload one or more PDF files.

- Wait for processing and the sidebar will list uploaded files.

- Type your question in the chat box.

- The app retrieves context from the vector DB and responds intelligently.

## 🧹 Troubleshooting

```
Error: Missing module → run pip install -r requirements.txt again.

Chromadb lock issue → delete .chromadb/ folder and re-run the app.

PDF not loading → ensure the file is not encrypted or scanned image-only (OCR not supported yet).
```
