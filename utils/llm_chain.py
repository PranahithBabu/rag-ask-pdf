from typing import List
import os
import groq
from .vector_store import VectorStore
from .config import GROQ_MODEL, PERSIST_DIR, TOP_K

class LLMChain:
    def __init__(self, api_key: str, persist_dir: str = PERSIST_DIR, groq_model: str = GROQ_MODEL):
        """
        Initialize LLMChain with Groq client and persistent VectorStore.
        """
        os.environ["GROQ_API_KEY"] = api_key
        self.client = groq.Groq(api_key=api_key)
        self.model_name = groq_model
        self.vector_store = VectorStore(persist_dir=persist_dir)
        self.last_texts = []

    def create_knowledge_base(self, texts: List[str], file_name: str):
        """
        Index a PDF's text chunks into Chroma persistent collection.
        """
        if not texts:
            return
        self.last_texts = texts
        self.vector_store.add_texts(texts, file_name=file_name)

    def ask_question(self, question: str, top_k: int = TOP_K) -> str:
        """
        Retrieve relevant chunks across all indexed PDFs and query Groq LLM.
        """
        if self.vector_store.collection.count() == 0:
            return "No documents indexed. Please upload a PDF first."

        context_chunks = self.vector_store.similarity_search(question, k=top_k)
        if not context_chunks:
            return "I couldn't find any relevant information in the documents."

        context = "\n\n".join(context_chunks)
        prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.
        If the answer is not in the context, say "I couldn't find that in the document."

        Context:
        {context}

        Question: {question}
        """

        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0
        )

        # Groq returns message content
        return completion.choices[0].message.content
