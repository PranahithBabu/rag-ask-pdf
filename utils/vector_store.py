from typing import List, Optional
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

from .config import PERSIST_DIR, EMBEDDING_MODEL, INDEX_FILE

class VectorStore:
    def __init__(self, collection_name: str = "pdf_collection", persist_dir: str = PERSIST_DIR,
                 embedding_model_name: str = EMBEDDING_MODEL):
        """
        VectorStore backed by Chroma (persistent). Embeddings are computed
        locally with sentence-transformers and passed to Chroma when adding docs.
        """
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)

        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)

        # Using a small CPU-friendly model to avoid meta-tensor / GPU issues
        self.model = SentenceTransformer(embedding_model_name, device="cpu")

        # simple local index to track filenames added to the collection
        self.index_path = os.path.join(self.persist_dir, INDEX_FILE)
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self.index = json.load(f)
            except Exception:
                self.index = {}
        else:
            self.index = {}

    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

    def list_pdfs(self) -> List[str]:
        """Return list of filenames already indexed (from local index)."""
        return list(self.index.keys())

    def has_pdf(self, filename: str) -> bool:
        return filename in self.index

    def add_texts(self, texts: List[str], file_name: str, ids_prefix: Optional[str] = None):
        """
        Add texts to collection. Each chunk gets an id and metadata {'source': file_name}.
        If file_name already exists in index, we will append new chunks with unique ids.
        """
        if not texts:
            return

        # Prepare IDs and metadatas
        if ids_prefix is None:
            ids_prefix = file_name.replace(" ", "_")

        start_idx = self.index.get(file_name, 0)
        ids = [f"{ids_prefix}_chunk_{start_idx + i}" for i in range(len(texts))]
        metadatas = [{"source": file_name} for _ in texts]

        # Compute embeddings (numpy array)
        emb = self.model.encode(texts, convert_to_numpy=True)
        # Chroma expects list of lists for embeddings
        emb_list = emb.tolist()

        # Add to collection (pass embeddings explicitly)
        self.collection.add(documents=texts, ids=ids, metadatas=metadatas, embeddings=emb_list)

        # Update local index: store new total count
        self.index[file_name] = start_idx + len(texts)
        self._save_index()

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """
        Search and return the top-k document texts (strings) relevant to query.
        We compute the query embedding with the same model and use collection.query.
        """
        if not query or self.collection.count() == 0:
            return []

        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_list = q_emb.tolist()

        # use query with embeddings directly
        res = self.collection.query(query_embeddings=q_list, n_results=k, include=["documents", "metadatas", "distances"])
        if not res or "documents" not in res or not res["documents"]:
            return []
        docs = res["documents"][0]
        return docs
