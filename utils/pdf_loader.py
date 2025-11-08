from typing import List
from PyPDF2 import PdfReader
import re

def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into smaller chunks (by words) with overlap.
    Default chunk_size is in words (≈ 300 words per chunk).
    """
    words = re.split(r"\s+", text)
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def load_pdf(file_path: str) -> List[str]:
    """
    Extract text from PDF and split into chunks.
    Returns: list of text chunks (strings).
    """
    pdf_reader = PdfReader(file_path)
    text_chunks = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if not text:
            continue
        page_chunks = split_text(text)
        text_chunks.extend(page_chunks)
    return text_chunks
