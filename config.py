"""Environment-backed settings for the RAG service."""

import os
from pathlib import Path


CHROMA_PATH = os.getenv("CHROMA_PATH", "/home/tsuki/Desktop/ChromaDB/Data")
CHROMA_DOCS_COLLECTION = os.getenv("CHROMA_DOCS_COLLECTION", "docs")
CHROMA_QUESTIONS_COLLECTION = os.getenv("CHROMA_QUESTIONS_COLLECTION", "Questions")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
DENSE_QUERY_MODE = os.getenv("DENSE_QUERY_MODE", "collection")
GENERATOR_MODEL_DEFAULT = os.getenv("GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct")
RRF_ALPHA = float(os.getenv("RRF_ALPHA", "60"))
RRF_BM25_WEIGHT = float(os.getenv("RRF_BM25_WEIGHT", "1.2"))
RRF_DENSE_WEIGHT = float(os.getenv("RRF_DENSE_WEIGHT", "0.8"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "10"))
BM25_CACHE_PATH = Path(os.getenv("BM25_CACHE_PATH", "data/bm25_store/bm25.pkl"))
GENERATOR_LOAD_ON_STARTUP = os.getenv("GENERATOR_LOAD_ON_STARTUP", "false").lower() == "true"
