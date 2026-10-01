"""Environment-backed settings for the RAG service."""

import os
from pathlib import Path


CHROMA_PATH = os.getenv("CHROMA_PATH", "/home/tsuki/Desktop/ChromaDB/Data")
CHROMA_DOCS_COLLECTION = os.getenv("CHROMA_DOCS_COLLECTION", "docs")
CHROMA_QUESTIONS_COLLECTION = os.getenv("CHROMA_QUESTIONS_COLLECTION", "Questions")
# Must match the model the docs collection was embedded with (Chroma's default).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DENSE_QUERY_MODE = os.getenv("DENSE_QUERY_MODE", "collection")

# "ollama" runs a quantized model on CPU; "transformers" needs a CUDA GPU for 4-bit loading.
GENERATOR_BACKEND = os.getenv("GENERATOR_BACKEND", "ollama")
GENERATOR_MODEL_DEFAULT = os.getenv("GENERATOR_MODEL", "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
GENERATOR_TIMEOUT_SECONDS = int(os.getenv("GENERATOR_TIMEOUT_SECONDS", "600"))
# ~1.1k prompt tokens: a CPU prefills this in about a minute; 8000 took several.
GENERATOR_MAX_CONTEXT_CHARS = int(os.getenv("GENERATOR_MAX_CONTEXT_CHARS", "4500"))
GENERATOR_MAX_NEW_TOKENS = int(os.getenv("GENERATOR_MAX_NEW_TOKENS", "256"))
GENERATOR_LOAD_ON_STARTUP = os.getenv("GENERATOR_LOAD_ON_STARTUP", "false").lower() == "true"

# Tuned on the benchmark's train split (scripts/benchmark.py): recall@10 0.699 vs 0.646 for BM25 alone.
RRF_ALPHA = float(os.getenv("RRF_ALPHA", "20"))
RRF_BM25_WEIGHT = float(os.getenv("RRF_BM25_WEIGHT", "1.0"))
RRF_DENSE_WEIGHT = float(os.getenv("RRF_DENSE_WEIGHT", "0.8"))
# Each retriever contributes this many candidates before fusion trims to top_k.
RRF_CANDIDATES = int(os.getenv("RRF_CANDIDATES", "50"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "10"))
BM25_CACHE_PATH = Path(os.getenv("BM25_CACHE_PATH", "data/bm25_store/bm25.joblib"))

# Suggested by scripts/train_calibrator.py on the train split. No mean cutoff let the
# router abstain without discarding questions whose evidence was still retrievable,
# so abstention is left to high model uncertainty and to the generator's prompt.
ROUTER_ABSTAIN_MEAN_THRESHOLD = float(os.getenv("ROUTER_ABSTAIN_MEAN_THRESHOLD", "0.0"))
ROUTER_ABSTAIN_STD_THRESHOLD = float(os.getenv("ROUTER_ABSTAIN_STD_THRESHOLD", "0.2"))
ROUTER_FAST_MEAN_THRESHOLD = float(os.getenv("ROUTER_FAST_MEAN_THRESHOLD", "0.89"))
CALIBRATOR_MODEL_PATH = Path(os.getenv("CALIBRATOR_MODEL_PATH", "models/calibrator.pkl"))

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Tuned on the train split: reranking the top 2*k hybrid candidates on their 2 best
# passages lifts recall@10 from 0.699 to 0.716. Any relevance cutoff lowered recall
# (0.01 -> 0.681), so filtering is off by default; raise it to trade recall for
# fewer off-topic documents in the prompt.
CRAG_RETRIEVAL_MULTIPLIER = int(os.getenv("CRAG_RETRIEVAL_MULTIPLIER", "2"))
CRAG_PASSAGES_PER_DOC = int(os.getenv("CRAG_PASSAGES_PER_DOC", "2"))
# Minimum cross-encoder relevance probability for a document to survive correction.
CRAG_MIN_RELEVANCE = float(os.getenv("CRAG_MIN_RELEVANCE", "0.0"))

# Generator-ablation run directories shown by the /ablation dashboard.
ABLATION_RUNS_DIR = Path(os.getenv("ABLATION_RUNS_DIR", str(Path(__file__).parent / "data" / "ablation_runs")))
# Written by scripts/evaluate_calibrator.py; shown on the dashboard's Calibrator view.
CALIBRATION_REPORT_PATH = Path(os.getenv("CALIBRATION_REPORT_PATH", str(Path(__file__).parent / "data" / "calibration_report.json")))
# Every experiment result is saved here (see core/results_store.py).
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(Path(__file__).parent / "results")))
