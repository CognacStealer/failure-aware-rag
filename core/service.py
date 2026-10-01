"""Runtime resources shared by the API routes."""

import chromadb

import config
from core.calibrator import RetrievalCalibrator
from core.crag_corrector import CRAGCorrector, CrossEncoderScorer
from core.generator import TextGenerator
from core.retriever_bm25 import BM25Retriever, corpus_signature
from core.retriever_chroma import ChromaDenseRetriever
from core.router import AdaptiveRouter
from pipelines.adaptive_pipeline import AdaptivePipeline
from pipelines.crag_pipeline import CRAGPipeline
from pipelines.hybrid_pipeline import HybridPipeline
from pipelines.vanilla_pipeline import VanillaPipeline

BATCH_SIZE = 5000


class RAGService:
    def __init__(self):
        self.ready = False
        self.error: str | None = None
        self.client = None
        self.collection = None
        self.dense = None
        self.sparse = None
        self.generator = None
        self.vanilla = None
        self.hybrid = None
        self.crag = None
        self.adaptive = None
        self.calibrator = None
        self.router = None

    def fetch_content(self, doc_ids: list[str]) -> dict[str, str]:
        rows = self.collection.get(ids=doc_ids, include=["documents"])
        return {doc_id: content or "" for doc_id, content in zip(rows["ids"], rows["documents"] or [])}

    def _iter_records(self, count: int):
        for offset in range(0, count, BATCH_SIZE):
            batch = self.collection.get(
                limit=BATCH_SIZE, offset=offset, include=["documents", "metadatas"]
            )
            for doc_id, content, metadata in zip(
                batch["ids"], batch["documents"] or [], batch["metadatas"] or []
            ):
                yield {"doc_id": doc_id, "content": content or "", "title": (metadata or {}).get("title", "")}

    def _load_bm25(self, count: int) -> BM25Retriever:
        doc_ids = []
        for offset in range(0, count, BATCH_SIZE):
            doc_ids.extend(self.collection.get(limit=BATCH_SIZE, offset=offset, include=[])["ids"])
        signature = corpus_signature(doc_ids)
        cached = BM25Retriever.load(config.BM25_CACHE_PATH, signature, self.fetch_content)
        if cached is not None:
            return cached
        sparse = BM25Retriever(self._iter_records(count), fetch_content=self.fetch_content)
        sparse.save(config.BM25_CACHE_PATH, signature)
        return sparse

    def initialize(self) -> None:
        self.error = None
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.collection = self.client.get_collection(name=config.CHROMA_DOCS_COLLECTION)
        count = self.collection.count()
        if count == 0:
            raise RuntimeError(f"Chroma collection '{config.CHROMA_DOCS_COLLECTION}' is empty")

        self.sparse = self._load_bm25(count)
        self.dense = ChromaDenseRetriever(
            self.collection, config.EMBEDDING_MODEL, mode=config.DENSE_QUERY_MODE
        )
        self.generator = TextGenerator(
            config.GENERATOR_MODEL_DEFAULT,
            backend=config.GENERATOR_BACKEND,
            host=config.OLLAMA_HOST,
            timeout=config.GENERATOR_TIMEOUT_SECONDS,
            max_context_chars=config.GENERATOR_MAX_CONTEXT_CHARS,
            max_new_tokens=config.GENERATOR_MAX_NEW_TOKENS,
            load_on_startup=config.GENERATOR_LOAD_ON_STARTUP,
        )
        self.vanilla = VanillaPipeline(self.dense, self.generator)
        self.hybrid = HybridPipeline(self.dense, self.sparse, self.generator, config)
        self.crag = CRAGPipeline(
            self.hybrid,
            CRAGCorrector(
                CrossEncoderScorer(config.RERANKER_MODEL),
                config.CRAG_MIN_RELEVANCE,
                config.CRAG_PASSAGES_PER_DOC,
            ),
            self.generator,
            config.CRAG_RETRIEVAL_MULTIPLIER,
        )
        self.calibrator = RetrievalCalibrator(config.CALIBRATOR_MODEL_PATH)
        try:
            self.calibrator.load()
        except Exception as exc:
            self.error = f"Calibrator could not be loaded; using fallback: {type(exc).__name__}: {exc}"
        self.router = AdaptiveRouter(config)
        self.adaptive = AdaptivePipeline(
            self.hybrid, self.crag, self.calibrator, self.router, self.generator
        )
        self.ready = True

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "chroma_connected": self.collection is not None,
            "bm25_built": self.sparse is not None,
            "generator_loaded": bool(self.generator and self.generator.loaded),
            "calibrator_fitted": bool(self.calibrator and self.calibrator.is_fitted),
            "calibrator_examples": self.calibrator.example_count if self.calibrator else 0,
            "error": self.error,
        }


service = RAGService()
