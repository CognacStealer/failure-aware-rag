"""Runtime resources shared by the API routes."""

import chromadb

import config
from core.generator import TextGenerator
from core.retriever_bm25 import BM25Retriever
from core.retriever_chroma import ChromaDenseRetriever
from pipelines.hybrid_pipeline import HybridPipeline
from pipelines.vanilla_pipeline import VanillaPipeline


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

    def initialize(self) -> None:
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.collection = self.client.get_collection(name=config.CHROMA_DOCS_COLLECTION)
        count = self.collection.count()
        if count == 0:
            raise RuntimeError(f"Chroma collection '{config.CHROMA_DOCS_COLLECTION}' is empty")

        records = []
        batch_size = 5000
        for offset in range(0, count, batch_size):
            batch = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            for doc_id, content, metadata in zip(
                batch["ids"], batch["documents"] or [], batch["metadatas"] or []
            ):
                records.append({
                    "doc_id": doc_id,
                    "content": content or "",
                    "title": (metadata or {}).get("title", ""),
                    "metadata": metadata or {},
                })
        self.sparse = BM25Retriever(records)
        self.dense = ChromaDenseRetriever(
            self.collection, config.EMBEDDING_MODEL, mode=config.DENSE_QUERY_MODE
        )
        self.generator = TextGenerator(
            config.GENERATOR_MODEL_DEFAULT,
            load_on_startup=config.GENERATOR_LOAD_ON_STARTUP,
        )
        self.vanilla = VanillaPipeline(self.dense, self.generator)
        self.hybrid = HybridPipeline(self.dense, self.sparse, self.generator, config)
        self.ready = True
        self.error = None

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "chroma_connected": self.collection is not None,
            "bm25_built": self.sparse is not None,
            "generator_loaded": bool(self.generator and self.generator.loaded),
            "error": self.error,
        }


service = RAGService()
