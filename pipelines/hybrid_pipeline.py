from typing import Any

from core.hybrid_rrf import reciprocal_rank_fusion


class HybridPipeline:
    def __init__(self, dense_retriever: Any, sparse_retriever: Any, generator: Any, config: Any):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.generator = generator
        self.config = config

    def run(self, query: str, top_k: int) -> dict[str, Any]:
        dense = self.dense.retrieve(query, top_k=top_k)
        sparse = self.sparse.retrieve(query, top_k=top_k)
        documents = reciprocal_rank_fusion(
            dense,
            sparse,
            alpha=self.config.RRF_ALPHA,
            dense_weight=self.config.RRF_DENSE_WEIGHT,
            sparse_weight=self.config.RRF_BM25_WEIGHT,
            top_k=top_k,
        )
        answer = self.generator.generate(query, documents)
        return {"answer": answer, "documents": documents}
