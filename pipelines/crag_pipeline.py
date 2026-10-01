from typing import Any


class CRAGPipeline:
    def __init__(
        self,
        hybrid_pipeline: Any,
        corrector: Any,
        generator: Any,
        retrieval_multiplier: int = 3,
    ):
        if retrieval_multiplier < 1:
            raise ValueError("retrieval_multiplier must be at least 1")
        self.hybrid = hybrid_pipeline
        self.corrector = corrector
        self.generator = generator
        self.retrieval_multiplier = retrieval_multiplier

    def retrieve_corrected(self, query: str, top_k: int) -> list[dict[str, Any]]:
        candidates = self.hybrid.retrieve(query, top_k * self.retrieval_multiplier)
        return self.corrector.correct(query, candidates, top_k)

    def run(self, query: str, top_k: int) -> dict[str, Any]:
        documents = self.retrieve_corrected(query, top_k)
        answer = self.generator.generate(query, documents)
        return {"answer": answer, "documents": documents}
