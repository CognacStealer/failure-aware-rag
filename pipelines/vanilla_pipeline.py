from typing import Any


class VanillaPipeline:
    def __init__(self, retriever: Any, generator: Any):
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str, top_k: int) -> dict[str, Any]:
        documents = self.retriever.retrieve(query, top_k=top_k)
        answer = self.generator.generate(query, documents)
        return {"answer": answer, "documents": documents}
