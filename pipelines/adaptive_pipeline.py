from typing import Any


class AdaptivePipeline:
    ABSTAIN_ANSWER = "I don't know."

    def __init__(self, hybrid_pipeline: Any, crag_pipeline: Any, calibrator: Any, router: Any, generator: Any):
        self.hybrid = hybrid_pipeline
        self.crag = crag_pipeline
        self.calibrator = calibrator
        self.router = router
        self.generator = generator

    def run(self, query: str, top_k: int) -> dict[str, Any]:
        # Retrieve the Corrective track's wider pool once. Fusion output is a ranked list,
        # so its top_k prefix is what plain hybrid retrieval returns; the calibrator and the
        # Fast track use that prefix, and the rerank reuses the pool instead of retrieving again.
        pool = self.hybrid.retrieve(query, self.crag.pool_size(top_k))
        candidates = pool[:top_k]
        signals = self.calibrator.extract_signals(query, candidates)
        mean, std = self.calibrator.predict_distribution(signals)
        decision = self.router.route(mean, std)

        if decision.track == "Abstain":
            documents = candidates
            answer = self.ABSTAIN_ANSWER
        elif decision.track == "Fast":
            documents = candidates
            answer = self.generator.generate(query, documents)
        else:
            documents = self.crag.rerank(query, pool, top_k)
            answer = self.generator.generate(query, documents)

        return {
            "track": decision.track,
            "confidence": {"mean": decision.mean, "std": decision.std},
            "answer": answer,
            "documents": documents,
        }
