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
        candidates = self.hybrid.retrieve(query, top_k)
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
            documents = self.crag.retrieve_corrected(query, top_k)
            answer = self.generator.generate(query, documents)

        return {
            "track": decision.track,
            "confidence": {"mean": decision.mean, "std": decision.std},
            "answer": answer,
            "documents": documents,
        }
