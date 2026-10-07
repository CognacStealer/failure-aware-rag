import unittest
from types import SimpleNamespace

from core import engine
from core.router import AdaptiveRouter
from pipelines.adaptive_pipeline import AdaptivePipeline
from pipelines.crag_pipeline import CRAGPipeline
from pipelines.hybrid_pipeline import HybridPipeline

SETTINGS = SimpleNamespace(
    RRF_ALPHA=20, RRF_DENSE_WEIGHT=0.8, RRF_BM25_WEIGHT=1.0, RRF_CANDIDATES=6,
    ROUTER_ABSTAIN_MEAN_THRESHOLD=0.2, ROUTER_ABSTAIN_STD_THRESHOLD=0.3, ROUTER_FAST_MEAN_THRESHOLD=0.8,
)


class FakeRetriever:
    def __init__(self, key, ids):
        self.key, self.ids = key, ids

    def retrieve(self, query, top_k):
        return [{"doc_id": i, "title": i.upper(), "content": f"text {i}", self.key: 1.0 / (n + 1)}
                for n, i in enumerate(self.ids[:top_k])]


class FakeCorrector:
    def correct(self, query, documents, top_k):
        ranked = sorted(documents, key=lambda d: d["doc_id"], reverse=True)
        return [{**d, "relevance_score": 0.9, "passage": d["content"]} for d in ranked[:top_k]]


class FakeCalibrator:
    is_fitted = True

    def __init__(self, mean, std=0.05):
        self.mean, self.std = mean, std

    def extract_signals(self, query, documents):
        return {"dense_top": 1.0}

    def predict_distribution(self, signals):
        return self.mean, self.std


class FakeGenerator:
    model_name = "fake-model"

    def stream(self, query, documents):
        yield "an "
        yield "answer"

    def generate(self, query, documents):
        return "".join(self.stream(query, documents))


def make_service(mean):
    dense = FakeRetriever("dense_score", ["a", "b", "c", "d", "e", "f"])
    sparse = FakeRetriever("bm25_score", ["c", "x", "a", "y", "z", "w"])
    generator = FakeGenerator()
    hybrid = HybridPipeline(dense, sparse, generator, SETTINGS)
    crag = CRAGPipeline(hybrid, FakeCorrector(), generator, retrieval_multiplier=2)
    calibrator, router = FakeCalibrator(mean), AdaptiveRouter(SETTINGS)
    return SimpleNamespace(dense=dense, sparse=sparse, hybrid=hybrid, crag=crag, calibrator=calibrator,
                           router=router, generator=generator,
                           adaptive=AdaptivePipeline(hybrid, crag, calibrator, router, generator))


def stages(events):
    return [(e["stage"], e["status"]) for e in events if e["type"] == "stage" and e["status"] != "running"]


class EngineTests(unittest.TestCase):
    def test_corrective_track_matches_the_adaptive_pipeline(self):
        service = make_service(mean=0.5)
        events = list(engine.run(service, "q", 3, "adaptive"))
        self.assertEqual(stages(events), [("dense", "done"), ("bm25", "done"), ("fusion", "done"),
                                          ("calibrator", "done"), ("route", "done"), ("rerank", "done"),
                                          ("generate", "done")])
        done = events[-1]
        expected = service.adaptive.run("q", 3)
        self.assertEqual(done["track"], expected["track"])
        self.assertEqual(done["answer"], expected["answer"])
        sources = next(e for e in events if e["type"] == "sources")["documents"]
        self.assertEqual([d["doc_id"] for d in sources], [d["doc_id"] for d in expected["documents"]])
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "an answer")

    def test_fast_track_skips_rerank_and_uses_hybrid_order(self):
        service = make_service(mean=0.9)
        events = list(engine.run(service, "q", 3, "adaptive"))
        self.assertIn(("rerank", "skipped"), stages(events))
        sources = next(e for e in events if e["type"] == "sources")["documents"]
        self.assertEqual([d["doc_id"] for d in sources], [d["doc_id"] for d in service.hybrid.retrieve("q", 3)])

    def test_abstain_skips_generation(self):
        events = list(engine.run(make_service(mean=0.1), "q", 3, "adaptive"))
        self.assertIn(("generate", "skipped"), stages(events))
        self.assertEqual(events[-1]["answer"], engine.ABSTAIN_ANSWER)
        self.assertEqual(events[-1]["track"], "Abstain")

    def test_vanilla_and_crag_plans(self):
        vanilla = list(engine.run(make_service(0.5), "q", 2, "vanilla"))
        self.assertEqual(stages(vanilla), [("dense", "done"), ("generate", "done")])
        self.assertEqual(vanilla[-1]["track"], "Vanilla")
        service = make_service(0.5)
        crag = list(engine.run(service, "q", 2, "crag"))
        self.assertEqual(crag[-1]["track"], "CRAG")
        sources = next(e for e in crag if e["type"] == "sources")["documents"]
        self.assertEqual([d["doc_id"] for d in sources], [d["doc_id"] for d in service.crag.retrieve_corrected("q", 2)])

    def test_trace_names_each_function_and_its_documents(self):
        service = make_service(mean=0.5)
        events = list(engine.run(service, "q", 3, "adaptive"))
        trace = events[-1]["trace"]
        self.assertEqual([(t["stage"], t["status"]) for t in trace], stages(events))
        self.assertEqual([t["step"] for t in trace], list(range(1, len(trace) + 1)))
        by_stage = {t["stage"]: t for t in trace}
        self.assertEqual(by_stage["fusion"]["function"], "core.hybrid_rrf.reciprocal_rank_fusion")
        self.assertEqual(by_stage["route"]["function"], "core.router.AdaptiveRouter.route")
        self.assertEqual(by_stage["route"]["outputs"]["track"], "Corrective")
        sources = next(e for e in events if e["type"] == "sources")["documents"]
        self.assertEqual([d["doc_id"] for d in by_stage["generate"]["inputs"]["documents"]],
                         [d["doc_id"] for d in sources])
        self.assertEqual(by_stage["dense"]["outputs"]["count"], len(by_stage["dense"]["outputs"]["documents"]))

    def test_trace_records_skipped_stages(self):
        trace = list(engine.run(make_service(mean=0.1), "q", 3, "adaptive"))[-1]["trace"]
        self.assertEqual(trace[-1]["stage"], "generate")
        self.assertEqual(trace[-1]["status"], "skipped")

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            list(engine.run(make_service(0.5), "q", 2, "bogus"))


if __name__ == "__main__":
    unittest.main()
