"""Predict whether retrieval succeeded, with model uncertainty, from retrieval signals.

A bootstrap ensemble of logistic regressions over standardized features:
``mean`` is the ensemble's average probability that the retrieved set contains
the evidence, and ``std`` is the spread across ensemble members (epistemic
uncertainty). Unlike a regression's predictive std, the spread shrinks as
training data grows, so it can be meaningfully thresholded by the router.
"""

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.retriever_bm25 import tokenize

FORMAT_VERSION = 2


def _top_two(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values, reverse=True) + [0.0, 0.0]
    return ordered[0], ordered[1]


class RetrievalCalibrator:
    FEATURE_NAMES = (
        "query_terms",
        "bm25_top",
        "bm25_margin",
        "dense_top",
        "dense_margin",
        "agreement",
    )
    UNFITTED_PREDICTION = (0.5, 0.0)  # routes to the Corrective track by default

    def __init__(self, model_path: str | Path, n_models: int = 30, seed: int = 0):
        self.model_path = Path(model_path)
        self.n_models = n_models
        self.seed = seed
        self.models: list[Any] = []
        self.example_count = 0
        self.last_refit: str | None = None

    @property
    def is_fitted(self) -> bool:
        return bool(self.models)

    @classmethod
    def extract_signals(cls, query: str, documents: list[dict[str, Any]]) -> dict[str, float]:
        """Features of a fused (hybrid) result list; see HybridPipeline.retrieve."""
        n_terms = len(set(tokenize(query)))
        bm25 = [float(doc["bm25_score"]) for doc in documents if "bm25_score" in doc]
        dense = [float(doc["dense_score"]) for doc in documents if "dense_score" in doc]
        bm25_first, bm25_second = _top_two(bm25)
        dense_first, dense_second = _top_two(dense)
        both = sum(1 for doc in documents if "bm25_score" in doc and "dense_score" in doc)
        return {
            "query_terms": math.log1p(n_terms),
            # Raw BM25 grows with query length; per-term scores compare across queries.
            "bm25_top": bm25_first / n_terms if n_terms else 0.0,
            "bm25_margin": (bm25_first - bm25_second) / bm25_first if bm25_first > 0 else 0.0,
            "dense_top": dense_first,
            "dense_margin": dense_first - dense_second,
            "agreement": both / len(documents) if documents else 0.0,
        }

    @classmethod
    def _features(cls, signals: dict[str, float]) -> list[float]:
        return [float(signals[name]) for name in cls.FEATURE_NAMES]

    @classmethod
    def _parse(cls, examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        features, labels = [], []
        for item in examples:
            signals = item.get("signals", item)
            missing = [name for name in cls.FEATURE_NAMES if name not in signals]
            if "label" not in item:
                missing.append("label")
            if missing:
                raise ValueError(f"example is missing fields: {missing}")
            if item["label"] not in (0, 1, True, False):
                raise ValueError("labels must be 0 (retrieval failed) or 1 (retrieval succeeded)")
            features.append(cls._features(signals))
            labels.append(int(item["label"]))
        return np.asarray(features, dtype=float), np.asarray(labels, dtype=int)

    def fit(self, examples: list[dict[str, Any]]) -> None:
        features, labels = self._parse(examples)
        if len(set(labels.tolist())) < 2:
            raise ValueError("training examples must include both successful and failed retrievals")
        rng = np.random.default_rng(self.seed)
        models = []
        while len(models) < self.n_models:
            sample = rng.integers(0, len(labels), len(labels))
            if len(set(labels[sample].tolist())) < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            models.append(model.fit(features[sample], labels[sample]))
        self.models = models
        self.example_count = len(labels)
        self.last_refit = datetime.now(timezone.utc).isoformat()

    def predict_distribution(self, signals: dict[str, float]) -> tuple[float, float]:
        if not self.is_fitted:
            return self.UNFITTED_PREDICTION
        row = np.asarray([self._features(signals)])
        probabilities = np.array([model.predict_proba(row)[0, 1] for model in self.models])
        return float(probabilities.mean()), float(probabilities.std())

    def save(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("cannot persist an unfitted calibrator")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "format_version": FORMAT_VERSION,
            "feature_names": self.FEATURE_NAMES,
            "models": self.models,
            "example_count": self.example_count,
            "last_refit": self.last_refit,
        }, self.model_path)

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        saved = joblib.load(self.model_path)
        if (
            not isinstance(saved, dict)
            or saved.get("format_version") != FORMAT_VERSION
            or tuple(saved.get("feature_names", ())) != self.FEATURE_NAMES
        ):
            raise ValueError(
                f"{self.model_path} was saved by an incompatible calibrator; "
                "retrain it with scripts/train_calibrator.py"
            )
        self.models = saved["models"]
        self.example_count = int(saved.get("example_count", 0))
        self.last_refit = saved.get("last_refit")
        return True
