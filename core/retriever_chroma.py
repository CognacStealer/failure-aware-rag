from typing import Any


def _distance_space(collection: Any) -> str:
    try:
        return (collection.configuration_json.get("hnsw") or {}).get("space", "l2")
    except AttributeError:
        return "l2"


class ChromaDenseRetriever:
    """Dense search against an existing Chroma collection."""

    def __init__(self, collection: Any, embedding_model: str, mode: str = "collection"):
        self.collection = collection
        self.mode = mode
        self.space = _distance_space(collection)
        self.encoder = None
        if mode == "external":
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(embedding_model)
        elif mode != "collection":
            raise ValueError("DENSE_QUERY_MODE must be 'collection' or 'external'")

    def similarity(self, distance: float) -> float:
        """Cosine similarity for normalized embeddings, whatever Chroma's distance space."""
        if self.space == "l2":  # Chroma reports squared L2: ||a - b||^2 = 2 - 2cos
            return 1.0 - distance / 2.0
        if self.space == "cosine":
            return 1.0 - distance
        return -distance  # inner product: Chroma reports 1 - dot

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        query_args = {"n_results": top_k, "include": ["documents", "metadatas", "distances"]}
        if self.mode == "external":
            vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
            response = self.collection.query(query_embeddings=[vector], **query_args)
        else:
            response = self.collection.query(query_texts=[query], **query_args)
        rows = []
        for doc_id, content, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0] or [],
            response["metadatas"][0] or [],
            response["distances"][0] or [],
        ):
            rows.append({
                "doc_id": doc_id,
                "content": content or "",
                "title": (metadata or {}).get("title", ""),
                "metadata": metadata or {},
                "distance": float(distance),
                "dense_score": self.similarity(float(distance)),
            })
        return rows
