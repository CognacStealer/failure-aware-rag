"""Character-window chunking shared by indexing and query-time passage selection."""

from core.retriever_bm25 import tokenize


def chunk_text(text: str, size: int = 1000, overlap: int = 100) -> list[str]:
    """Split text into ~size-character windows, breaking on whitespace where possible."""
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def best_passages(query: str, content: str, limit: int = 3, size: int = 1000) -> list[str]:
    """Return the chunks of a document that share the most terms with the query."""
    chunks = chunk_text(content, size=size) or [content]
    query_terms = set(tokenize(query))
    ranked = sorted(
        range(len(chunks)),
        key=lambda i: (-len(query_terms.intersection(tokenize(chunks[i]))), i),
    )
    return [chunks[i] for i in ranked[:limit]]
