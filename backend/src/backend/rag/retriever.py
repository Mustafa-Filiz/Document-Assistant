"""Question -> embedding -> k most similar chunks from the Chroma index built in week 2."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import settings
from ..ingestion import store


class IndexNotReadyError(RuntimeError):
    """The vector index is missing, empty, or was built with a different embedding model."""


@dataclass
class Hit:
    score: float  # cosine similarity in [-1, 1]; higher = closer
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


class Retriever:
    def __init__(self, embedder, collection, strategy: str = "clause"):
        self.embedder = embedder
        self.collection = collection
        self.strategy = strategy
        self._lock = threading.Lock()  # the encoder is not guaranteed thread-safe

    @classmethod
    def from_settings(
        cls,
        strategy: Optional[str] = None,
        model_name: Optional[str] = None,
        chroma_path: Optional[Path] = None,
    ) -> "Retriever":
        strategy = strategy or settings.rag_strategy
        model_name = model_name or settings.embedding_model
        chroma_path = Path(chroma_path or settings.chroma_path)
        name = store.collection_name(strategy)
        try:
            client = store.get_client(chroma_path)
            # get_collection (not get_or_create): never create an empty index by accident.
            col = client.get_collection(name)
        except ImportError as e:
            raise IndexNotReadyError(
                "chromadb / sentence-transformers are not installed. "
                "Run `uv sync --extra ingest` (the Docker image does this already)."
            ) from e
        except Exception as e:
            raise IndexNotReadyError(
                f"No collection {name!r} in {chroma_path}. Build it first: "
                f"`python -m backend.ingestion --index --strategy {strategy}`"
            ) from e
        if col.count() == 0:
            raise IndexNotReadyError(f"Collection {name!r} is empty. Re-run the ingestion with --index.")
        built_with = (col.metadata or {}).get("embedding_model")
        if built_with and built_with != model_name:
            raise IndexNotReadyError(
                f"Index {name!r} was built with {built_with!r} but EMBEDDING_MODEL is {model_name!r}. "
                f"Set EMBEDDING_MODEL={built_with} or rebuild the index with --reset."
            )
        from ..ingestion.embeddings import Embedder  # heavy import, keep lazy

        return cls(Embedder(model_name), col, strategy)

    def search(self, question: str, k: int = 5, doc_id: Optional[str] = None) -> List[Hit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        with self._lock:
            q_emb = self.embedder.embed_queries([question])[0]
        n = min(k, self.collection.count())
        if n == 0:
            return []
        rows = store.query(self.collection, q_emb, k=n, doc_id=doc_id)
        # The collection uses cosine space: Chroma's distance = 1 - cosine similarity.
        hits = [Hit(score=round(1.0 - float(r["distance"]), 4), text=r["text"], meta=r["meta"]) for r in rows]
        hits.sort(key=lambda h: -h.score)
        return hits
