"""Chroma persistent store. One collection per chunking strategy so strategies can be compared."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import settings
from .models import Chunk


def get_client(path: Optional[Path] = None):
    import chromadb

    return chromadb.PersistentClient(path=str(path or settings.chroma_path))


def collection_name(strategy: str) -> str:
    return f"legal_{strategy}"


def get_collection(client, strategy: str, model_name: str, reset: bool = False):
    name = collection_name(strategy)
    if reset:
        try:
            client.delete_collection(name)
        except Exception:  # collection did not exist
            pass
    col = client.get_or_create_collection(
        name, metadata={"hnsw:space": "cosine", "embedding_model": model_name}
    )
    existing = (col.metadata or {}).get("embedding_model")
    if existing and existing != model_name:
        raise RuntimeError(
            f"Collection {name!r} was built with {existing!r}, not {model_name!r}. "
            "Vectors from different models are not comparable - re-run with --reset."
        )
    return col


def replace_document(col, doc_id: str, chunks: List[Chunk], embeddings: Sequence, batch: int = 256) -> None:
    """Idempotent: delete the document's old chunks, then insert the new ones."""
    col.delete(where={"doc_id": doc_id})
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        col.upsert(
            ids=[c.chunk_id for c in part],
            embeddings=[list(map(float, e)) for e in embeddings[i : i + batch]],
            documents=[c.text for c in part],
            metadatas=[c.to_metadata() for c in part],
        )


def query(col, query_embedding, k: int = 5, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
    res = col.query(
        query_embeddings=[list(map(float, query_embedding))],
        n_results=k,
        where={"doc_id": doc_id} if doc_id else None,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": d, "meta": m, "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
