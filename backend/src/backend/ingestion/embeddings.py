"""Embedding wrapper (sentence-transformers). The model is chosen via EMBEDDING_MODEL, see docs/embedding-model.md."""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..config import settings


class Embedder:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name or settings.embedding_model
        lower = self.model_name.lower()
        self.model = SentenceTransformer(
            self.model_name, device=device, trust_remote_code="jina" in lower
        )
        self.batch_size = batch_size
        # e5 models were trained with these prefixes; without them quality drops.
        self.passage_prefix = "passage: " if "e5" in lower else ""
        self.query_prefix = "query: " if "e5" in lower else ""

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def _encode(self, texts: Sequence[str], show_progress: bool):
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,  # cosine == dot product
            show_progress_bar=show_progress,
        )

    def embed_passages(self, texts: Sequence[str], show_progress: bool = False):
        return self._encode([self.passage_prefix + t for t in texts], show_progress)

    def embed_queries(self, texts: Sequence[str]):
        return self._encode([self.query_prefix + t for t in texts], False)

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        ids = self.model.tokenizer(
            [self.passage_prefix + t for t in texts], add_special_tokens=True, truncation=False
        )["input_ids"]
        return [len(x) for x in ids]
