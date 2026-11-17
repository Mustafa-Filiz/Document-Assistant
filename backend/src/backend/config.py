import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
# e5 cosine scores are compressed (unrelated text still scores ~0.7-0.8). This default is a
# STARTING GUESS, not a measured value: run `python -m backend.rag.calibrate` on your index and
# set MIN_SIMILARITY from the result.
DEFAULT_MIN_SIMILARITY = 0.80


@dataclass(frozen=True)
class Settings:
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "dev"))
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
            if o.strip()
        ]
    )
    # Week 2: ingestion
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR") or _REPO_ROOT / "data"))
    chroma_path: Path = field(
        default_factory=lambda: Path(os.getenv("CHROMA_PATH") or _REPO_ROOT / ".chroma")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    )
    # Week 3: retrieval + answer generation
    rag_strategy: str = field(default_factory=lambda: os.getenv("RAG_STRATEGY") or "clause")
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K") or 5))
    min_similarity: float = field(
        default_factory=lambda: float(os.getenv("MIN_SIMILARITY") or DEFAULT_MIN_SIMILARITY)
    )
    llm_provider: str = field(default_factory=lambda: (os.getenv("LLM_PROVIDER") or "anthropic").lower())
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY") or "")
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL") or "")
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or "")
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS") or 800))
    llm_timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT") or 60))


settings = Settings()
