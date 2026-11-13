import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


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


settings = Settings()
