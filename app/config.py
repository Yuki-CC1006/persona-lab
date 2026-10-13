from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.branding import DB_FILENAME

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""

    redaction_enabled: bool = True
    bind_host: str = "127.0.0.1"
    bind_port: int = 8765

    database_url: str = f"sqlite:///{ROOT_DIR / 'data' / DB_FILENAME}"
    upload_dir: str = str(ROOT_DIR / "data" / "uploads")
    backup_dir: str = str(ROOT_DIR / "data" / "backups")
    backup_keep: int = 5

    # Extraction / prediction thresholds
    min_prediction_confidence: float = 0.55
    min_prediction_evidence: int = 2
    chunk_max_messages: int = 40
    chunk_max_chars: int = 6000


@lru_cache
def get_settings() -> Settings:
    return Settings()


def sqlite_file_path(database_url: str) -> Path | None:
    """Filesystem path behind a sqlite URL, or ``None`` for in-memory/other."""
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return None
    raw = database_url[len(prefix) :]
    if not raw or raw.startswith(":memory:"):
        return None
    return Path(raw)
