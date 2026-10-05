"""Central configuration.

Secrets and machine-specific paths come from the environment (.env locally, a secrets
manager in CI/production). Nothing sensitive is hard-coded and no path contains a username.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

SEED = 42


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or weak."""


@dataclass(frozen=True)
class Settings:
    pepper: str | None
    raw_dir: Path
    data_dir: Path
    model_dir: Path
    db_path: Path
    mlflow_uri: str
    api_key: str | None

    def require_pepper(self) -> str:
        if not self.pepper or len(self.pepper) < 32:
            raise ConfigError(
                "ANONYMIZATION_PEPPER must be set to a random value of >= 32 chars "
                "(openssl rand -hex 32). See .env.example."
            )
        return self.pepper

    @property
    def processed_csv(self) -> Path:
        return self.data_dir / "processed" / "cases_anonymized.csv"

    @property
    def raw_csv(self) -> Path:
        return self.raw_dir / "raw_cases.csv"


def get_settings() -> Settings:
    """Read settings from the environment on every call (keeps tests hermetic)."""
    load_dotenv(Path.cwd() / ".env", override=False)
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    return Settings(
        pepper=os.getenv("ANONYMIZATION_PEPPER") or None,
        raw_dir=Path(os.getenv("RAW_DATA_DIR", data_dir / "raw")),
        data_dir=data_dir,
        model_dir=Path(os.getenv("MODEL_DIR", "models")),
        db_path=Path(os.getenv("DB_PATH", data_dir / "claims.db")),
        mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        api_key=os.getenv("API_KEY") or None,
    )
