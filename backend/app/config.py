"""
Application Configuration
=========================
Centralised settings using pydantic-settings.

- DATABASE_URL defaults to SQLite for local development.
- Switch to PostgreSQL by setting the DATABASE_URL env var:
    export DATABASE_URL=postgresql://user:pass@host:5432/market_analyzer
"""

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    All configuration is loaded from environment variables (or .env file).
    Defaults are provided for frictionless local development.
    """

    # ── Database ──────────────────────────────────────────────────────────
    # SQLite for local dev; PostgreSQL in Docker / Render.
    DATABASE_URL: str = "sqlite:///./market_analyzer.db"

    # ── CORS ──────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",    # Next.js dev server
        "http://127.0.0.1:3000",
    ]

    # Additional CORS origins from env (comma-separated).
    # Set EXTRA_CORS_ORIGINS=https://your-app.onrender.com for deployment.
    EXTRA_CORS_ORIGINS: str = ""

    # ── Financial Constants ───────────────────────────────────────────────
    RISK_FREE_RATE: float = 0.065           # Indian T-bill ~6.5 %
    TRADING_DAYS_PER_YEAR: int = 252
    DEFAULT_STARTING_CAPITAL: float = 100_000.0  # INR
    DEFAULT_LOOKBACK_YEARS: int = 5

    # ── Experiment Tracking (MLflow) ──────────────────────────────────────
    # No-op automatically if the `mlflow` package is not installed.
    MLFLOW_ENABLED: bool = True
    # Local default: a SQLite store in backend/ (supports the model registry).
    # Point at a server instead, e.g. http://localhost:5000 (docker compose).
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "market-direction-classifier"
    # Register each selected model as a new version of
    # "market-direction-<TICKER>" in the MLflow Model Registry.
    MLFLOW_REGISTER_MODELS: bool = True

    # ── Walk-forward validation ───────────────────────────────────────────
    # Defaults keep /api/analyze responsive on a small instance; the offline
    # evaluation (experiments/evaluate.py) re-tunes at every fold.
    WF_INITIAL_TRAIN_DAYS: int = 504   # ~2 trading years before the first OOS day
    WF_STEP_DAYS: int = 63             # refit and predict one quarter at a time
    WF_RETUNE_EVERY: int = 4           # re-run the hyperparameter search every N folds (~yearly)
    ML_N_JOBS: int = -1                # joblib workers for tuning (-1 = all cores)

    @property
    def all_cors_origins(self) -> List[str]:
        """Combine static and dynamic CORS origins."""
        origins = list(self.CORS_ORIGINS)
        if self.EXTRA_CORS_ORIGINS:
            origins.extend(
                o.strip() for o in self.EXTRA_CORS_ORIGINS.split(",") if o.strip()
            )
        return origins

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
