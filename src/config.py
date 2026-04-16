"""
Configuration centralisée via pydantic-settings.

Lit les variables d'environnement (ou .env) une seule fois au démarrage.
Toutes les clés sensibles sont marquées Secret pour éviter leur affichage
dans les logs.

Usage :
    from src.config import get_settings
    settings = get_settings()
    token = settings.github_token.get_secret_value()
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres de configuration du benchmark."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Clés API ──────────────────────────────────────────────────────────────
    anthropic_api_key: SecretStr = Field(
        description="Clé API Anthropic (ANTHROPIC_API_KEY)."
    )
    github_token: SecretStr = Field(
        description="Token GitHub pour le serveur MCP (GITHUB_TOKEN)."
    )

    # ── Modèle LLM ────────────────────────────────────────────────────────────
    model_name: str = Field(
        default="claude-sonnet-4-5",
        description="ID du modèle Anthropic utilisé pour tous les agents.",
    )

    # ── Chemins ───────────────────────────────────────────────────────────────
    mcp_binary_path: Path = Field(
        default=Path(__file__).parent.parent / "github-mcp-server",
        description="Chemin vers le binaire github-mcp-server.",
    )
    results_dir: Path = Field(
        default=Path(__file__).parent.parent / "results",
        description="Répertoire de sortie pour les logs et rapports.",
    )

    # ── Benchmark ────────────────────────────────────────────────────────────
    max_iterations: int = Field(
        default=20,
        description="Nombre maximum d'itérations ReAct par agent.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retourne les settings en cache (singleton).

    Raises:
        ValidationError: Si une variable requise est manquante.
    """
    return Settings()
