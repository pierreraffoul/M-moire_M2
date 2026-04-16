"""
Logger structuré JSON pour le pipeline de mesure MCP.

Chaque appel MCP produit une ligne JSON sur stderr/fichier, indépendante
des logs applicatifs. Le format est conçu pour l'analyse post-benchmark :
grouper par run_id, déduire les redondances via mcp_params_hash, calculer
le coût par architecture.

Usage :
    from src.instrumentation.logger import get_logger, setup_logging

    setup_logging(log_file="results/logs/run_001.jsonl")
    log = get_logger()
    log.info("mcp_call", tool="search_code", duration_ms=342)
"""

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Configure structlog pour produire des logs JSON structurés.

    Args:
        log_file: Chemin vers le fichier JSONL de sortie. Si None, écrit sur stderr.
        level: Niveau de log Python (logging.INFO par défaut).
    """
    # Processors communs (appliqués avant le renderer final)
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Configurer structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Renderer JSON pour la sortie finale
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    # Handler : fichier JSONL ou stderr
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(formatter)

    # Appliquer au root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def get_logger(name: str = "audit") -> structlog.stdlib.BoundLogger:
    """Retourne un logger structlog nommé.

    Args:
        name: Nom du logger (apparaît dans le champ "logger" du JSON).

    Returns:
        Logger structlog bound prêt à l'emploi.
    """
    return structlog.get_logger(name)
