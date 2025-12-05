"""
logging.py
-----------
Système de logging centralisé pour SoDistinct.

Fonctionnalités :
- Initialisation unique (idempotente)
- Format texte ou JSON
- Output console + fichier
- Chargé automatiquement via settings
"""

from __future__ import annotations

import logging
import json
import sys
from pathlib import Path
from typing import Optional

from sodistinct.config.settings import settings


# ==============================================================================
# JSON Formatter simple
# ==============================================================================

class JsonFormatter(logging.Formatter):
    """Format JSON minimal pour logs structurés."""
    
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }

        # Ajout d'extra si présents
        if record.args:
            payload["args"] = record.args

        return json.dumps(payload)


# ==============================================================================
# Initialisation du système de logs
# ==============================================================================

_LOGGING_INITIALIZED = False


def init_logging():
    """
    Initialise le logging global de SoDistinct.
    - Lit settings.logging
    - Ajoute handlers console + fichier (si défini)
    - Format JSON optionnel
    """

    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    cfg = settings.logging
    log_level = getattr(logging, cfg.level.upper(), logging.INFO)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Supprime tous les handlers pour éviter les doublons
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Formatter
    if cfg.json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s — %(message)s"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optionnel : fichier
    if cfg.log_file:
        log_path = Path(cfg.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("Logging initialisé — niveau: %s", cfg.level.upper())

    _LOGGING_INITIALIZED = True


# ==============================================================================
# Helper pour utilisation rapide
# ==============================================================================

def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré."""
    init_logging()
    return logging.getLogger(name)
