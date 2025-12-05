"""
persistence.py
---------------
Gestion centralisée de la persistance dans SoDistinct :
- Sauvegarde/chargement de SimulationResult
- Checkpoints des simulations/batchs
- Cache local (pickle)
- Support optionnel Redis si installé

Basé sur settings.storage.*
"""

from __future__ import annotations

import pickle
import json
import time
from pathlib import Path
from typing import Any, Optional

from sodistinct.core.engine import SimulationResult
from sodistinct.config.settings import settings
from sodistinct.utils.logging import get_logger

logger = get_logger("sodistinct.utils.persistence")

# Essai de Redis (optionnel)
try:
    import redis
except Exception:
    redis = None


# ==============================================================================
# Répertoires
# ==============================================================================

DATA_DIR = Path(settings.storage.data_dir)
RESULTS_DIR = Path(settings.storage.results_dir)
TMP_DIR = Path(settings.storage.tmp_dir)

for d in (DATA_DIR, RESULTS_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Cache local (pickle)
# ==============================================================================

def _cache_path(key: str) -> Path:
    """Retourne le chemin du fichier cache associé à une clé."""
    safe_key = key.replace("/", "_")
    return TMP_DIR / f"cache_{safe_key}.pkl"


def cache_set(key: str, value: Any):
    """Stocke un objet dans le cache disque."""
    path = _cache_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump(value, f)
        logger.debug("Cache set: %s", key)
    except Exception as e:
        logger.error("Erreur écriture cache %s: %s", key, e)


def cache_get(key: str) -> Optional[Any]:
    """Récupère un objet depuis le cache disque."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error("Erreur lecture cache %s: %s", key, e)
        return None


# ==============================================================================
# Cache Redis (optionnel)
# ==============================================================================

def get_redis_client() -> Optional[Any]:
    """Retourne un client Redis si disponible."""
    if redis is None:
        return None

    try:
        return redis.Redis()
    except Exception:
        return None


def redis_set(key: str, value: Any, expire: Optional[int] = None):
    client = get_redis_client()
    if client is None:
        return

    try:
        client.set(key, pickle.dumps(value), ex=expire)
    except Exception as e:
        logger.error("Erreur Redis set %s: %s", key, e)


def redis_get(key: str) -> Optional[Any]:
    client = get_redis_client()
    if client is None:
        return None

    try:
        raw = client.get(key)
        if raw is None:
            return None
        return pickle.loads(raw)
    except Exception as e:
        logger.error("Erreur Redis get %s: %s", key, e)
        return None


# ==============================================================================
# Sauvegarde / Chargement de SimulationResult
# ==============================================================================

def save_result(result: SimulationResult, name: Optional[str] = None) -> Path:
    """
    Sauvegarde un SimulationResult sous forme pickle.
    """
    if name is None:
        ts = int(time.time())
        name = f"result_{result.model_name}_{ts}.pkl"

    path = RESULTS_DIR / name
    with open(path, "wb") as f:
        pickle.dump(result, f)

    logger.info("Résultat sauvegardé : %s", path)
    return path


def load_result(path: str | Path) -> SimulationResult:
    """
    Charge un SimulationResult sauvegardé.
    """
    path = Path(path)
    with open(path, "rb") as f:
        result = pickle.load(f)

    return result


# ==============================================================================
# Checkpoints
# ==============================================================================

def save_checkpoint(data: Any, name: str = "checkpoint.pkl") -> Path:
    """
    Sauvegarde un checkpoint (pickle) dans TMP_DIR.
    """
    path = TMP_DIR / name
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.debug("Checkpoint sauvegardé : %s", path)
        return path
    except Exception as e:
        logger.error("Erreur sauvegarde checkpoint %s: %s", path, e)
        raise


def load_checkpoint(name: str = "checkpoint.pkl") -> Optional[Any]:
    """
    Charge un checkpoint si disponible.
    """
    path = TMP_DIR / name
    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error("Erreur lecture checkpoint %s: %s", path, e)
        return None


# ==============================================================================
# Rotation / nettoyage des résultats
# ==============================================================================

def clear_old_results(max_results: int = None):
    """
    Supprime les résultats les plus anciens si on dépasse max_results.
    Utilise settings.storage.max_result_history si max_results=None.
    """
    if max_results is None:
        max_results = settings.storage.max_result_history

    files = sorted(
        RESULTS_DIR.glob("*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if len(files) <= max_results:
        return

    to_delete = files[max_results:]

    for f in to_delete:
        try:
            f.unlink()
            logger.info("Suppression ancien résultat : %s", f)
        except Exception as e:
            logger.error("Erreur suppression %s : %s", f, e)
