"""
src/sodistinct/config/settings.py
---------------------------------
Configuration COMPATIBLE pour SoDistinct - Version simplifiée
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings

# Logger simple
logger = logging.getLogger("sodistinct.config")

# ============================================================================
# Settings principal - STRUCTURE PLATE pour compatibilité
# ============================================================================

class Settings(BaseSettings):
    # Runtime
    backend: str = Field("local", description="Backend: 'local' | 'ray' | 'dask'")
    num_workers: int = Field(4, ge=1, description="Nombre de workers")
    use_gpu: bool = Field(False, description="Utilisation GPU")
    max_concurrent_runs: Optional[int] = Field(None)
    
    # Storage
    data_dir: str = Field("data/", description="Répertoire des données")
    results_dir: str = Field("results/", description="Répertoire résultats")
    tmp_dir: str = Field("tmp/", description="Répertoire temporaire")
    persist_checkpoints: bool = Field(True, description="Sauvegarde checkpoints")
    max_result_history: int = Field(50, description="Historique résultats")
    
    # Logging
    log_level: str = Field("INFO", description="Niveau de log")
    log_json_format: bool = Field(False, description="Format JSON")
    log_file: Optional[str] = Field(None, description="Fichier de log")
    
    # Metadata - STRUCTURE PLATE pour compatibilité avec server.py
    project_name: str = Field("SoDistinct", description="Nom du projet")
    version: str = Field("0.1.0", description="Version")
    author: str = Field("Nour", description="Auteur")
    description: str = Field("Framework distribué pour la simulation de diffusion d'information", description="Description")
    
    # Extras
    random_seed: int = Field(42, description="Seed pour reproductibilité")
    
    class Config:
        env_prefix = "SODISTINCT_"
        case_sensitive = False

# ============================================================================
# Instance globale des settings
# ============================================================================

# Création de l'instance globale
settings = Settings()

# Log de confirmation
logger.info(f"✅ Settings chargés: {settings.project_name} v{settings.version}")

# ============================================================================
# Fonction pour obtenir les settings (pour compatibilité)
# ============================================================================

def get_settings() -> Settings:
    """Renvoie l'instance des settings (pour compatibilité avec d'autres modules)"""
    return settings

# ============================================================================
# Valeurs par défaut pour les modèles (utile pour les extras)
# ============================================================================

def get_default_model_params() -> Dict[str, Any]:
    """Retourne les paramètres par défaut des modèles"""
    return {
        "ic": {"p": 0.03},
        "lt": {"threshold": 0.2},
        "si": {"transmission_rate": 0.1},
        "sir": {"transmission_rate": 0.1, "recovery_rate": 0.05}
    }

def get_experiment_defaults() -> Dict[str, Any]:
    """Retourne les valeurs par défaut des expériences"""
    return {
        "runs": 50,
        "graph_format": "edgelist",
        "autosave_results": True
    }