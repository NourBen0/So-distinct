"""
local_async.py
---------------
Orchestrateur local asynchrone pour exécuter plusieurs simulations
en parallèle via asyncio.

Objectifs :
- Exécuter N runs en parallèle dans un seul nœud local
- Supporter à la fois :
    * ThreadPoolExecutor (léger)
    * ProcessPoolExecutor (CPU-bound)
- Exposer une API simple utilisable par FastAPI, dashboards, CLI
- Gérer la progression, l'annulation et la récupération propre

Ce module ne gère PAS la distribution (voir ray_backend & dask_backend).
"""

from __future__ import annotations

import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, Iterable, List, Dict, Any, Optional

from sodistinct.core.engine import run_simulation, SimulationResult
from sodistinct.core.models import DiffusionModel
from sodistinct.core.graph_wrapper import GraphWrapper

logger = logging.getLogger("sodistinct.orchestrator.local_async")


# ==============================================================================
# Orchestrateur local principal
# ==============================================================================

class AsyncLocalOrchestrator:
    """
    Orchestrateur local utilisant asyncio + executors.

    Paramètres :
        max_workers      : nombre de workers pour le pool local
        use_processes    : si True, utilise ProcessPoolExecutor
        progress_enabled : si True, envoie des callbacks de progression
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = True,
        progress_enabled: bool = True,
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.progress_enabled = progress_enabled

        self._executor = (
            ProcessPoolExecutor(max_workers=max_workers)
            if use_processes
            else ThreadPoolExecutor(max_workers=max_workers)
        )

        self._cancel_flag = False

    # ----------------------------------------------------------------------
    # Annulation
    # ----------------------------------------------------------------------

    def cancel(self):
        """Annule toutes les tâches en cours."""
        logger.warning("Annulation reçue — les tâches en cours seront arrêtées.")
        self._cancel_flag = True

    # ----------------------------------------------------------------------
    # Lancement d’un batch de runs
    # ----------------------------------------------------------------------

    async def run_many(
        self,
        model: DiffusionModel,
        graph: GraphWrapper,
        seed_sets: List[Iterable[Any]],
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        base_rng_seed: Optional[int] = None,
    ) -> List[SimulationResult]:
        """
        Exécute un grand nombre de simulations en parallèle.

        Paramètres :
            model             : modèle à utiliser
            graph             : graphe à utiliser
            seed_sets         : liste de seed_sets (1 par run)
            params            : paramètres du modèle
            progress_callback : fonction appelée pour chaque progression
            base_rng_seed     : seed racine (décalé à chaque run)

        Retour :
            Liste de SimulationResult
        """

        loop = asyncio.get_running_loop()
        results: List[SimulationResult] = []

        async def submit_one(i: int, seed_set: Iterable[Any]):
            """Soumet une simulation individuelle dans l'executor."""
            if self._cancel_flag:
                return None

            rng_seed = base_rng_seed + i if base_rng_seed is not None else None

            # Exécution dans un thread ou process
            res = await loop.run_in_executor(
                self._executor,
                lambda: run_simulation(model, graph, seed_set, params, rng_seed=rng_seed),
            )

            # Progression
            if self.progress_enabled and progress_callback:
                progress_callback(
                    {
                        "index": i,
                        "total": len(seed_sets),
                        "progress": (i + 1) / len(seed_sets),
                        "seed_set": list(seed_set),
                        "steps": res.steps,
                        "runtime_ms": res.runtime_ms,
                    }
                )

            return res

        tasks = [
            asyncio.create_task(submit_one(i, seed_sets[i]))
            for i in range(len(seed_sets))
        ]

        # Collecte des résultats
        for coro in asyncio.as_completed(tasks):
            if self._cancel_flag:
                break
            res = await coro
            if res is not None:
                results.append(res)

        return results

    # ----------------------------------------------------------------------
    # Fermeture propre
    # ----------------------------------------------------------------------

    def shutdown(self):
        """Ferme l'executor local."""
        logger.info("Fermeture de l'orchestrateur local (executor).")
        self._executor.shutdown(wait=True)


# ==============================================================================
# Helper simple (API friendly)
# ==============================================================================

async def run_batch_async(
    model: DiffusionModel,
    graph: GraphWrapper,
    seed_sets: List[Iterable[Any]],
    params: Dict[str, Any],
    max_workers: int = 4,
    use_processes: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """
    Helper externe très simple pour lancer un batch de simulations.

    Utilisé souvent par:
        - API FastAPI
        - Streamlit Dashboard
        - Tests d’intégration
    """
    orchestrator = AsyncLocalOrchestrator(
        max_workers=max_workers,
        use_processes=use_processes,
        progress_enabled=True,
    )

    try:
        return await orchestrator.run_many(
            model=model,
            graph=graph,
            seed_sets=seed_sets,
            params=params,
            progress_callback=progress_callback,
        )
    finally:
        orchestrator.shutdown()
