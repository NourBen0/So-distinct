"""
parallel.py
------------
Orchestrateur parallèle synchrone basé sur ThreadPoolExecutor
ou ProcessPoolExecutor.

Objectif :
- Fournir une alternative à local_async.py pour exécuter des simulations
  en parallèle **sans asyncio**.
- Utilisé pour :
    * CLI
    * Benchmarks (bench/)
    * Tests de performance
    * Expérimentations massives localisées

API :
    ParallelExecutor(...)
    executor.run_many(...)
    executor.map_runs(...)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Iterable, List, Dict, Any, Optional, Callable

from sodistinct.core.engine import run_simulation, SimulationResult
from sodistinct.core.models import DiffusionModel
from sodistinct.core.graph_wrapper import GraphWrapper

logger = logging.getLogger("sodistinct.orchestrator.parallel")


# ==============================================================================
# Parallel Executor
# ==============================================================================

class ParallelExecutor:
    """
    Exécute plusieurs simulations en parallèle via futures.

    Paramètres :
        max_workers      : nombre de workers
        use_processes    : True → ProcessPoolExecutor (CPU bound)
                           False → ThreadPoolExecutor (I/O bound)
        progress_callback : fonction appelée après chaque run (optionnel)
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.progress_callback = progress_callback

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
        logger.warning("Annulation déclenchée — les futures restantes seront ignorées.")
        self._cancel_flag = True

    # ----------------------------------------------------------------------
    # Exécution d’une liste de seed-sets
    # ----------------------------------------------------------------------

    def run_many(
        self,
        model: DiffusionModel,
        graph: GraphWrapper,
        seed_sets: List[Iterable[Any]],
        params: Dict[str, Any],
        base_rng_seed: Optional[int] = None,
    ) -> List[SimulationResult]:
        """
        Exécute N runs en parallèle.

        Retour :
            Liste de SimulationResult
        """

        futures = []
        results = []

        for i, seeds in enumerate(seed_sets):
            if self._cancel_flag:
                break

            rng_seed = base_rng_seed + i if base_rng_seed is not None else None

            fut = self._executor.submit(
                run_simulation,
                model,
                graph,
                seeds,
                params,
                rng_seed,
            )
            futures.append((i, seeds, fut))

        for idx, seeds, fut in futures:
            if self._cancel_flag:
                break

            res = fut.result()

            if self.progress_callback:
                self.progress_callback(
                    {
                        "index": idx,
                        "total": len(seed_sets),
                        "progress": (idx + 1) / len(seed_sets),
                        "seed_set": list(seeds),
                        "steps": res.steps,
                        "runtime_ms": res.runtime_ms,
                    }
                )

            results.append(res)

        return results

    # ----------------------------------------------------------------------
    # Mapping parallèle simple
    # ----------------------------------------------------------------------

    def map_runs(
        self,
        fn: Callable[..., Any],
        args_list: List[Dict[str, Any]],
    ) -> List[Any]:
        """
        Exécute en parallèle une fonction générique.

        args_list = [
            {"arg1": ..., "arg2": ...},
            ...
        ]

        Retour : liste des résultats
        """
        results = []
        futures = []

        for args in args_list:
            if self._cancel_flag:
                break
            fut = self._executor.submit(fn, **args)
            futures.append(fut)

        for fut in as_completed(futures):
            if self._cancel_flag:
                break
            results.append(fut.result())

        return results

    # ----------------------------------------------------------------------
    # Fermeture propre
    # ----------------------------------------------------------------------

    def shutdown(self):
        logger.info("Fermeture du ParallelExecutor.")
        self._executor.shutdown(wait=True)


# ==============================================================================
# Helper très simple (API minimaliste)
# ==============================================================================

def run_batch_parallel(
    model: DiffusionModel,
    graph: GraphWrapper,
    seed_sets: List[Iterable[Any]],
    params: Dict[str, Any],
    max_workers: int = 4,
    use_processes: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[SimulationResult]:
    """
    Helper simple pour exécuter un batch sans instancier manuellement l'executor.
    """
    executor = ParallelExecutor(
        max_workers=max_workers,
        use_processes=use_processes,
        progress_callback=progress_callback,
    )

    try:
        return executor.run_many(
            model=model,
            graph=graph,
            seed_sets=seed_sets,
            params=params,
        )
    finally:
        executor.shutdown()
