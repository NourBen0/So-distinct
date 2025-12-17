
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



class AsyncLocalOrchestrator:

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


    def cancel(self):
        
        logger.warning("Annulation reçue — les tâches en cours seront arrêtées.")
        self._cancel_flag = True


    async def run_many(
        self,
        model: DiffusionModel,
        graph: GraphWrapper,
        seed_sets: List[Iterable[Any]],
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        base_rng_seed: Optional[int] = None,
    ) -> List[SimulationResult]:

        loop = asyncio.get_running_loop()
        results: List[SimulationResult] = []

        async def submit_one(i: int, seed_set: Iterable[Any]):
            
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

    def shutdown(self):
        """Ferme l'executor local."""
        logger.info("Fermeture de l'orchestrateur local (executor).")
        self._executor.shutdown(wait=True)


async def run_batch_async(
    model: DiffusionModel,
    graph: GraphWrapper,
    seed_sets: List[Iterable[Any]],
    params: Dict[str, Any],
    max_workers: int = 4,
    use_processes: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):

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
