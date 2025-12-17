from __future__ import annotations

import ray
import logging
from typing import List, Iterable, Dict, Any, Optional

from sodistinct.core.engine import run_simulation, SimulationResult
from sodistinct.core.models import DiffusionModel
from sodistinct.core.graph_wrapper import GraphWrapper

logger = logging.getLogger("sodistinct.distributed.ray_backend")

def init_ray(address: Optional[str] = None, num_cpus: Optional[int] = None):

    if ray.is_initialized():
        return False

    ray.init(address=address, num_cpus=num_cpus, ignore_reinit_error=True)
    logger.info(f"Ray initialisé (address={address}, num_cpus={num_cpus})")
    return True


@ray.remote
def _ray_run_simulation(
    model: DiffusionModel,
    graph: GraphWrapper,
    seed_set: Iterable[Any],
    params: Dict[str, Any],
    rng_seed: Optional[int],
) -> SimulationResult:
    """Tâche Ray minimaliste pour exécuter une simulation."""
    return run_simulation(model, graph, seed_set, params, rng_seed)



@ray.remote
class SimulationActor:

    def __init__(self, model: DiffusionModel, graph: GraphWrapper):
        self.model = model
        self.graph = graph

    def run(self, seed_set: Iterable[Any], params: Dict[str, Any], rng_seed: Optional[int]):
        return run_simulation(self.model, self.graph, seed_set, params, rng_seed)


class RayBackend:


    def __init__(
        self,
        use_actors: bool = False,
        num_actors: int = 4,
        address: Optional[str] = None,
        num_cpus: Optional[int] = None,
    ):
        self.use_actors = use_actors
        self.num_actors = num_actors
        self.address = address
        self.num_cpus = num_cpus

        
        init_ray(address=address, num_cpus=num_cpus)

       
        self.actors = []

 

    def _ensure_actors(self, model: DiffusionModel, graph: GraphWrapper):
        """Crée un pool d'acteurs si nécessaire."""
        if not self.use_actors or self.actors:
            return

        logger.info(f"Création de {self.num_actors} acteurs Ray…")
        self.actors = [
            SimulationActor.remote(model, graph)
            for _ in range(self.num_actors)
        ]



    def run_many(
        self,
        model: DiffusionModel,
        graph: GraphWrapper,
        seed_sets: List[Iterable[Any]],
        params: Dict[str, Any],
        base_rng_seed: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[SimulationResult]:
     

        self._ensure_actors(model, graph)

        futures = []

        if self.use_actors:
            logger.info(f"Distribution via {self.num_actors} acteurs Ray.")

            for i, seeds in enumerate(seed_sets):
                rng_seed = base_rng_seed + i if base_rng_seed is not None else None
                actor = self.actors[i % self.num_actors]
                fut = actor.run.remote(seeds, params, rng_seed)
                futures.append((i, seeds, fut))

        else:
            logger.info("Distribution via Ray Tasks.")

            for i, seeds in enumerate(seed_sets):
                rng_seed = base_rng_seed + i if base_rng_seed is not None else None
                fut = _ray_run_simulation.remote(model, graph, seeds, params, rng_seed)
                futures.append((i, seeds, fut))

        results = []
        for idx, seeds, fut in futures:
            res: SimulationResult = ray.get(fut)

            if progress_callback:
                progress_callback(
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



def run_batch_ray(
    model: DiffusionModel,
    graph: GraphWrapper,
    seed_sets: List[Iterable[Any]],
    params: Dict[str, Any],
    use_actors: bool = False,
    num_actors: int = 4,
    address: Optional[str] = None,
    num_cpus: Optional[int] = None,
    progress_callback: Optional[callable] = None,
) -> List[SimulationResult]:

    backend = RayBackend(
        use_actors=use_actors,
        num_actors=num_actors,
        address=address,
        num_cpus=num_cpus,
    )

    return backend.run_many(
        model=model,
        graph=graph,
        seed_sets=seed_sets,
        params=params,
        progress_callback=progress_callback,
    )
