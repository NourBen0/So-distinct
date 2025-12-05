"""
dask_backend.py
----------------
Backend distribué basé sur Dask pour exécuter des simulations SoDistinct
sur un cluster Dask (local ou distant).

Fonctionnalités :
- Initialisation/connexion à un Dask Client
- Scatter (broadcast) du graphe et (optionnellement) du modèle vers les workers
- Soumission de tâches via client.submit / client.map
- Collecte des résultats et reporting de progression
- API simple et compatible avec SoDistinct (engine/models/graph_wrapper)
"""

from __future__ import annotations

import logging
from typing import List, Iterable, Dict, Any, Optional, Callable

try:
    from dask.distributed import Client, as_completed
except Exception as e:  # fallback si dask n'est pas installé
    Client = None  # type: ignore
    as_completed = None  # type: ignore
    _dask_import_error = e

from sodistinct.core.engine import run_simulation, SimulationResult
from sodistinct.core.models import DiffusionModel
from sodistinct.core.graph_wrapper import GraphWrapper

logger = logging.getLogger("sodistinct.distributed.dask_backend")


# ==============================================================================
# Helpers d'initialisation
# ==============================================================================

def init_dask(address: Optional[str] = None, **client_kwargs) -> Any:
    """
    Initialise un Client Dask (local par défaut).
    """
    if Client is None:
        raise RuntimeError(
            "Dask n'est pas disponible. Installez 'dask[distributed]' pour utiliser ce backend. "
            f"Erreur d'import: {_dask_import_error}"
        )

    if address:
        logger.info("Connexion au cluster Dask à l'adresse %s ...", address)
        client = Client(address, **client_kwargs)
    else:
        logger.info("Initialisation d'un Client Dask en mode local ...")
        client = Client(**client_kwargs)

    logger.info("Dask client initialisé. Dashboard : %s", getattr(client, "dashboard_link", None))
    return client


# ==============================================================================
# Tâche exécutée par les workers Dask
# ==============================================================================

def _dask_run_simulation(
    model: DiffusionModel,
    graph: GraphWrapper,
    seed_set: Iterable[Any],
    params: Dict[str, Any],
    rng_seed: Optional[int],
) -> SimulationResult:
    return run_simulation(model, graph, seed_set, params, rng_seed)


# ==============================================================================
# Backend principal Dask
# ==============================================================================

class DaskBackend:
    """
    Backend Dask pour exécuter de nombreuses simulations en parallèle.

    Args:
        address: adresse du scheduler Dask (None → local)
        client: client Dask existant (optionnel)
        scatter_graph: si True → diffusion du graphe aux workers
    """

    def __init__(
        self,
        address: Optional[str] = None,
        client: Optional[Any] = None,   # FIX PYLANCE
        scatter_graph: bool = True,
        **client_kwargs,
    ):
        if Client is None:
            raise RuntimeError(
                "Dask n'est pas disponible. Installez 'dask[distributed]'."
            )

        if client is not None:
            self.client: Any = client  # FIX PYLANCE
            self._owns_client = False
            logger.info("Utilisation d'un client Dask fourni par l'utilisateur.")
        else:
            self.client = init_dask(address=address, **client_kwargs)
            self._owns_client = True

        self.scatter_graph = scatter_graph
        self._scattered_graph: Optional[Any] = None  # FIX PYLANCE
        self._scattered_model: Optional[Any] = None  # FIX PYLANCE

    # ----------------------------------------------------------------------
    # Scatter (broadcast) graph/model
    # ----------------------------------------------------------------------

    def _maybe_scatter_graph(self, graph: GraphWrapper) -> Any:
        if not self.scatter_graph:
            return graph

        if self._scattered_graph is not None:
            return self._scattered_graph

        logger.info("Broadcast du graphe vers les workers Dask...")
        self._scattered_graph = self.client.scatter(graph, broadcast=True)
        return self._scattered_graph

    def _maybe_scatter_model(self, model: DiffusionModel) -> Any:
        if self._scattered_model is not None:
            return self._scattered_model

        try:
            self._scattered_model = self.client.scatter(model, broadcast=True)
            return self._scattered_model
        except Exception:
            logger.debug("Scatter du modèle impossible, utilisation directe.")
            return model

    # ----------------------------------------------------------------------
    # Exécution distribuée
    # ----------------------------------------------------------------------

    def run_many(
        self,
        model: DiffusionModel,
        graph: GraphWrapper,
        seed_sets: List[Iterable[Any]],
        params: Dict[str, Any],
        base_rng_seed: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        use_scatter_for_model: bool = False,
    ) -> List[SimulationResult]:

        graph_ref = self._maybe_scatter_graph(graph)
        model_ref = self._maybe_scatter_model(model) if use_scatter_for_model else model

        futures = []

        # Soumission des runs
        for i, seeds in enumerate(seed_sets):
            rng_seed = base_rng_seed + i if base_rng_seed is not None else None

            fut = self.client.submit(
                _dask_run_simulation,
                model_ref,
                graph_ref,
                seeds,
                params,
                rng_seed,
            )
            futures.append((i, seeds, fut))

        results: List[SimulationResult] = []

        # Collecte asynchrone
        if as_completed is None:  # fallback
            for idx, seeds, fut in futures:
                res = fut.result()
                if progress_callback:
                    progress_callback(
                        {"index": idx, "total": len(seed_sets),
                         "progress": (idx + 1) / len(seed_sets),
                         "seed_set": list(seeds),
                         "steps": res.steps, "runtime_ms": res.runtime_ms}
                    )
                results.append(res)
            return results

        for completed in as_completed([f for (_, _, f) in futures]):
            res = completed.result()

            # retrouver l'index (petit coût, acceptable)
            idx, seeds = -1, []
            for i, s, fut in futures:
                if fut.key == completed.key:
                    idx, seeds = i, s
                    break

            if progress_callback:
                progress_callback(
                    {
                        "index": idx,
                        "total": len(seed_sets),
                        "progress": None if idx == -1 else (idx + 1) / len(seed_sets),
                        "seed_set": list(seeds),
                        "steps": res.steps,
                        "runtime_ms": res.runtime_ms,
                    }
                )

            results.append(res)

        return results

    # ----------------------------------------------------------------------
    # Fermeture propre
    # ----------------------------------------------------------------------

    def shutdown(self):
        try:
            if self._scattered_graph is not None:
                try:
                    self._scattered_graph.release()
                except Exception:
                    pass
                self._scattered_graph = None

            if self._scattered_model is not None:
                try:
                    self._scattered_model.release()
                except Exception:
                    pass
                self._scattered_model = None

            if self._owns_client and self.client is not None:
                logger.info("Fermeture du client Dask.")
                self.client.close()

        except Exception as e:
            logger.exception("Erreur lors du shutdown DaskBackend : %s", e)


# ==============================================================================
# Helper simple
# ==============================================================================

def run_batch_dask(
    model: DiffusionModel,
    graph: GraphWrapper,
    seed_sets: List[Iterable[Any]],
    params: Dict[str, Any],
    address: Optional[str] = None,
    scatter_graph: bool = True,
    base_rng_seed: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    **client_kwargs,
) -> List[SimulationResult]:

    backend = DaskBackend(address=address, scatter_graph=scatter_graph, **client_kwargs)
    try:
        return backend.run_many(
            model=model,
            graph=graph,
            seed_sets=seed_sets,
            params=params,
            base_rng_seed=base_rng_seed,
            progress_callback=progress_callback,
        )
    finally:
        backend.shutdown()
