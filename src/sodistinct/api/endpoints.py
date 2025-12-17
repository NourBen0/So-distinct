
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException

from sodistinct.api.schemas import (
    SimulationRequest,
    BatchSimulationRequest,
    SimulationResponse,
    BatchSimulationResponse,
    MetricsRequest,
    MetricsResponse,
    GraphLoadRequest,
    GraphLoadResponse,
)

from sodistinct.core.engine import run_simulation, run_simulation_async
from sodistinct.core.models import (
    SIModel,
    SIRModel,
    ICModel,
    LTModel,
)
from sodistinct.core.metrics import compute_all_metrics
from sodistinct.core.graph_wrapper import GraphWrapper
from sodistinct.orchestrator.local_async import run_batch_async
from sodistinct.io.loader import load_graph

logger = logging.getLogger("sodistinct.api.endpoints")

router = APIRouter(tags=["simulation"])



_MODEL_REGISTRY = {
    "si": SIModel,
    "sir": SIRModel,
    "ic": ICModel,
    "lt": LTModel,
}


def get_model(model_name: str):
    model_name = model_name.lower()
    if model_name not in _MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle inconnu '{model_name}'. Models disponibles: {list(_MODEL_REGISTRY)}",
        )
    return _MODEL_REGISTRY[model_name]()


def load_graph_wrapper(path: str) -> GraphWrapper:
    try:
        g = load_graph(path)
        return GraphWrapper.from_networkx(g)
    except Exception as e:
        raise HTTPException(500, f"Erreur chargement graphe: {e}")



@router.post("/simulate", response_model=SimulationResponse)
async def simulate(req: SimulationRequest):
    """
    Exécute une seule simulation.
    Async mais exécute run_simulation dans un executor.
    """
    model = get_model(req.model)
    graph = load_graph_wrapper(req.graph_path)

    try:
        result = await run_simulation_async(
            model=model,
            graph=graph,
            seed_set=req.seed_set,
            params=req.params,
            rng_seed=req.rng_seed,
        )
    except Exception as e:
        logger.exception("Erreur simulation : %s", e)
        raise HTTPException(500, f"Erreur durant la simulation : {e}")

    return SimulationResponse.from_result(result)


@router.post("/simulate/batch", response_model=BatchSimulationResponse)
async def simulate_batch(req: BatchSimulationRequest):

    model = get_model(req.model)
    graph = load_graph_wrapper(req.graph_path)

    try:
        results = await run_batch_async(
            model=model,
            graph=graph,
            seed_sets=req.seed_sets,
            params=req.params,
            max_workers=req.max_workers,
            use_processes=req.use_processes,
        )
    except Exception as e:
        logger.exception("Erreur batch : %s", e)
        raise HTTPException(500, f"Erreur durant le batch : {e}")

    return BatchSimulationResponse.from_results(results)



@router.post("/metrics", response_model=MetricsResponse)
def compute_metrics(req: MetricsRequest):

    try:
        metrics = compute_all_metrics(req.simulation.to_result(), req.total_nodes)
    except Exception as e:
        logger.exception("Erreur métriques : %s", e)
        raise HTTPException(500, f"Erreur calcul métriques : {e}")

    return MetricsResponse.from_metrics(metrics)


@router.post("/graph/load", response_model=GraphLoadResponse)
def load_graph_api(req: GraphLoadRequest):

    graph = load_graph_wrapper(req.graph_path)
    nxg = graph.to_networkx()

    return GraphLoadResponse(
        num_nodes=nxg.number_of_nodes(),
        num_edges=nxg.number_of_edges(),
        directed=nxg.is_directed(),
        nodes=list(nxg.nodes())[: req.preview],  # preview
        edges=list(nxg.edges())[: req.preview],
    )
