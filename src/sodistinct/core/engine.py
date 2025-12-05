"""
engine.py
---------
Moteur principal d'exécution pour un run de simulation dans SoDistinct.

Objectifs :
- Orchestrer un run de diffusion avec n'importe quel modèle (IC, SI, SIR, LT…)
- Renvoyer un SimulationResult complet, normalisé et exploitable
- Supporter :
    * mode synchrone (pour multiprocessing)
    * mode asynchrone (pour asyncio / FastAPI)
- Gestion des seeds, du temps, et instrumentation basique (timing)

Ce module ne s'occupe que :
- De l'exécution d’UNE simulation
- De la boucle step-by-step

Les orchestrateurs (local_async, parallel, ray_backend, dask_backend)
gèrent les multiples runs ou clusterisation.
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, List

from sodistinct.core.models import DiffusionModel, DiffusionState
from sodistinct.core.graph_wrapper import GraphWrapper


# ============================================================================
# Structure de sortie (résultat)
# ============================================================================

@dataclass
class SimulationResult:
    """
    Résultat complet d’un run de diffusion.

    Attributes :
        timeline           : liste d'états (activations par step)
        active_final       : ensemble final des nœuds activés
        steps              : nombre total de steps
        runtime_ms         : temps d'exécution du run
        seed_set           : liste des nœuds initiaux
        model_name         : nom du modèle utilisé
        params             : paramètres fournis au modèle
        metadata           : informations additionnelles
    """

    timeline: List[Dict[str, Any]] = field(default_factory=list)
    active_final: Iterable[Any] = field(default_factory=list)
    steps: int = 0
    runtime_ms: int = 0
    seed_set: Iterable[Any] = field(default_factory=list)
    model_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Fonction principale : exécuter un run
# ============================================================================

def run_simulation(
    model: DiffusionModel,
    graph: GraphWrapper,
    seed_set: Iterable[Any],
    params: Dict[str, Any],
    rng_seed: Optional[int] = None,
) -> SimulationResult:
    """
    Exécute UNE simulation séquentielle d'un modèle de diffusion.

    Paramètres :
        model       : instance d’un modèle (ICModel(), SIModel(), ...)
        graph       : GraphWrapper
        seed_set    : liste / ensemble de nœuds initiaux
        params      : dict de paramètres (p, threshold, rates…)
        rng_seed    : seed optionnel pour reproductibilité locale

    Retour :
        SimulationResult
    """

    t_start = time.time()

    # Reproductibilité : seed RNG local
    if rng_seed is not None:
        random.seed(rng_seed)

    # Initialisation du modèle
    state: DiffusionState = model.initialize(graph, seed_set, params)

    # Timeline stocke les activations à chaque step : utile pour LT/SIR/SI
    timeline = []
    timeline.append(
        {
            "step": 0,
            "active": list(state.active),
            "new_active": list(state.new_active),
            "removed": list(state.removed),
        }
    )

    # Boucle de diffusion
    while not model.is_finished(state):
        # Step
        state = model.step(graph, state, params)

        timeline.append(
            {
                "step": state.time,
                "active": list(state.active),
                "new_active": list(state.new_active),
                "removed": list(state.removed),
            }
        )

    runtime_ms = int((time.time() - t_start) * 1000)

    # Résultat final
    return SimulationResult(
        timeline=timeline,
        active_final=list(state.active),
        steps=state.time,
        runtime_ms=runtime_ms,
        seed_set=list(seed_set),
        model_name=model.__class__.__name__,
        params=params.copy(),
        metadata=state.metadata.copy(),
    )


# ============================================================================
# Version asynchrone (compatible asyncio)
# ============================================================================

async def run_simulation_async(
    model: DiffusionModel,
    graph: GraphWrapper,
    seed_set: Iterable[Any],
    params: Dict[str, Any],
    rng_seed: Optional[int] = None,
) -> SimulationResult:
    """
    Version asynchrone d’une simulation.
    Internally, steps remain synchronous, but the wrapper allows integration
    with orchestrators asyncio (FastAPI, dashboards…).
    """

    # Pour ne pas bloquer l'event loop pour de grosses simulations :
    # on peut exécuter la version synchrone dans un thread.
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_simulation(
            model=model,
            graph=graph,
            seed_set=seed_set,
            params=params,
            rng_seed=rng_seed,
        ),
    )
