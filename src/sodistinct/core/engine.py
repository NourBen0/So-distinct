
from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, List

from sodistinct.core.models import DiffusionModel, DiffusionState
from sodistinct.core.graph_wrapper import GraphWrapper


@dataclass
class SimulationResult:

    timeline: List[Dict[str, Any]] = field(default_factory=list)
    active_final: Iterable[Any] = field(default_factory=list)
    steps: int = 0
    runtime_ms: int = 0
    seed_set: Iterable[Any] = field(default_factory=list)
    model_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_simulation(
    model: DiffusionModel,
    graph: GraphWrapper,
    seed_set: Iterable[Any],
    params: Dict[str, Any],
    rng_seed: Optional[int] = None,
) -> SimulationResult:

    
    t_start = time.time()

    
    if rng_seed is not None:
        random.seed(rng_seed)

    
    state: DiffusionState = model.initialize(graph, seed_set, params)

   
    timeline = []
    timeline.append(
        {
            "step": 0,
            "active": list(state.active),
            "new_active": list(state.new_active),
            "removed": list(state.removed),
        }
    )

    
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




async def run_simulation_async(
    model: DiffusionModel,
    graph: GraphWrapper,
    seed_set: Iterable[Any],
    params: Dict[str, Any],
    rng_seed: Optional[int] = None,
) -> SimulationResult:

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
