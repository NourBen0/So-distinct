"""
schemas.py
-----------
Définition des schémas Pydantic utilisés par l’API SoDistinct.

Structure :
- SimulationRequest / SimulationResponse
- BatchSimulationRequest / BatchSimulationResponse
- SimulationResultSchema (Pydantic wrapper)
- MetricsRequest / MetricsResponse
- GraphLoadRequest / GraphLoadResponse
"""

from __future__ import annotations

from typing import Any, Dict, List, Iterable, Optional
from pydantic import BaseModel, Field

from sodistinct.core.engine import SimulationResult
from sodistinct.core.metrics import (
    CoverageMetrics,
    SpeedMetrics,
    ReachTimeMetrics,
    InfluenceMetrics,
)


# ==============================================================================
# Schemas Simulation
# ==============================================================================

class SimulationRequest(BaseModel):
    model: str = Field(..., example="ic")
    graph_path: str = Field(..., description="Chemin vers le fichier du graphe")
    seed_set: List[Any] = Field(..., example=[1, 42])
    params: Dict[str, Any] = Field(default_factory=dict)
    rng_seed: Optional[int] = Field(None)


class SimulationResultSchema(BaseModel):
    """Représente un SimulationResult en format JSON."""
    timeline: List[Dict[str, Any]]
    active_final: List[Any]
    steps: int
    runtime_ms: int
    seed_set: List[Any]
    model_name: str
    params: Dict[str, Any]
    metadata: Dict[str, Any]

    @classmethod
    def from_result(cls, r: SimulationResult):
        return cls(
            timeline=r.timeline,
            active_final=list(r.active_final),
            steps=r.steps,
            runtime_ms=r.runtime_ms,
            seed_set=list(r.seed_set),
            model_name=r.model_name,
            params=r.params,
            metadata=r.metadata,
        )

    def to_result(self) -> SimulationResult:
        """Permet de reconstruire un SimulationResult depuis Pydantic."""
        return SimulationResult(
            timeline=self.timeline,
            active_final=self.active_final,
            steps=self.steps,
            runtime_ms=self.runtime_ms,
            seed_set=self.seed_set,
            model_name=self.model_name,
            params=self.params,
            metadata=self.metadata,
        )


class SimulationResponse(BaseModel):
    result: SimulationResultSchema

    @classmethod
    def from_result(cls, r: SimulationResult):
        return cls(result=SimulationResultSchema.from_result(r))


# ==============================================================================
# Schemas Batch Simulation
# ==============================================================================

class BatchSimulationRequest(BaseModel):
    model: str = Field(..., example="ic")
    graph_path: str
    seed_sets: List[List[Any]]
    params: Dict[str, Any] = Field(default_factory=dict)
    max_workers: int = 4
    use_processes: bool = True


class BatchSimulationResponse(BaseModel):
    results: List[SimulationResultSchema]

    @classmethod
    def from_results(cls, results: List[SimulationResult]):
        return cls(
            results=[SimulationResultSchema.from_result(r) for r in results]
        )


# ==============================================================================
# Schemas Metrics
# ==============================================================================

class MetricsRequest(BaseModel):
    simulation: SimulationResultSchema
    total_nodes: int


class CoverageMetricsSchema(BaseModel):
    final_activated: int
    total_nodes: int
    coverage_ratio: float


class SpeedMetricsSchema(BaseModel):
    steps: int
    avg_new_activations_per_step: float
    max_step_activation: int
    time_to_half_coverage: Optional[int]


class ReachTimeMetricsSchema(BaseModel):
    node_reach_time: Dict[Any, int]
    avg_reach_time: Optional[float]


class InfluenceMetricsSchema(BaseModel):
    seed_set: List[Any]
    influence: Dict[Any, int]


class MetricsResponse(BaseModel):
    coverage: CoverageMetricsSchema
    speed: SpeedMetricsSchema
    reach_time: ReachTimeMetricsSchema
    influence: InfluenceMetricsSchema

    @classmethod
    def from_metrics(cls, m: Dict[str, Any]):
        return cls(
            coverage=CoverageMetricsSchema(
                final_activated=m["coverage"].final_activated,
                total_nodes=m["coverage"].total_nodes,
                coverage_ratio=m["coverage"].coverage_ratio,
            ),
            speed=SpeedMetricsSchema(
                steps=m["speed"].steps,
                avg_new_activations_per_step=m["speed"].avg_new_activations_per_step,
                max_step_activation=m["speed"].max_step_activation,
                time_to_half_coverage=m["speed"].time_to_half_coverage,
            ),
            reach_time=ReachTimeMetricsSchema(
                node_reach_time=m["reach_time"].node_reach_time,
                avg_reach_time=m["reach_time"].avg_reach_time,
            ),
            influence=InfluenceMetricsSchema(
                seed_set=m["influence"].seed_set,
                influence=m["influence"].influence,
            ),
        )


# ==============================================================================
# Schemas Graph Loading
# ==============================================================================

class GraphLoadRequest(BaseModel):
    graph_path: str
    preview: int = 20


class GraphLoadResponse(BaseModel):
    num_nodes: int
    num_edges: int
    directed: bool
    nodes: List[Any]
    edges: List[Any]
