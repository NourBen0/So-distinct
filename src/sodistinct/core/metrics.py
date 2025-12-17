from __future__ import annotations

from typing import Dict, Any, Iterable, Optional
from dataclasses import dataclass
from statistics import mean

import networkx as nx

from sodistinct.core.engine import SimulationResult
from sodistinct.core.graph_wrapper import GraphWrapper

@dataclass
class CoverageMetrics:
    final_activated: int
    total_nodes: int
    coverage_ratio: float


@dataclass
class SpeedMetrics:
    steps: int
    avg_new_activations_per_step: float
    max_step_activation: int
    time_to_half_coverage: Optional[int]


@dataclass
class ReachTimeMetrics:
    node_reach_time: Dict[Any, int]
    avg_reach_time: Optional[float]


@dataclass
class InfluenceMetrics:
    seed_set: Iterable[Any]
    influence: Dict[Any, int]


def compute_coverage(result: SimulationResult, total_nodes: int) -> CoverageMetrics:
    final_activated = len(result.active_final)
    ratio = final_activated / total_nodes if total_nodes > 0 else 0.0

    return CoverageMetrics(
        final_activated=final_activated,
        total_nodes=total_nodes,
        coverage_ratio=ratio,
    )


def compute_speed(result: SimulationResult) -> SpeedMetrics:
    timeline = result.timeline
    new_activ_list = [len(step["new_active"]) for step in timeline]

    avg_new = mean(new_activ_list[1:]) if len(new_activ_list) > 1 else 0.0
    max_step = max(new_activ_list) if new_activ_list else 0

    half = len(result.active_final) / 2
    cumulative = 0
    half_time = None

    for step in timeline:
        cumulative += len(step["new_active"])
        if cumulative >= half and half_time is None:
            half_time = step["step"]
            break

    return SpeedMetrics(
        steps=result.steps,
        avg_new_activations_per_step=avg_new,
        max_step_activation=max_step,
        time_to_half_coverage=half_time,
    )


def compute_reach_time(result: SimulationResult) -> ReachTimeMetrics:
    reach = {}
    for step in result.timeline:
        t = step["step"]
        for node in step["new_active"]:
            if node not in reach:
                reach[node] = t

    avg_reach = mean(reach.values()) if reach else None

    return ReachTimeMetrics(
        node_reach_time=reach,
        avg_reach_time=avg_reach,
    )


def compute_influence(result: SimulationResult) -> InfluenceMetrics:
    influence = {}
    seed_set = result.seed_set

    for seed in seed_set:
        count = 0
        for step in result.timeline:
            for node in step["new_active"]:
                if node != seed:
                    count += 1
        influence[seed] = count

    return InfluenceMetrics(seed_set=seed_set, influence=influence)


def compute_all_metrics(result: SimulationResult, total_nodes: int) -> Dict[str, Any]:
    return {
        "coverage": compute_coverage(result, total_nodes),
        "speed": compute_speed(result),
        "reach_time": compute_reach_time(result),
        "influence": compute_influence(result),
    }



def coverage(graph: GraphWrapper, active: Iterable[Any]) -> float:
    total = graph.number_of_nodes()
    if total == 0:
        return 0.0
    return len(active) / total


def cascade_size(active: Iterable[Any]) -> int:
    return len(active)


def reach_time(timeline, node):
    for step in timeline:
        if node in step["new_active"]:
            return step["step"]
    return -1


def degree_centrality(graph: GraphWrapper):
    g = graph.unwrap()
    if g.number_of_nodes() == 0:
        return {}
    return nx.degree_centrality(g)


def closeness_centrality(graph: GraphWrapper):
    g = graph.unwrap()
    if g.number_of_nodes() == 0:
        return {}
    return nx.closeness_centrality(g)


def betweenness_centrality(graph: GraphWrapper):
    g = graph.unwrap()
    if g.number_of_nodes() == 0:
        return {}
    return nx.betweenness_centrality(g)


def average_path_length(graph: GraphWrapper):
    g = graph.unwrap()
    if g.number_of_nodes() == 0:
        return 0.0
    try:
        return nx.average_shortest_path_length(g)
    except nx.NetworkXError:
        return 0.0  # graphe non connexe
