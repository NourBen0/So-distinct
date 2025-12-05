"""
plots.py
---------
Fonctions de visualisation pour SoDistinct.

Basé sur matplotlib (backend standard), sans dépendances lourdes.
Fourni :
    - Courbe d’activation cumulée (timeline)
    - Nouvelles activations par step
    - Histogramme des reach-times
    - Comparaison multi-simulations
    - Visualisation statique du réseau (NetworkX)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

from sodistinct.core.engine import SimulationResult
from sodistinct.core.metrics import (
    compute_reach_time,
)
from sodistinct.core.graph_wrapper import GraphWrapper


# ==============================================================================
# Helpers internes
# ==============================================================================

def _extract_timeline_counts(result: SimulationResult):
    """Retourne:
        - steps[]
        - cumulative_activations[]
        - new_activations[]
    """
    steps = []
    cumulative = []
    new = []

    cum = 0
    for step in result.timeline:
        s = step["step"]
        na = len(step["new_active"])
        cum += na

        steps.append(s)
        cumulative.append(cum)
        new.append(na)

    return steps, cumulative, new


# ==============================================================================
# Courbe d’activation cumulée
# ==============================================================================

def plot_cumulative_activation(result: SimulationResult, show: bool = True):
    """
    Trace la courbe cumulée des activations (coverage over time).
    """
    steps, cumulative, _ = _extract_timeline_counts(result)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, cumulative)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Activated Nodes")
    plt.title(f"Cumulative Activation Curve — {result.model_name}")

    if show:
        plt.show()


# ==============================================================================
# Nouvelles activations par step
# ==============================================================================

def plot_new_activations(result: SimulationResult, show: bool = True):
    """
    Trace les nouvelles activations à chaque step.
    """
    steps, _, new = _extract_timeline_counts(result)

    plt.figure(figsize=(8, 4))
    plt.bar(steps, new)
    plt.xlabel("Step")
    plt.ylabel("New Activations")
    plt.title(f"New Activations per Step — {result.model_name}")

    if show:
        plt.show()


# ==============================================================================
# Histogramme des reach-times
# ==============================================================================

def plot_reach_time_histogram(result: SimulationResult, bins: int = 10, show: bool = True):
    """
    Histogramme des reach-times par nœud.
    """
    rt = compute_reach_time(result)
    times = list(rt.node_reach_time.values())

    plt.figure(figsize=(8, 4))
    plt.hist(times, bins=bins)
    plt.xlabel("Reach Time")
    plt.ylabel("Number of Nodes")
    plt.title("Reach-Time Distribution")

    if show:
        plt.show()


# ==============================================================================
# Comparaison multi-runs (cumulative)
# ==============================================================================

def plot_compare_cumulative(
    results: List[SimulationResult],
    labels: Optional[List[str]] = None,
    show: bool = True,
):
    """
    Compare plusieurs résultats de simulation (courbes cumulées).
    """
    plt.figure(figsize=(8, 5))

    for i, r in enumerate(results):
        steps, cum, _ = _extract_timeline_counts(r)
        label = labels[i] if labels else f"Run {i+1}"
        plt.plot(steps, cum, label=label)

    plt.xlabel("Step")
    plt.ylabel("Cumulative Activated")
    plt.title("Comparison of Cumulative Activation Across Runs")
    plt.legend()

    if show:
        plt.show()


# ==============================================================================
# Visualisation statique du réseau (NetworkX)
# ==============================================================================

def plot_graph(graph: GraphWrapper, show: bool = True, with_labels: bool = False):
    """
    Visualisation statique simple du réseau.
    """
    import networkx as nx

    g = graph.to_networkx()

    plt.figure(figsize=(7, 7))
    pos = nx.spring_layout(g)
    nx.draw(g, pos, with_labels=with_labels, node_size=300)

    plt.title("Graph Visualization")

    if show:
        plt.show()
