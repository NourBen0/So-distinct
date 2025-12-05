import pytest
import networkx as nx

from sodistinct.core.graph_wrapper import GraphWrapper
from sodistinct.core.metrics import (
    coverage,
    cascade_size,
    reach_time,
    degree_centrality,
    closeness_centrality,
    betweenness_centrality,
    average_path_length,
)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def simple_graph():
    # Graphe 0–1–2–3 (chaîne)
    g = nx.path_graph(4)
    return GraphWrapper(g)


# ------------------------------------------------------------------------------
# Coverage
# ------------------------------------------------------------------------------

def test_coverage_basic():
    g = simple_graph()
    active = {0, 1}
    cov = coverage(g, active)
    assert cov == 2 / 4  # 50%


def test_coverage_empty_active():
    g = simple_graph()
    assert coverage(g, set()) == 0.0


def test_coverage_full():
    g = simple_graph()
    assert coverage(g, {0, 1, 2, 3}) == 1.0


# ------------------------------------------------------------------------------
# Cascade size
# ------------------------------------------------------------------------------

def test_cascade_size():
    assert cascade_size({1, 2, 3}) == 3
    assert cascade_size(set()) == 0


# ------------------------------------------------------------------------------
# Reach time
# ------------------------------------------------------------------------------

def test_reach_time_basic():
    timeline = [
        {"step": 0, "active": [0], "new_active": [0]},
        {"step": 1, "active": [0, 1], "new_active": [1]},
        {"step": 2, "active": [0, 1, 2], "new_active": [2]},
    ]
    assert reach_time(timeline, 2) == 2
    assert reach_time(timeline, 1) == 1


def test_reach_time_never_reached():
    timeline = [
        {"step": 0, "active": [0], "new_active": [0]},
        {"step": 1, "active": [0, 1], "new_active": [1]},
    ]
    assert reach_time(timeline, 5) == -1


# ------------------------------------------------------------------------------
# Centralities
# ------------------------------------------------------------------------------

def test_degree_centrality():
    g = simple_graph()
    dc = degree_centrality(g)

    # Graphe 0–1–2–3
    # degré : 0=1, 1=2, 2=2, 3=1
    assert dc[1] == dc[2]  # centres
    assert dc[0] == dc[3]  # extrémités


def test_closeness_centrality():
    g = simple_graph()
    cc = closeness_centrality(g)

    # Noeud 1 ou 2 est le plus central
    assert cc[1] > cc[0]
    assert cc[2] > cc[3]


def test_betweenness_centrality():
    g = simple_graph()
    bc = betweenness_centrality(g)

    # 1 et 2 sont les plus importants dans un chemin
    assert bc[1] == bc[2]
    assert bc[1] > bc[0]


# ------------------------------------------------------------------------------
# Average path length
# ------------------------------------------------------------------------------

def test_average_path_length():
    g = simple_graph()
    apl = average_path_length(g)
    # Calcul NX pour vérifier
    nx_apl = nx.average_shortest_path_length(g.unwrap())
    assert pytest.approx(apl, rel=1e-6) == nx_apl


def test_average_path_length_disconnected():
    g = GraphWrapper(nx.Graph())  # graph vide
    assert average_path_length(g) == 0.0


# ------------------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------------------

def test_metrics_on_empty_graph():
    g = GraphWrapper(nx.Graph())

    assert coverage(g, set()) == 0.0
    assert degree_centrality(g) == {}
    assert closeness_centrality(g) == {}
    assert betweenness_centrality(g) == {}
    assert average_path_length(g) == 0.0
