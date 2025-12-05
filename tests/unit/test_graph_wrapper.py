import networkx as nx
import pytest
from sodistinct.core.graph_wrapper import GraphWrapper


# ------------------------------------------------------------------------------
# Construction / validation
# ------------------------------------------------------------------------------

def test_init_accepts_networkx():
    g = nx.Graph()
    wrapper = GraphWrapper(g)
    assert isinstance(wrapper, GraphWrapper)


def test_init_rejects_invalid():
    with pytest.raises(TypeError):
        GraphWrapper("not a graph")


def test_from_networkx():
    g = nx.Graph()
    g.add_edge(1, 2)
    w = GraphWrapper.from_networkx(g)
    assert isinstance(w, GraphWrapper)
    assert w.number_of_edges() == 1


# ------------------------------------------------------------------------------
# Infos de base
# ------------------------------------------------------------------------------

def test_basic_info():
    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3)])
    w = GraphWrapper(g)

    assert w.number_of_nodes() == 3
    assert w.number_of_edges() == 2
    assert set(w.nodes()) == {1, 2, 3}
    assert set(w.edges()) == {(1, 2), (2, 3)}


# ------------------------------------------------------------------------------
# Voisins / degré
# ------------------------------------------------------------------------------

def test_neighbors_and_degree():
    g = nx.Graph()
    g.add_edges_from([(1, 2), (1, 3)])
    w = GraphWrapper(g)

    assert set(w.neighbors(1)) == {2, 3}
    assert w.degree(1) == 2


# ------------------------------------------------------------------------------
# Attributs nœuds / arêtes
# ------------------------------------------------------------------------------

def test_node_attributes():
    g = nx.Graph()
    g.add_node(1)
    w = GraphWrapper(g)

    w.set_node_attr(1, "color", "red")
    assert w.get_node_attr(1, "color") == "red"
    assert w.get_node_attr(1, "missing", "default") == "default"


def test_edge_attributes():
    g = nx.Graph()
    g.add_edge(1, 2)
    w = GraphWrapper(g)

    w.set_edge_attr(1, 2, "weight", 4.5)
    assert w.get_edge_attr(1, 2, "weight") == 4.5
    assert w.get_edge_attr(1, 2, "missing", "default") == "default"


# ------------------------------------------------------------------------------
# Ajout noeuds / arêtes
# ------------------------------------------------------------------------------

def test_add_node_and_edge():
    g = nx.Graph()
    w = GraphWrapper(g)

    w.add_node(5)
    w.add_edge(1, 2)

    assert w.has_node(5)
    assert w.has_edge(1, 2)


# ------------------------------------------------------------------------------
# Copies / sous-graphes
# ------------------------------------------------------------------------------

def test_copy_graph():
    g = nx.path_graph(4)
    w = GraphWrapper(g)

    w2 = w.copy()
    assert w2.number_of_nodes() == 4
    assert w2.number_of_edges() == 3

    # modification de la copie → original inchangé
    w2.add_node("X")
    assert w.number_of_nodes() == 4
    assert w2.number_of_nodes() == 5


def test_subgraph():
    g = nx.path_graph(5)
    w = GraphWrapper(g)

    w2 = w.subgraph([1, 2, 3])
    assert w2.number_of_nodes() == 3
    assert w2.number_of_edges() == 2
    assert set(w2.nodes()) == {1, 2, 3}


# ------------------------------------------------------------------------------
# Conversion / export
# ------------------------------------------------------------------------------

def test_to_networkx():
    g = nx.complete_graph(3)
    w = GraphWrapper(g)
    ng = w.to_networkx()

    assert isinstance(ng, nx.Graph)
    assert ng.number_of_nodes() == 3


def test_to_adjlist():
    g = nx.path_graph(4)   # 0-1-2-3
    w = GraphWrapper(g)

    adj = w.to_adjlist()
    assert adj[1] == [0, 2]


# ------------------------------------------------------------------------------
# Weighted neighbors
# ------------------------------------------------------------------------------

def test_weighted_neighbors():
    g = nx.Graph()
    g.add_edge(1, 2, weight=0.5)
    g.add_edge(1, 3)  # default weight = 1.0
    w = GraphWrapper(g)

    neigh = dict(w.weighted_neighbors(1))
    assert neigh[2] == 0.5
    assert neigh[3] == 1.0


# ------------------------------------------------------------------------------
# Repr
# ------------------------------------------------------------------------------

def test_repr():
    g = nx.path_graph(3)
    w = GraphWrapper(g)

    r = repr(w)
    assert "nodes=3" in r
    assert "edges=2" in r
