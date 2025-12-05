"""
graph_wrapper.py
----------------
Abstraction unifiée pour les graphes utilisés dans SoDistinct.

Objectifs :
- Fournir une API commune pour manipuler les graphes, indépendamment du backend.
- Faciliter l’intégration future d’autres bibliothèques (igraph, GraphTool).
- Encapsuler NetworkX comme backend principal.
- Exposer uniquement les opérations nécessaires aux modèles de diffusion
  (accès aux voisins, gestion des attributs, etc.)

Usage :
    from sodistinct.core.graph_wrapper import GraphWrapper
    g = GraphWrapper.from_networkx(nx_graph)
    nodes = g.nodes()
    neighbors = g.neighbors("A")
"""

from __future__ import annotations

import networkx as nx
from typing import Any, Iterable, List, Dict, Optional


class GraphWrapper:
    """
    Wrapper pour abstraire le backend des graphes.

    Backend par défaut : NetworkX.
    """

    # ----------------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------------

    def __init__(self, graph: nx.Graph):
        if not isinstance(graph, nx.Graph):
            raise TypeError("GraphWrapper nécessite un objet networkx.Graph")
        self._graph = graph

    @classmethod
    def from_networkx(cls, g: nx.Graph) -> "GraphWrapper":
        return cls(g)

    def unwrap(self) -> nx.Graph:
        """Retourne l'objet NetworkX interne."""
        return self._graph

    # ----------------------------------------------------------------------
    # Informations de base
    # ----------------------------------------------------------------------

    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    def nodes(self) -> Iterable[Any]:
        return self._graph.nodes()

    def edges(self) -> Iterable[Any]:
        return self._graph.edges()

    # ----------------------------------------------------------------------
    # Accès aux voisins
    # ----------------------------------------------------------------------

    def neighbors(self, node: Any) -> Iterable[Any]:
        """Retourne les voisins d’un nœud."""
        return self._graph.neighbors(node)

    # ----------------------------------------------------------------------
    # Attributs nœuds / arêtes
    # ----------------------------------------------------------------------

    def get_node_attr(self, node: Any, key: str, default: Any = None) -> Any:
        return self._graph.nodes[node].get(key, default)

    def set_node_attr(self, node: Any, key: str, value: Any):
        self._graph.nodes[node][key] = value

    def get_edge_attr(self, u: Any, v: Any, key: str, default: Any = None) -> Any:
        return self._graph.edges[u, v].get(key, default)

    def set_edge_attr(self, u: Any, v: Any, key: str, value: Any):
        self._graph.edges[u, v][key] = value

    # ----------------------------------------------------------------------
    # Actions basiques
    # ----------------------------------------------------------------------

    def add_node(self, node: Any, **attrs):
        self._graph.add_node(node, **attrs)

    def add_edge(self, u: Any, v: Any, **attrs):
        self._graph.add_edge(u, v, **attrs)

    def has_node(self, node: Any) -> bool:
        return self._graph.has_node(node)

    def has_edge(self, u: Any, v: Any) -> bool:
        return self._graph.has_edge(u, v)

    # ----------------------------------------------------------------------
    # Copies / sous-graphes
    # ----------------------------------------------------------------------

    def copy(self) -> "GraphWrapper":
        """Retourne une copie profonde du graphe."""
        return GraphWrapper(self._graph.copy())

    def subgraph(self, nodes: Iterable[Any]) -> "GraphWrapper":
        """Sous-graphe induit."""
        return GraphWrapper(self._graph.subgraph(nodes).copy())

    # ----------------------------------------------------------------------
    # Conversion / export
    # ----------------------------------------------------------------------

    def to_networkx(self) -> nx.Graph:
        """Alias de unwrap."""
        return self._graph

    def to_adjlist(self) -> Dict[Any, List[Any]]:
        """Retourne un dictionnaire node -> liste des voisins."""
        return {n: list(self.neighbors(n)) for n in self.nodes()}

    # ----------------------------------------------------------------------
    # Fonctions avancées utiles dans la diffusion
    # ----------------------------------------------------------------------

    def degree(self, node: Any) -> int:
        """Degré d’un nœud."""
        return self._graph.degree(node)

    def weighted_neighbors(self, node: Any, weight_key: str = "weight") -> List[Any]:
        """
        Retourne les voisins avec poids d’arête, utilisé dans certains modèles
        (ex: Linear Threshold pondéré).
        """
        res = []
        for v in self.neighbors(node):
            w = self.get_edge_attr(node, v, weight_key, 1.0)
            res.append((v, w))
        return res

    # ----------------------------------------------------------------------
    # Représentation
    # ----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GraphWrapper(nodes={self.number_of_nodes()}, "
            f"edges={self.number_of_edges()})"
        )
