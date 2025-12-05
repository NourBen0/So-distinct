"""
models.py
---------
Implémentation des modèles de diffusion utilisés dans SoDistinct.

Contient :
- Une interface abstraite DiffusionModel
- Les modèles standards :
    * SI  (Susceptible → Infected)
    * SIR (Susceptible → Infected → Recovered)
    * Independent Cascade (IC)
    * Linear Threshold (LT)

Tous les modèles doivent :
- Implémenter `.initialize(graph, seed_set, params)`
- Implémenter `.step(state)`
- Optionnel : `.is_finished(state)`
- Fournir un état contenant :
      .active (nœuds "infectés"/"propagés")
      .new_active (nœuds activés au dernier step)
      .time (step courant)
      .metadata (infos utiles)

L’état final est consommé par engine.py pour produire SimulationResult.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Optional, Iterable
from abc import ABC, abstractmethod
import random

from sodistinct.core.graph_wrapper import GraphWrapper


# ============================================================================
# Helper : structure d'état interne des modèles
# ============================================================================

class DiffusionState:
    """
    Contient l'état de la diffusion pour un modèle :
    - active      : tous les nœuds déjà activés (infectés, informés, etc.)
    - new_active  : les nœuds activés au dernier step
    - removed     : utilisé par SIR (noeuds guéris/inactifs)
    - time        : step de la simulation
    - finished    : si la diffusion est terminée
    - metadata    : informations diverses
    """

    def __init__(
        self,
        active: Set[Any],
        new_active: Set[Any],
        removed: Optional[Set[Any]] = None,
        time: int = 0,
        finished: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.active = active
        self.new_active = new_active
        self.removed = removed if removed is not None else set()
        self.time = time
        self.finished = finished
        self.metadata = metadata or {}

    def is_finished(self) -> bool:
        return self.finished

    def __repr__(self):
        return (
            f"DiffusionState(time={self.time}, active={len(self.active)}, "
            f"new_active={len(self.new_active)}, removed={len(self.removed)})"
        )


# ============================================================================
# Interface abstraite
# ============================================================================

class DiffusionModel(ABC):
    """Interface générique pour un modèle de diffusion."""

    @abstractmethod
    def initialize(
        self, graph: GraphWrapper, seed_set: Iterable[Any], params: Dict[str, Any]
    ) -> DiffusionState:
        pass

    @abstractmethod
    def step(
        self, graph: GraphWrapper, state: DiffusionState, params: Dict[str, Any]
    ) -> DiffusionState:
        pass

    def is_finished(self, state: DiffusionState) -> bool:
        return state.is_finished()


# ============================================================================
# SI Model
# ============================================================================

class SIModel(DiffusionModel):
    """
    Modèle SI :
        S (Susceptible) -> I (Infected)
    Paramètre principal :
        - transmission_rate (float)
    """

    def initialize(self, graph, seed_set, params):
        active = set(seed_set)
        return DiffusionState(active=active, new_active=set(active))

    def step(self, graph, state, params):
        if not state.new_active:
            state.finished = True
            return state

        beta = params.get("transmission_rate", 0.1)

        new_act = set()
        for u in state.new_active:
            for v in graph.neighbors(u):
                if v not in state.active:
                    if random.random() < beta:
                        new_act.add(v)

        state.active |= new_act
        state.new_active = new_act
        state.time += 1

        if not new_act:
            state.finished = True

        return state


# ============================================================================
# SIR Model
# ============================================================================

class SIRModel(DiffusionModel):
    """
    Modèle SIR :
        S (Susceptible) → I (Infected) → R (Recovered)

    Paramètres :
        - transmission_rate
        - recovery_rate
    """

    def initialize(self, graph, seed_set, params):
        active = set(seed_set)
        return DiffusionState(active=active, new_active=set(active), removed=set())

    def step(self, graph, state, params):
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.05)

        if not state.new_active:
            state.finished = True
            return state

        new_infected = set()
        new_removed = set()

        # Propagation
        for u in state.new_active:
            for v in graph.neighbors(u):
                if v not in state.active and v not in state.removed:
                    if random.random() < beta:
                        new_infected.add(v)

        # Récupération
        for u in state.new_active:
            if random.random() < gamma:
                new_removed.add(u)

        state.active |= new_infected
        state.removed |= new_removed

        # Les nouveaux infectés deviennent les "new_active" du step suivant
        state.new_active = new_infected
        state.time += 1

        if not new_infected:
            state.finished = True

        return state


# ============================================================================
# Independent Cascade (IC)
# ============================================================================

class ICModel(DiffusionModel):
    """
    Independent Cascade (IC)
    Paramètre principal :
        - p : probabilité qu’un nœud active un voisin
    """

    def initialize(self, graph, seed_set, params):
        active = set(seed_set)
        return DiffusionState(active=active, new_active=set(active))

    def step(self, graph, state, params):
        if not state.new_active:
            state.finished = True
            return state

        p = params.get("p", 0.03)
        new_act = set()

        for u in state.new_active:
            for v in graph.neighbors(u):
                if v not in state.active:
                    if random.random() < p:
                        new_act.add(v)

        state.active |= new_act
        state.new_active = new_act
        state.time += 1

        if not new_act:
            state.finished = True

        return state


# ============================================================================
# Linear Threshold (LT)
# ============================================================================

class LTModel(DiffusionModel):
    """
    Linear Threshold Model (LT)
    Paramètres :
        - threshold : valeur par défaut si aucun poids
        - thresholds : dict optionnel specific node->threshold
        - weight_key : clé utilisée pour récupérer les poids sur les arêtes
    """

    def initialize(self, graph, seed_set, params):
        active = set(seed_set)
        metadata = {}

        # Seuils de chaque nœud
        default_th = params.get("threshold", 0.2)
        node_thresholds = params.get("thresholds", {})
        metadata["thresholds"] = {
            n: node_thresholds.get(n, default_th) for n in graph.nodes()
        }

        return DiffusionState(
            active=active,
            new_active=set(active),
            metadata=metadata,
        )

    def step(self, graph, state, params):
        if not state.new_active:
            state.finished = True
            return state

        weight_key = params.get("weight_key", "weight")
        thresholds = state.metadata["thresholds"]

        new_act = set()

        for node in graph.nodes():
            if node in state.active:
                continue

            # influence totale reçue
            influence = 0.0
            for (nbr, w) in graph.weighted_neighbors(node, weight_key):
                if nbr in state.active:
                    influence += w

            if influence >= thresholds[node]:
                new_act.add(node)

        state.active |= new_act
        state.new_active = new_act
        state.time += 1

        if not new_act:
            state.finished = True

        return state
