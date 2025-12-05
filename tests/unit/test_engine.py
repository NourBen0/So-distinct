import pytest
import asyncio
import networkx as nx
import time

from sodistinct.core.graph_wrapper import GraphWrapper
from sodistinct.core.models import SIModel, ICModel, SIRModel
from sodistinct.core.engine import run_simulation, run_simulation_async, SimulationResult


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def build_path_graph():
    g = nx.path_graph(4)  # 0–1–2–3
    return GraphWrapper(g)


# ------------------------------------------------------------------------------
# Test SimulationResult structure
# ------------------------------------------------------------------------------

def test_simulation_result_structure():
    r = SimulationResult(
        timeline=[],
        active_final=[1, 2],
        steps=3,
        runtime_ms=123,
        seed_set=[0],
        model_name="SIModel",
        params={"beta": 1},
        metadata={"info": 1},
    )

    assert r.model_name == "SIModel"
    assert r.steps == 3
    assert r.active_final == [1, 2]
    assert isinstance(r.runtime_ms, int)


# ------------------------------------------------------------------------------
# SI Model tests
# ------------------------------------------------------------------------------

def test_run_simulation_si_basic():
    g = build_path_graph()
    m = SIModel()

    result = run_simulation(
        model=m,
        graph=g,
        seed_set=[0],
        params={"transmission_rate": 1.0},
        rng_seed=1,
    )

    assert isinstance(result, SimulationResult)
    assert result.seed_set == [0]
    assert result.model_name == "SIModel"

    # Diffusion complète sur une chaîne => tous les nœuds activés
    assert set(result.active_final) == {0, 1, 2, 3}

    # Timeline cohérente
    assert result.timeline[0]["step"] == 0
    assert "active" in result.timeline[0]
    assert "new_active" in result.timeline[0]


def test_run_simulation_si_finishes():
    g = build_path_graph()
    m = SIModel()

    result = run_simulation(
        model=m,
        graph=g,
        seed_set=[0],
        params={"transmission_rate": 1.0},
    )
    assert result.steps > 0
    assert len(result.active_final) == 4


# ------------------------------------------------------------------------------
# IC Model tests
# ------------------------------------------------------------------------------

def test_run_simulation_ic_deterministic():
    g = build_path_graph()
    m = ICModel()

    # rng_seed => diffusion déterministe
    result = run_simulation(
        model=m,
        graph=g,
        seed_set=[0],
        params={"p": 1.0},
        rng_seed=0,
    )

    assert set(result.active_final) == {0, 1, 2, 3}
    assert result.steps >= 1


def test_ic_no_spread_when_p_zero():
    g = build_path_graph()
    m = ICModel()

    result = run_simulation(
        model=m,
        graph=g,
        seed_set=[0],
        params={"p": 0.0},
        rng_seed=0,
    )

    # Aucun voisin n'est activé
    assert set(result.active_final) == {0}
    assert result.steps == 1  # step 0 initial + step 1 sans propagation


# ------------------------------------------------------------------------------
# SIR Model tests
# ------------------------------------------------------------------------------

def test_sir_recovery():
    g = build_path_graph()
    m = SIRModel()

    result = run_simulation(
        model=m,
        graph=g,
        seed_set=[0],
        params={"transmission_rate": 1.0, "recovery_rate": 1.0},
        rng_seed=0,
    )

    # Le patient 0 doit être dans removed
    last_state = result.timeline[-1]
    assert 0 in last_state["removed"]


# ------------------------------------------------------------------------------
# Timeline & structure validation
# ------------------------------------------------------------------------------

def test_timeline_structure_consistency():
    g = build_path_graph()
    m = SIModel()

    r = run_simulation(
        model=m,
        graph=g,
        seed_set=[0],
        params={"transmission_rate": 1.0},
    )

    for entry in r.timeline:
        assert "step" in entry
        assert "active" in entry
        assert "new_active" in entry
        assert "removed" in entry


# ------------------------------------------------------------------------------
# Runtime consistency
# ------------------------------------------------------------------------------

def test_runtime_is_positive():
    g = build_path_graph()
    m = SIModel()
    r = run_simulation(
        model=m,
        graph=g,
        seed_set=[0],
        params={"transmission_rate": 1.0},
    )
    assert r.runtime_ms >= 0


# ------------------------------------------------------------------------------
# Async version
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_simulation_async():
    g = build_path_graph()
    m = SIModel()

    r = await run_simulation_async(
        model=m,
        graph=g,
        seed_set=[0],
        params={"transmission_rate": 1.0},
        rng_seed=0,
    )

    assert isinstance(r, SimulationResult)
    assert set(r.active_final) == {0, 1, 2, 3}


# ------------------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------------------

def test_simulation_with_empty_seed():
    g = build_path_graph()
    m = SIModel()

    r = run_simulation(
        model=m,
        graph=g,
        seed_set=[],
        params={"transmission_rate": 1.0},
    )

    # Aucun nœud ne se propage
    assert r.active_final == []
    assert r.steps == 0


def test_simulation_zero_step_if_state_finished_immediately():
    class DummyModel:
        def initialize(self, g, seeds, params):
            from sodistinct.core.models import DiffusionState
            return DiffusionState(active=set(), new_active=set(), finished=True)

        def step(self, g, state, params):
            return state

        def is_finished(self, state):
            return True

    g = build_path_graph()
    m = DummyModel()

    r = run_simulation(
        model=m,
        graph=g,
        seed_set=[],
        params={},
    )

    assert r.steps == 0
    assert r.active_final == []
    assert len(r.timeline) == 1  # état initial seulement
