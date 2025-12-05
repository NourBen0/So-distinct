import networkx as nx
import pytest
import random

from sodistinct.core.graph_wrapper import GraphWrapper
from sodistinct.core.models import (
    DiffusionState,
    SIModel,
    SIRModel,
    ICModel,
    LTModel,
)


# ------------------------------------------------------------------------------
# DiffusionState Tests
# ------------------------------------------------------------------------------

def test_diffusion_state_basic():
    s = DiffusionState(active={1, 2}, new_active={2}, removed=set(), time=3)
    assert s.time == 3
    assert s.active == {1, 2}
    assert s.new_active == {2}
    assert s.removed == set()
    assert s.is_finished() is False

def test_diffusion_state_finished_flag():
    s = DiffusionState(active=set(), new_active=set(), finished=True)
    assert s.is_finished() is True


# ------------------------------------------------------------------------------
# SI Model
# ------------------------------------------------------------------------------

def test_si_initialize():
    g = GraphWrapper(nx.Graph())
    model = SIModel()

    state = model.initialize(g, seed_set=[1, 2], params={})
    assert state.active == {1, 2}
    assert state.new_active == {1, 2}

def test_si_step_full_spread():
    g = nx.Graph()
    g.add_edges_from([(0,1), (1,2)])
    w = GraphWrapper(g)

    model = SIModel()
    random.seed(1)

    state = model.initialize(w, seed_set=[0], params={"transmission_rate": 1.0})
    state = model.step(w, state, params={"transmission_rate": 1.0})

    assert state.active == {0,1}
    assert state.time == 1


# ------------------------------------------------------------------------------
# SIR Model
# ------------------------------------------------------------------------------

def test_sir_initialize():
    g = GraphWrapper(nx.Graph())
    m = SIRModel()

    s = m.initialize(g, seed_set=[5], params={})
    assert s.active == {5}
    assert s.new_active == {5}
    assert s.removed == set()

def test_sir_spread_and_recover():
    g = nx.path_graph(3)   # 0-1-2
    w = GraphWrapper(g)

    m = SIRModel()
    random.seed(1)

    s = m.initialize(w, seed_set=[1], params={})
    s = m.step(w, s, params={"transmission_rate": 1.0, "recovery_rate": 1.0})

    # Node 1 must recover
    assert 1 in s.removed

    # Node 2 gets infected
    assert 2 in s.active


# ------------------------------------------------------------------------------
# IC Model
# ------------------------------------------------------------------------------

def test_ic_initialize():
    g = GraphWrapper(nx.Graph())
    m = ICModel()

    s = m.initialize(g, seed_set=[0], params={})
    assert s.active == {0}
    assert s.new_active == {0}

def test_ic_step_spread():
    g = nx.Graph()
    g.add_edge(0,1)
    w = GraphWrapper(g)

    m = ICModel()
    random.seed(0)

    s = m.initialize(w, seed_set=[0], params={"p": 1.0})
    s = m.step(w, s, params={"p": 1.0})

    assert s.active == {0,1}
    assert s.time == 1


# ------------------------------------------------------------------------------
# LT Model
# ------------------------------------------------------------------------------

def test_lt_initialize_thresholds():
    g = nx.path_graph(3)
    w = GraphWrapper(g)

    m = LTModel()
    s = m.initialize(
        w,
        seed_set=[0],
        params={"threshold": 0.2, "thresholds": {1: 0.5}}
    )

    assert s.active == {0}
    assert s.new_active == {0}
    assert s.metadata["thresholds"][0] == 0.2
    assert s.metadata["thresholds"][1] == 0.5
    assert s.metadata["thresholds"][2] == 0.2

def test_lt_spread_simple():
    g = nx.Graph()
    g.add_edge(0,1, weight=1.0)
    w = GraphWrapper(g)

    m = LTModel()
    s = m.initialize(
        w, seed_set=[0],
        params={"threshold": 0.5}
    )

    s = m.step(w, s, params={"threshold": 0.5})
    assert s.active == {0,1}
    assert s.new_active == {1}

def test_lt_no_spread_under_threshold():
    g = nx.Graph()
    g.add_edge(0,1, weight=0.2)
    w = GraphWrapper(g)

    m = LTModel()
    s = m.initialize(
        w, seed_set=[0],
        params={"threshold": 0.5}
    )
    s = m.step(w, s, params={"threshold": 0.5})

    # Not enough influence to activate node 1
    assert s.active == {0}
    assert s.new_active == set()
    assert s.finished is True
