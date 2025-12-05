import pytest
import asyncio
import networkx as nx

from sodistinct.core.graph_wrapper import GraphWrapper
from sodistinct.core.models import SIModel, ICModel
from sodistinct.core.engine import run_simulation, SimulationResult
from sodistinct.core.metrics import compute_all_metrics
from sodistinct.orchestrator.local_async import AsyncLocalOrchestrator
from sodistinct.orchestrator.parallel import ParallelExecutor


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def simple_graph():
    """Graphe simple 0–1–2–3 utilisé pour tous les tests."""
    g = nx.path_graph(4)
    return GraphWrapper(g)


# ------------------------------------------------------------------------------
# ENGINE : test end-to-end
# ------------------------------------------------------------------------------

def test_engine_end_to_end_si(simple_graph):
    model = SIModel()

    res = run_simulation(
        model=model,
        graph=simple_graph,
        seed_set=[0],
        params={"transmission_rate": 1.0},
        rng_seed=123,
    )

    assert isinstance(res, SimulationResult)
    assert set(res.active_final) == {0, 1, 2, 3}
    assert res.steps >= 1

    # Check timeline consistency
    for step in res.timeline:
        assert "active" in step and "new_active" in step


def test_engine_end_to_end_ic(simple_graph):
    model = ICModel()

    res = run_simulation(
        model=model,
        graph=simple_graph,
        seed_set=[0],
        params={"p": 1.0},
        rng_seed=5,
    )

    assert set(res.active_final) == {0, 1, 2, 3}
    assert res.steps >= 1


# ------------------------------------------------------------------------------
# METRICS : end-to-end with real SimulationResult
# ------------------------------------------------------------------------------

def test_metrics_end_to_end(simple_graph):
    model = SIModel()

    result = run_simulation(
        model=model,
        graph=simple_graph,
        seed_set=[0],
        params={"transmission_rate": 1.0},
        rng_seed=1,
    )

    metrics = compute_all_metrics(result, total_nodes=4)

    assert "coverage" in metrics
    assert "speed" in metrics
    assert "reach_time" in metrics
    assert "influence" in metrics

    # Coverage should be full
    assert metrics["coverage"].coverage_ratio == 1.0

    # Reach time: node 0 at step 0
    assert metrics["reach_time"].node_reach_time[0] == 0


# ------------------------------------------------------------------------------
# ORCHESTRATOR ASYNC : full pipeline
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_orchestrator_full_pipeline(simple_graph):
    model = SIModel()
    seed_sets = [[0], [1], [2]]
    params = {"transmission_rate": 1.0}

    orch = AsyncLocalOrchestrator(
        max_workers=2,
        use_processes=False,
        progress_enabled=False,
    )

    try:
        results = await orch.run_many(
            model=model,
            graph=simple_graph,
            seed_sets=seed_sets,
            params=params,
            base_rng_seed=10,
        )

        assert len(results) == 3
        for res in results:
            assert isinstance(res, SimulationResult)

        # Compute metrics for each result
        metrics_list = [compute_all_metrics(r, total_nodes=4) for r in results]
        assert len(metrics_list) == 3

        # All coverage should be complete
        assert all(m["coverage"].coverage_ratio == 1.0 for m in metrics_list)

    finally:
        orch.shutdown()


# ------------------------------------------------------------------------------
# ORCHESTRATOR PARALLEL : full pipeline
# ------------------------------------------------------------------------------

def test_parallel_orchestrator_full_pipeline(simple_graph):
    model = SIModel()
    seed_sets = [[0], [1], [2]]
    params = {"transmission_rate": 1.0}

    executor = ParallelExecutor(
        max_workers=2,
        use_processes=False,  # threads for reliable testing
    )

    try:
        results = executor.run_many(
            model=model,
            graph=simple_graph,
            seed_sets=seed_sets,
            params=params,
            base_rng_seed=5,
        )

        assert len(results) == 3
        for r in results:
            assert isinstance(r, SimulationResult)

        # Metrics integration
        metrics_all = [compute_all_metrics(r, total_nodes=4) for r in results]
        assert all(m["coverage"].coverage_ratio == 1.0 for m in metrics_all)

    finally:
        executor.shutdown()


# ------------------------------------------------------------------------------
# FULL INTEGRATION SCENARIO:
#   orchestrator → engine → metrics → aggregated results
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_system_pipeline(simple_graph):
    """
    Vérifie toute la chaîne :
    orchestrator async → engine → metrics → résultats agrégés
    """

    model = ICModel()
    seed_sets = [[0], [1], [2], [3]]
    params = {"p": 1.0}

    orch = AsyncLocalOrchestrator(
        max_workers=4,
        use_processes=False,
        progress_enabled=False,
    )

    try:
        results = await orch.run_many(
            model=model,
            graph=simple_graph,
            seed_sets=seed_sets,
            params=params,
            base_rng_seed=99,
        )

        assert len(results) == 4

        metrics_all = [compute_all_metrics(r, total_nodes=4) for r in results]

        # Validate all metrics exist
        for m in metrics_all:
            assert "coverage" in m
            assert "speed" in m
            assert "reach_time" in m
            assert "influence" in m

        # Check deterministic behavior via seed
        # ICModel with p=1 fully activates chain
        assert all(set(r.active_final) == {0, 1, 2, 3} for r in results)

    finally:
        orch.shutdown()
