import pytest
from unittest.mock import patch, MagicMock

import networkx as nx

from sodistinct.core.graph_wrapper import GraphWrapper
from sodistinct.core.engine import SimulationResult
from sodistinct.core.models import SIModel
from sodistinct.orchestrator.parallel import (
    ParallelExecutor,
    run_batch_parallel,
)


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def simple_graph():
    g = nx.path_graph(4)
    return GraphWrapper(g)


@pytest.fixture
def fake_result():
    return SimulationResult(
        timeline=[{"step": 0, "active": [0], "new_active": [0], "removed": []}],
        active_final=[0],
        steps=0,
        runtime_ms=1,
        seed_set=[0],
        model_name="Mock",
        params={},
        metadata={}
    )


# ------------------------------------------------------------------------------
# run_many() tests
# ------------------------------------------------------------------------------

def test_run_many_basic(simple_graph, fake_result):
    model = SIModel()

    # Mock run_simulation
    with patch("sodistinct.orchestrator.parallel.run_simulation") as mock_run:
        mock_run.return_value = fake_result

        exec = ParallelExecutor(
            max_workers=2,
            use_processes=False,   # threadpool pour tests rapides
        )

        seed_sets = [[0], [1], [2]]
        params = {"transmission_rate": 1.0}

        results = exec.run_many(
            model=model,
            graph=simple_graph,
            seed_sets=seed_sets,
            params=params,
        )

        assert len(results) == 3
        for r in results:
            assert isinstance(r, SimulationResult)
            assert r.active_final == [0]

        exec.shutdown()


def test_run_many_with_base_rng_seed(simple_graph, fake_result):
    model = SIModel()

    with patch("sodistinct.orchestrator.parallel.run_simulation") as mock_run:
        mock_run.return_value = fake_result

        exec = ParallelExecutor(
            max_workers=2,
            use_processes=False,
        )

        seed_sets = [[0], [1]]
        params = {}

        exec.run_many(
            model=model,
            graph=simple_graph,
            seed_sets=seed_sets,
            params=params,
            base_rng_seed=10,
        )

        # Vérification des seeds envoyés
        used_rng = [call.args[4] for call in mock_run.call_args_list]  # rng_seed is 5th param
        assert used_rng == [10, 11]

        exec.shutdown()


def test_run_many_progress_callback(simple_graph, fake_result):
    model = SIModel()
    events = []

    def cb(ev):
        events.append(ev)

    with patch("sodistinct.orchestrator.parallel.run_simulation") as mock_run:
        mock_run.return_value = fake_result

        exec = ParallelExecutor(
            max_workers=2,
            use_processes=False,
            progress_callback=cb,
        )

        seed_sets = [[0], [1], [2]]
        exec.run_many(model, simple_graph, seed_sets, params={})

        assert len(events) == 3
        assert events[0]["index"] == 0

        exec.shutdown()


def test_run_many_cancel(simple_graph, fake_result):
    model = SIModel()

    with patch("sodistinct.orchestrator.parallel.run_simulation") as mock_run:
        mock_run.return_value = fake_result

        exec = ParallelExecutor(
            max_workers=4,
            use_processes=False,
        )

        seed_sets = [[i] for i in range(20)]

        # Annuler immédiatement
        exec.cancel()

        results = exec.run_many(
            model=model,
            graph=simple_graph,
            seed_sets=seed_sets,
            params={},
        )

        # Zéro ou très peu de résultats
        assert len(results) <= 1

        exec.shutdown()


# ------------------------------------------------------------------------------
# map_runs() tests
# ------------------------------------------------------------------------------

def test_map_runs_basic():
    exec = ParallelExecutor(
        max_workers=3,
        use_processes=False,
    )

    # Fonction simple à mapper
    def add_fn(a, b):
        return a + b

    args = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

    res = exec.map_runs(add_fn, args)
    assert sorted(res) == [3, 7]

    exec.shutdown()


def test_map_runs_cancel():
    exec = ParallelExecutor(
        max_workers=3,
        use_processes=False,
    )

    exec.cancel()

    def fn(x): return x * 2

    args = [{"x": i} for i in range(10)]
    results = exec.map_runs(fn, args)

    # Si cancel appelé avant soumission → 0 result
    assert len(results) == 0

    exec.shutdown()


# ------------------------------------------------------------------------------
# run_batch_parallel() helper
# ------------------------------------------------------------------------------

def test_run_batch_parallel(simple_graph, fake_result):
    model = SIModel()

    with patch("sodistinct.orchestrator.parallel.run_simulation") as mock_run:
        mock_run.return_value = fake_result

        results = run_batch_parallel(
            model=model,
            graph=simple_graph,
            seed_sets=[[0], [1], [2]],
            params={},
            max_workers=2,
            use_processes=False,
        )

        assert len(results) == 3
        for r in results:
            assert isinstance(r, SimulationResult)
            assert r.steps == 0
