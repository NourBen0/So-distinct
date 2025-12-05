import pytest
import asyncio
from unittest.mock import patch, MagicMock

import networkx as nx

from sodistinct.core.graph_wrapper import GraphWrapper
from sodistinct.core.engine import SimulationResult
from sodistinct.core.models import SIModel
from sodistinct.orchestrator.local_async import (
    AsyncLocalOrchestrator,
    run_batch_async,
)


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def simple_graph():
    """Petit graphe simple utilisé par tous les tests."""
    g = nx.path_graph(4)  # 0–1–2–3
    return GraphWrapper(g)


@pytest.fixture
def fake_simulation_result():
    """Résultat de simulation synthétique pour mocks."""
    return SimulationResult(
        timeline=[{"step": 0, "active": [0], "new_active": [0], "removed": []}],
        active_final=[0],
        steps=0,
        runtime_ms=1,
        seed_set=[0],
        model_name="Mock",
        params={},
        metadata={},
    )


# ------------------------------------------------------------------------------
# Test run_many (en utilisant un mock pour run_simulation)
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_many_with_mocked_simulation(simple_graph, fake_simulation_result):

    model = SIModel()

    # Patcher run_simulation pour retourner un résultat immédiat
    with patch("sodistinct.orchestrator.local_async.run_simulation") as mock_run:
        mock_run.return_value = fake_simulation_result

        orchestrator = AsyncLocalOrchestrator(
            max_workers=2,
            use_processes=False,  # thread pool pour la stabilité en test
            progress_enabled=False,
        )

        seed_sets = [[0], [1], [2]]
        params = {"transmission_rate": 1.0}

        results = await orchestrator.run_many(
            model=model,
            graph=simple_graph,
            seed_sets=seed_sets,
            params=params,
        )

        assert len(results) == 3
        for res in results:
            assert isinstance(res, SimulationResult)
            assert res.active_final == [0]

        orchestrator.shutdown()


# ------------------------------------------------------------------------------
# Test base_rng_seed (mock + vérification des appels)
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_many_rng_seed(simple_graph, fake_simulation_result):
    model = SIModel()

    with patch("sodistinct.orchestrator.local_async.run_simulation") as mock_run:
        mock_run.return_value = fake_simulation_result

        orch = AsyncLocalOrchestrator(
            max_workers=2,
            use_processes=False,
            progress_enabled=False,
        )

        seed_sets = [[0], [1]]
        params = {}

        await orch.run_many(
            model=model,
            graph=simple_graph,
            seed_sets=seed_sets,
            params=params,
            base_rng_seed=100,
        )

        # Vérifie que run_simulation a été appelé avec les bons rng_seed
        rng_seeds = [call.kwargs["rng_seed"] for call in mock_run.call_args_list]
        assert rng_seeds == [100, 101]

        orch.shutdown()


# ------------------------------------------------------------------------------
# Test progress_callback
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_many_progress_callback(simple_graph, fake_simulation_result):
    model = SIModel()
    progress_events = []

    def callback(event):
        progress_events.append(event)

    with patch("sodistinct.orchestrator.local_async.run_simulation") as mock_run:
        mock_run.return_value = fake_simulation_result

        orch = AsyncLocalOrchestrator(
            max_workers=2,
            use_processes=False,
            progress_enabled=True,
        )

        seed_sets = [[0], [1], [2]]

        await orch.run_many(
            model=model,
            graph=simple_graph,
            seed_sets=seed_sets,
            params={},
            progress_callback=callback,
        )

        # On s'attend à 3 callbacks
        assert len(progress_events) == 3
        assert progress_events[0]["index"] == 0

        orch.shutdown()


# ------------------------------------------------------------------------------
# Test cancel()
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_many_cancel(simple_graph, fake_simulation_result):

    model = SIModel()

    with patch("sodistinct.orchestrator.local_async.run_simulation") as mock_run:
        # Simule un délai pour laisser le temps d'annuler
        async def delayed_result():
            await asyncio.sleep(0.01)
            return fake_simulation_result

        mock_run.side_effect = lambda *args, **kwargs: fake_simulation_result

        orch = AsyncLocalOrchestrator(
            max_workers=4,
            use_processes=False,
            progress_enabled=False,
        )

        seed_sets = [[i] for i in range(20)]

        # Annulation juste après le lancement
        asyncio.get_event_loop().call_soon(orch.cancel)

        results = await orch.run_many(
            model=model,
            graph=simple_graph,
            seed_sets=seed_sets,
            params={},
        )

        # Pas forcément 0 résultats, mais < total
        assert len(results) < 20

        orch.shutdown()


# ------------------------------------------------------------------------------
# Test run_batch_async helper
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_batch_async_helper(simple_graph, fake_simulation_result):

    model = SIModel()

    with patch("sodistinct.orchestrator.local_async.run_simulation") as mock_run:
        mock_run.return_value = fake_simulation_result

        results = await run_batch_async(
            model=model,
            graph=simple_graph,
            seed_sets=[[0], [1], [2], [3]],
            params={},
            max_workers=2,
            use_processes=False,
        )

        assert len(results) == 4
        for r in results:
            assert isinstance(r, SimulationResult)
            assert r.steps == 0
