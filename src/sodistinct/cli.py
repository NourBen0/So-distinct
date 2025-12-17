import json
from pathlib import Path
from typing import Optional, List, Any
import typer

from sodistinct.utils.logging import get_logger
from sodistinct.utils.persistence import (
    save_result,
    load_result,
)
from sodistinct.io.loader import load_graph
from sodistinct.core.graph_wrapper import GraphWrapper
from sodistinct.core.models import (
    SIModel,
    SIRModel,
    ICModel,
    LTModel,
)
from sodistinct.core.engine import run_simulation, SimulationResult
from sodistinct.orchestrator.parallel import run_batch_parallel



app = typer.Typer(help="SoDistinct — CLI officielle")
logger = get_logger("sodistinct.cli")

MODEL_REGISTRY = {
    "si": SIModel,
    "sir": SIRModel,
    "ic": ICModel,
    "lt": LTModel,
}


# ==============================================================================
# Helpers
# ==============================================================================

def load_graph_wrapper(path: str) -> GraphWrapper:
    g = load_graph(path)
    return GraphWrapper.from_networkx(g)


def get_model(name: str):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        typer.echo(
            f"Modèle inconnu '{name}'. "
            f"Modèles disponibles : {list(MODEL_REGISTRY)}"
        )
        raise typer.Exit()
    return MODEL_REGISTRY[name]()


# ==============================================================================
# Commande : Simulation simple
# ==============================================================================

@app.command()
def run(
    model: str = typer.Argument(..., help="Modèle : si | sir | ic | lt"),
    graph: str = typer.Argument(..., help="Chemin du graphe"),
    seed_set: str = typer.Option("0", help="Seed set (ex: '1,5,10')"),
    params: str = typer.Option("{}", help="Paramètres JSON du modèle"),
    rng_seed: Optional[int] = typer.Option(None, help="Seed aléatoire"),
    out_json: bool = typer.Option(False, help="Sortie JSON"),
):
    """
    Exécute une simulation simple.
    """
    try:
        seeds = [s.strip() for s in seed_set.split(",")]
        params_dict = json.loads(params)
    except Exception as e:
        typer.echo(f"Erreur parsing paramètres: {e}")
        raise typer.Exit()

    model_inst = get_model(model)
    graph_wrap = load_graph_wrapper(graph)

    result = run_simulation(
        model=model_inst,
        graph=graph_wrap,
        seed_set=seeds,
        params=params_dict,
        rng_seed=rng_seed,
    )

    if out_json:
        typer.echo(json.dumps(result.__dict__, indent=2))
    else:
        typer.echo(f"Simulation terminée en {result.runtime_ms:.2f} ms")
        typer.echo(f"Steps: {result.steps}, Final activated: {len(result.active_final)}")


# ==============================================================================
# Commande : Batch local parallèle
# ==============================================================================

@app.command()
def batch(
    model: str = typer.Argument(...),
    graph: str = typer.Argument(...),
    seed_sets_file: str = typer.Argument(..., help="Fichier JSON contenant une liste de seed sets"),
    params: str = typer.Option("{}", help="Paramètres JSON du modèle"),
    workers: int = typer.Option(4),
    use_processes: bool = typer.Option(True),
    out: Optional[str] = typer.Option(None, help="Fichier output JSON"),
):
    """
    Exécute un batch de simulations en parallèle local.
    """
    model_inst = get_model(model)
    graph_wrap = load_graph_wrapper(graph)

    try:
        with open(seed_sets_file, "r") as f:
            seed_sets = json.load(f)
    except Exception as e:
        typer.echo(f"Erreur lecture seed_sets_file: {e}")
        raise typer.Exit()

    params_dict = json.loads(params)

    results = run_batch_parallel(
        model=model_inst,
        graph=graph_wrap,
        seed_sets=seed_sets,
        params=params_dict,
        max_workers=workers,
        use_processes=use_processes,
    )

    if out:
        Path(out).write_text(json.dumps([r.__dict__ for r in results], indent=2))
        typer.echo(f"Résultats sauvegardés dans {out}")
    else:
        typer.echo(f"{len(results)} simulations terminées.")


# ==============================================================================
# Commande : Infos graphe
# ==============================================================================

@app.command("graph-info")
def graph_info(
    path: str = typer.Argument(...),
    preview: int = typer.Option(10),
):
    """
    Affiche des informations basiques sur un graphe.
    """
    g = load_graph(path)
    typer.echo(f"{g.number_of_nodes()} nodes / {g.number_of_edges()} edges")
    typer.echo(f"Directed: {g.is_directed()}")
    typer.echo(f"Preview nodes: {list(g.nodes())[:preview]}")
    typer.echo(f"Preview edges: {list(g.edges())[:preview]}")


# ==============================================================================
# Commande : Sauvegarder un résultat
# ==============================================================================

@app.command("save-result")
def save_result_cmd(
    path_json: str = typer.Argument(...),
):
    """
    Sauvegarde un SimulationResult (fichier JSON → pickle officiel SoDistinct)
    """
    try:
        data = json.loads(Path(path_json).read_text())
        result = SimulationResult(**data)
    except Exception as e:
        typer.echo(f"Erreur parsing JSON: {e}")
        raise typer.Exit()

    save_path = save_result(result)
    typer.echo(f"Résultat sauvegardé dans {save_path}")


# ==============================================================================
# Commande : Charger un résultat
# ==============================================================================

@app.command("load-result")
def load_result_cmd(
    path_pkl: str = typer.Argument(...),
    show_json: bool = typer.Option(True),
):
    """
    Charge un SimulationResult pickle et l’affiche.
    """
    try:
        result = load_result(path_pkl)
    except Exception as e:
        typer.echo(f"Erreur lecture pickle : {e}")
        raise typer.Exit()

    if show_json:
        typer.echo(json.dumps(result.__dict__, indent=2))
    else:
        typer.echo(result)


# ==============================================================================
# Entrée principale
# ==============================================================================

if __name__ == "__main__":
    app()
