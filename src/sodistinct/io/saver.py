"""
saver.py
--------
Module responsable de la sauvegarde des graphes et résultats pour SoDistinct.

Fonctionnalités :
- Sauvegarde en plusieurs formats : .edgelist, .csv, .gexf, .graphml
- Version synchrone et asynchrone
- Intégration directe avec GraphWrapper (NetworkX)
- Gestion automatique des répertoires
- Support d'options : overwrite, création automatique des dossiers
"""

from __future__ import annotations

import os
import csv
import pathlib
import logging
from typing import Optional, Dict, Any

import aiofiles
import networkx as nx

from sodistinct.core.graph_wrapper import GraphWrapper

logger = logging.getLogger("sodistinct.io.saver")

SUPPORTED_SAVE_FORMATS = {".edgelist", ".txt", ".csv", ".gexf", ".graphml"}


# -----------------------------------------------------------------------------
# Utilitaires
# -----------------------------------------------------------------------------

def _prepare_path(path: str, overwrite: bool = False):
    p = pathlib.Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"Le fichier {path} existe déjà (overwrite=False)")

    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def detect_save_format(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    if ext not in SUPPORTED_SAVE_FORMATS:
        raise ValueError(
            f"Extension '{ext}' non supportée pour la sauvegarde. "
            f"Formats acceptés : {sorted(SUPPORTED_SAVE_FORMATS)}"
        )
    return ext


# -----------------------------------------------------------------------------
# Sauvegardes synchrones
# -----------------------------------------------------------------------------

def save_graph(graph: GraphWrapper, path: str, overwrite: bool = False):
    """
    Sauvegarde synchrone d'un graphe dans différents formats.

    Paramètres :
        graph       : GraphWrapper (NetworkX en interne)
        path        : chemin vers le fichier de sortie
        overwrite   : si True, écrase le fichier existant
    """
    p = _prepare_path(path, overwrite)
    ext = detect_save_format(path)

    g: nx.Graph = graph.unwrap()

    logger.info(f"Sauvegarde du graphe vers '{path}' (format {ext})")

    if ext in {".edgelist", ".txt"}:
        _save_edgelist(g, p)
    elif ext == ".csv":
        _save_csv(g, p)
    elif ext == ".gexf":
        nx.write_gexf(g, p)
    elif ext == ".graphml":
        nx.write_graphml(g, p)
    else:
        raise RuntimeError(f"Format de sauvegarde non géré : {ext}")

    logger.info(f"Graphe sauvegardé ({g.number_of_nodes()} nœuds, {g.number_of_edges()} arêtes).")


def _save_edgelist(g: nx.Graph, path: pathlib.Path):
    with open(path, "w", encoding="utf-8") as f:
        for u, v, data in g.edges(data=True):
            if "weight" in data:
                f.write(f"{u} {v} {data['weight']}\n")
            else:
                f.write(f"{u} {v}\n")


def _save_csv(g: nx.Graph, path: pathlib.Path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["source", "target", "weight"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for u, v, data in g.edges(data=True):
            writer.writerow(
                {
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 1.0),
                }
            )


# -----------------------------------------------------------------------------
# Sauvegardes asynchrones
# -----------------------------------------------------------------------------

async def save_graph_async(graph: GraphWrapper, path: str, overwrite: bool = False):
    """
    Sauvegarde asynchrone d'un graphe.
    Utilise aiofiles pour les formats textuels.
    Pour GEXF/GraphML (NetworkX), fallback vers thread executor.
    """
    p = _prepare_path(path, overwrite)
    ext = detect_save_format(path)

    g: nx.Graph = graph.unwrap()

    logger.info(f"[async] Sauvegarde du graphe vers '{path}' (format {ext})")

    # Formats NX non compatibles async → fallback thread
    if ext in {".gexf", ".graphml"}:
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: save_graph(graph, path, overwrite=True))
        return

    if ext in {".edgelist", ".txt"}:
        await _save_edgelist_async(g, p)
    elif ext == ".csv":
        await _save_csv_async(g, p)
    else:
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: save_graph(graph, path, overwrite=True))
        return

    logger.info(f"[async] Graphe sauvegardé ({g.number_of_nodes()} nœuds, {g.number_of_edges()} arêtes).")


async def _save_edgelist_async(g: nx.Graph, path: pathlib.Path):
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        for u, v, data in g.edges(data=True):
            if "weight" in data:
                await f.write(f"{u} {v} {data['weight']}\n")
            else:
                await f.write(f"{u} {v}\n")


async def _save_csv_async(g: nx.Graph, path: pathlib.Path):
    """
    Sauvegarde CSV asynchrone (écriture manuelle car CSV module n’est pas async).
    """
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write("source,target,weight\n")
        for u, v, data in g.edges(data=True):
            w = data.get("weight", 1.0)
            await f.write(f"{u},{v},{w}\n")


# -----------------------------------------------------------------------------
# Wrapper unifié
# -----------------------------------------------------------------------------

def save_any(graph: GraphWrapper, path: str, async_mode: bool = False, overwrite: bool = False):
    """
    Wrapper pratique : save sync ou async.
    """
    if async_mode:
        import asyncio
        return asyncio.run(save_graph_async(graph, path, overwrite=overwrite))
    return save_graph(graph, path, overwrite=overwrite)
