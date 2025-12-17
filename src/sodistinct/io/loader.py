from __future__ import annotations

import os
import csv
import logging
import pathlib
from typing import Optional, Dict, Any

import networkx as nx
import aiofiles

from sodistinct.core.graph_wrapper import GraphWrapper

logger = logging.getLogger("sodistinct.io.loader")




SUPPORTED_FORMATS = {".edgelist", ".txt", ".csv", ".gexf", ".graphml", ".xml"}


def detect_format(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Format non supporté : '{ext}'. Formats acceptés : {sorted(SUPPORTED_FORMATS)}"
        )
    return ext



def load_graph(path: str) -> GraphWrapper:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    ext = detect_format(path)
    logger.info(f"Chargement du graphe depuis '{path}' (format {ext})")

    if ext in {".edgelist", ".txt"}:
        g = _load_edgelist(path)
    elif ext == ".csv":
        g = _load_csv(path)
    elif ext == ".gexf":
        g = nx.read_gexf(path)
    elif ext in {".graphml", ".xml"}:
        g = nx.read_graphml(path)
    else:
        raise RuntimeError(f"Extension non gérée : {ext}")

    logger.info(f"Graphe chargé. Nombre de nœuds: {g.number_of_nodes()}, arêtes: {g.number_of_edges()}")
    return GraphWrapper(g)



async def load_graph_async(path: str) -> GraphWrapper:
   
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    ext = detect_format(path)
    logger.info(f"[async] Chargement du graphe depuis '{path}' (format {ext})")

    if ext in {".gexf", ".graphml", ".xml"}:
        import asyncio
        loop = asyncio.get_running_loop()
        g = await loop.run_in_executor(None, lambda: load_graph(path))
        return g

    if ext in {".edgelist", ".txt"}:
        g = await _load_edgelist_async(path)
    elif ext == ".csv":
        g = await _load_csv_async(path)
    else:
        import asyncio
        loop = asyncio.get_running_loop()
        g = await loop.run_in_executor(None, lambda: load_graph(path))
        return g

    logger.info(f"[async] Graphe chargé. Nombre de nœuds: {g.number_of_nodes()}, arêtes: {g.number_of_edges()}")
    return GraphWrapper(g)



def _load_edgelist(path: str) -> nx.Graph:
    g = nx.Graph()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                u, v = parts
                g.add_edge(u, v)
            elif len(parts) == 3:
                u, v, w = parts
                try:
                    w = float(w)
                except ValueError:
                    w = 1.0
                g.add_edge(u, v, weight=w)
    return g


def _load_csv(path: str) -> nx.Graph:
    g = nx.Graph()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "source" not in reader.fieldnames or "target" not in reader.fieldnames:
            raise ValueError("CSV doit contenir au moins les colonnes 'source' et 'target'.")

        for row in reader:
            u = row["source"]
            v = row["target"]
            w = None
            if "weight" in row and row["weight"] not in (None, "", " "):
                try:
                    w = float(row["weight"])
                except ValueError:
                    w = 1.0
            if w is None:
                g.add_edge(u, v)
            else:
                g.add_edge(u, v, weight=w)
    return g



async def _load_edgelist_async(path: str) -> nx.Graph:
    g = nx.Graph()
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        async for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                u, v = parts
                g.add_edge(u, v)
            elif len(parts) == 3:
                u, v, w = parts
                try:
                    w = float(w)
                except ValueError:
                    w = 1.0
                g.add_edge(u, v, weight=w)
    return g


async def _load_csv_async(path: str) -> nx.Graph:
    g = nx.Graph()
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        header = await f.readline()
        columns = [c.strip() for c in header.split(",")]
        if "source" not in columns or "target" not in columns:
            raise ValueError("CSV doit contenir 'source' et 'target'.")

        idx_source = columns.index("source")
        idx_target = columns.index("target")
        idx_weight = columns.index("weight") if "weight" in columns else None

        async for line in f:
            parts = [x.strip() for x in line.split(",")]
            u = parts[idx_source]
            v = parts[idx_target]

            if idx_weight is not None:
                try:
                    w = float(parts[idx_weight])
                except ValueError:
                    w = 1.0
                g.add_edge(u, v, weight=w)
            else:
                g.add_edge(u, v)

    return g



def is_graph_file(path: str) -> bool:
    return pathlib.Path(path).suffix.lower() in SUPPORTED_FORMATS


def load_any(path: str, async_mode: bool = False):
    if async_mode:
        import asyncio
        return asyncio.run(load_graph_async(path))
    return load_graph(path)
