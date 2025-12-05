"""
stream.py
---------
Module dédié aux opérations d'I/O en streaming pour SoDistinct.

Objectifs :
- Lire des fichiers volumineux sans charger tout en mémoire.
- Fournir des générateurs synchrones et asynchrones de lignes (streaming).
- Fournir des helpers pour écrire en flux, très utile pour :
    * générer de grands graphes synthétiques
    * logguer des résultats d'expérience
    * écrire des checkpoints
- Utiliser aiofiles pour maximiser l'efficacité en mode async.

Ce module complète loader.py / saver.py.
"""

from __future__ import annotations

import os
import pathlib
import logging
from typing import AsyncGenerator, Generator, Optional

import aiofiles

logger = logging.getLogger("sodistinct.io.stream")

# ---------------------------------------------------------------------------
# Générateurs synchrones
# ---------------------------------------------------------------------------

def stream_lines(path: str, skip_empty: bool = True) -> Generator[str, None, None]:
    """
    Générateur synchrone de lignes, adapté aux très grands fichiers.

    Paramètres :
        path         : chemin du fichier
        skip_empty   : ignore les lignes vides ou contenant seulement des espaces

    Usage :
        for line in stream_lines("data/big.edgelist"):
            ...
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    logger.debug(f"Streaming sync depuis {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if skip_empty and not line.strip():
                continue
            yield line.rstrip("\n")


def stream_chunks(path: str, chunk_size: int = 1024 * 1024) -> Generator[str, None, None]:
    """
    Stream synchrone par chunks (blocs de taille fixe).

    Utile pour analyser ou transformer de très gros fichiers.

    Paramètres :
        path        : fichier
        chunk_size  : taille des blocs (par défaut 1 Mo)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    logger.debug(f"Streaming {path} par chunks de {chunk_size} bytes")

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


# ---------------------------------------------------------------------------
# Générateurs asynchrones
# ---------------------------------------------------------------------------

async def stream_lines_async(path: str, skip_empty: bool = True) -> AsyncGenerator[str, None]:
    """
    Générateur asynchrone de lignes, basé sur aiofiles.

    Usage:
        async for line in stream_lines_async("data/big.edgelist"):
            ...
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    logger.debug(f"[async] Streaming lines depuis {path}")

    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        async for line in f:
            if skip_empty and not line.strip():
                continue
            yield line.rstrip("\n")


async def stream_chunks_async(path: str, chunk_size: int = 1024 * 1024) -> AsyncGenerator[str, None]:
    """
    Stream asynchrone par chunks.

    Utile pour API FastAPI ou pipelines asyncio où on veut éviter de bloquer l'event-loop.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    logger.debug(f"[async] Streaming chunks depuis {path}")

    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            yield chunk


# ---------------------------------------------------------------------------
# Écriture synchrone et asynchrone en mode flux
# ---------------------------------------------------------------------------

def stream_writer(path: str, overwrite: bool = False):
    """
    Générateur/gestionnaire pour écriture synchrone par flux.

    Usage:
        with stream_writer("results/log.txt") as w:
            w.write("message 1\n")
            w.write("message 2\n")

    """
    p = pathlib.Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{path} existe déjà (overwrite=False).")

    p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, "w", encoding="utf-8")


async def stream_writer_async(path: str, overwrite: bool = False):
    """
    Gestionnaire async pour écriture en flux.

    Usage:
        async with stream_writer_async("results/log.txt") as w:
            await w.write("Hello\n")
            await w.flush()
    """
    p = pathlib.Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{path} existe déjà (overwrite=False).")

    p.parent.mkdir(parents=True, exist_ok=True)
    return await aiofiles.open(p, "w", encoding="utf-8")


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def file_size(path: str) -> int:
    """Retourne la taille du fichier en octets."""
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path)


def count_lines(path: str) -> int:
    """Compter le nombre de lignes d'un fichier volumineux sans tout charger."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    c = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            c += 1
    return c


async def count_lines_async(path: str) -> int:
    """Version asynchrone de count_lines."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    c = 0
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        async for _ in f:
            c += 1
    return c
