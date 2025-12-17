from __future__ import annotations

import os
import pathlib
import logging
from typing import AsyncGenerator, Generator, Optional

import aiofiles

logger = logging.getLogger("sodistinct.io.stream")


def stream_lines(path: str, skip_empty: bool = True) -> Generator[str, None, None]:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    logger.debug(f"Streaming sync depuis {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if skip_empty and not line.strip():
                continue
            yield line.rstrip("\n")


def stream_chunks(path: str, chunk_size: int = 1024 * 1024) -> Generator[str, None, None]:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    logger.debug(f"Streaming {path} par chunks de {chunk_size} bytes")

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk



async def stream_lines_async(path: str, skip_empty: bool = True) -> AsyncGenerator[str, None]:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    logger.debug(f"[async] Streaming lines depuis {path}")

    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        async for line in f:
            if skip_empty and not line.strip():
                continue
            yield line.rstrip("\n")


async def stream_chunks_async(path: str, chunk_size: int = 1024 * 1024) -> AsyncGenerator[str, None]:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    logger.debug(f"[async] Streaming chunks depuis {path}")

    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            yield chunk



def stream_writer(path: str, overwrite: bool = False):

    p = pathlib.Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{path} existe déjà (overwrite=False).")

    p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, "w", encoding="utf-8")


async def stream_writer_async(path: str, overwrite: bool = False):

    p = pathlib.Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{path} existe déjà (overwrite=False).")

    p.parent.mkdir(parents=True, exist_ok=True)
    return await aiofiles.open(p, "w", encoding="utf-8")




def file_size(path: str) -> int:
    
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path)


def count_lines(path: str) -> int:
    
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    c = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            c += 1
    return c


async def count_lines_async(path: str) -> int:
    
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    c = 0
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        async for _ in f:
            c += 1
    return c
