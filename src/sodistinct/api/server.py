from __future__ import annotations

import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from sodistinct.config.settings import settings
from sodistinct.api.endpoints import router as api_router

logger = logging.getLogger("sodistinct.api.server")




app = FastAPI(
    title=settings.project_name,
    description=settings.description,
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
)





app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configurable selon déploiement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{duration:.2f}"
    return response




@app.on_event("startup")
async def startup_event():
    logger.info("🚀 API SoDistinct — Initialisation...")
    logger.info(f"Chargement configuration depuis {settings=}")
    # Ici tu peux précharger graphiques, modèles, cluster Dask/Ray…


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 API SoDistinct — Arrêt du serveur...")



@app.get("/")
def home():
    return {
        "message": f"Bienvenue dans {settings.project_name}!",
        "version": settings.version,
        "author": settings.author,
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/config")
def show_config():
    """Affiche la configuration actuelle"""
    return {
        "project": settings.project_name,
        "version": settings.version,
        "backend": settings.backend,
        "num_workers": settings.num_workers,
        "data_dir": settings.data_dir
    }



app.include_router(api_router, prefix="/api")



@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "version": settings.version}
