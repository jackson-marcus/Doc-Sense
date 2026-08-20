"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from docsense import __version__
from docsense.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="docsense",
        description="Document-intelligence RAG API with swappable LLM backends",
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()
