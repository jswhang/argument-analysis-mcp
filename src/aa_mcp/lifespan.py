"""Server lifespan: create the ArgumentAnalysisEngine at startup, close at shutdown."""

from __future__ import annotations

from contextlib import asynccontextmanager

from aa_mcp.config import ServerConfig
from aa_mcp.engine import ArgumentAnalysisEngine

# AppState IS the engine — tool functions receive it via ctx.request_context.lifespan_context
AppState = ArgumentAnalysisEngine


@asynccontextmanager
async def app_lifespan(app):
    async with ArgumentAnalysisEngine(ServerConfig()) as engine:
        yield engine
