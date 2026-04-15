"""MCP tool wrappers for RAG ingestion — thin adapters over ArgumentAnalysisEngine."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from aa_mcp.engine import ArgumentAnalysisEngine


def _engine(ctx: Context) -> ArgumentAnalysisEngine:
    return ctx.request_context.lifespan_context


async def _prog(ctx, pct, msg):
    await ctx.report_progress(pct, 100, msg)


async def ingest_file(
    path: str,
    collection: str = "default",
    title: str | None = None,
    metadata: dict | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Ingest a local file (PDF, .md, .txt, or any text format) into the RAG store."""
    async def _on_progress(pct, msg): await _prog(ctx, pct, msg)
    return await _engine(ctx).ingest_file(path, collection, title=title, metadata=metadata,
                                          on_progress=_on_progress)


async def ingest_url(
    url: str,
    collection: str = "default",
    title: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Fetch and ingest a web page into the RAG store."""
    async def _on_progress(pct, msg): await _prog(ctx, pct, msg)
    return await _engine(ctx).ingest_url(url, collection, title=title,
                                         on_progress=_on_progress)


async def ingest_text(
    text: str,
    title: str,
    collection: str = "default",
    metadata: dict | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Ingest raw text directly into the RAG store."""
    async def _on_progress(pct, msg): await _prog(ctx, pct, msg)
    return await _engine(ctx).ingest_text(text, title, collection, metadata=metadata,
                                          on_progress=_on_progress)


async def ingest_directory(
    path: str,
    collection: str = "default",
    pattern: str = "*",
    recursive: bool = False,
    skip_existing: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Ingest all matching files in a directory into the RAG store.

    Args:
        path: Directory to scan.
        collection: RAG collection name.
        pattern: Glob pattern for file matching (e.g. "*.pdf", "**/*.md"). Default "*".
        recursive: If True, scan subdirectories recursively.
        skip_existing: If True (default), skip files already in the collection.
    """
    async def _on_progress(pct, msg): await _prog(ctx, pct, msg)
    return await _engine(ctx).ingest_directory(path, collection, pattern=pattern, recursive=recursive,
                                               skip_existing=skip_existing,
                                               on_progress=_on_progress)


async def ingest_file_list(
    paths: list[str],
    collection: str = "default",
    skip_existing: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Ingest an explicit list of local file paths into the RAG store.

    Args:
        paths: List of absolute or relative file paths (PDF, .md, .txt, etc.).
        collection: RAG collection name.
        skip_existing: If True (default), skip files already in the collection.
    """
    async def _on_progress(pct, msg): await _prog(ctx, pct, msg)
    return await _engine(ctx).ingest_file_list(paths, collection, skip_existing=skip_existing,
                                               on_progress=_on_progress)


async def ingest_url_list(
    urls: list[str],
    collection: str = "default",
    skip_existing: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Fetch and ingest a list of URLs into the RAG store.

    Args:
        urls: List of URLs to fetch and ingest.
        collection: RAG collection name.
        skip_existing: If True (default), skip URLs already in the collection.
    """
    async def _on_progress(pct, msg): await _prog(ctx, pct, msg)
    return await _engine(ctx).ingest_url_list(urls, collection, skip_existing=skip_existing,
                                              on_progress=_on_progress)
