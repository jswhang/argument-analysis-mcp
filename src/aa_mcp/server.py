"""Main MCP server: register all tools, resources, and prompts."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from aa_mcp.lifespan import app_lifespan

mcp = FastMCP(
    name="argument-analysis",
    instructions=(
        "An argument analysis server. "
        "Primary use: assess_argument(text) — analyzes any argument for logical validity, "
        "structural soundness, and reasoning quality. Detects logical fallacies, flags weak "
        "components, and returns an overall validity score with a prose summary. "
        "No specialist knowledge required — just submit the text. "
        "Optionally add reference documents via ingest_file/ingest_url/ingest_text to enable "
        "cross-checking claims against domain-specific sources; logical analysis works without it. "
        "Use create_argument_map/add_argument_node/link_arguments to build visual argument graphs."
    ),
    lifespan=app_lifespan,
)

# ── Analysis tools ────────────────────────────────────────────────────────────
from aa_mcp.tools.analysis import (  # noqa: E402
    analyze_argument_structure,
    assess_argument,
    detect_fallacies,
    get_assessment,
)

mcp.tool()(assess_argument)
mcp.tool()(analyze_argument_structure)
mcp.tool()(detect_fallacies)
mcp.tool()(get_assessment)

# ── Mapping tools ─────────────────────────────────────────────────────────────
from aa_mcp.tools.mapping import (  # noqa: E402
    add_argument_node,
    auto_map_argument,
    create_argument_map,
    get_argument_map,
    link_arguments,
)

mcp.tool()(create_argument_map)
mcp.tool()(add_argument_node)
mcp.tool()(link_arguments)
mcp.tool()(get_argument_map)
mcp.tool()(auto_map_argument)

# ── RAG ingest tools ──────────────────────────────────────────────────────────
from aa_mcp.tools.rag_ingest import (  # noqa: E402
    ingest_directory,
    ingest_file,
    ingest_file_list,
    ingest_text,
    ingest_url,
    ingest_url_list,
)

mcp.tool()(ingest_file)
mcp.tool()(ingest_url)
mcp.tool()(ingest_text)
mcp.tool()(ingest_directory)
mcp.tool()(ingest_file_list)
mcp.tool()(ingest_url_list)

# ── RAG search tools ──────────────────────────────────────────────────────────
from aa_mcp.tools.rag_search import get_document, search, synthesize_answer  # noqa: E402

mcp.tool()(search)
mcp.tool()(get_document)
mcp.tool()(synthesize_answer)

# ── Management tools ──────────────────────────────────────────────────────────
from aa_mcp.tools.manage import (  # noqa: E402
    collection_stats,
    delete_argument_map,
    delete_document,
    export_argument_map,
    list_argument_maps,
    list_documents,
)

mcp.tool()(list_argument_maps)
mcp.tool()(delete_argument_map)
mcp.tool()(export_argument_map)
mcp.tool()(list_documents)
mcp.tool()(delete_document)
mcp.tool()(collection_stats)

# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("aa://rag/stats")
async def rag_stats() -> str:
    """RAG knowledge base statistics."""
    import json
    # Note: resources don't have access to lifespan context directly in all MCP clients,
    # so we re-read from SQLite here.
    from aa_mcp.config import ServerConfig
    from aa_mcp.store.document_store import DocumentStore

    config = ServerConfig()
    store = DocumentStore(config.sqlite_path)
    await store.initialize()
    collections = await store.list_collections()
    stats = {}
    for col in collections:
        stats[col] = await store.collection_stats(col)
    return json.dumps({"collections": stats}, indent=2)


@mcp.resource("aa://rag/documents/{collection}")
async def rag_documents(collection: str) -> str:
    """List all documents in a RAG collection."""
    import json

    from aa_mcp.config import ServerConfig
    from aa_mcp.store.document_store import DocumentStore

    config = ServerConfig()
    store = DocumentStore(config.sqlite_path)
    await store.initialize()
    docs = await store.list_documents(collection)
    return json.dumps(
        [{"id": d.id, "title": d.title, "source": d.source, "chunks": d.chunk_count} for d in docs],
        indent=2,
    )


@mcp.resource("aa://maps")
async def argument_maps_list() -> str:
    """List all argument maps."""
    import json

    from aa_mcp.config import ServerConfig
    from aa_mcp.store.argument_store import ArgumentStore

    config = ServerConfig()
    store = ArgumentStore(config.sqlite_path)
    await store.initialize()
    maps = await store.list_maps()
    return json.dumps(maps, indent=2, default=str)


@mcp.resource("aa://maps/{map_id}")
async def argument_map_detail(map_id: str) -> str:
    """Full argument map with nodes and edges."""
    import json

    from aa_mcp.config import ServerConfig
    from aa_mcp.store.argument_store import ArgumentStore

    config = ServerConfig()
    store = ArgumentStore(config.sqlite_path)
    await store.initialize()
    map_ = await store.get_map(map_id)
    if not map_:
        return json.dumps({"error": "not_found", "map_id": map_id})
    return map_.model_dump_json(indent=2)


# ── Prompts ───────────────────────────────────────────────────────────────────

@mcp.prompt()
def assess_argument_prompt(text: str, collection: str = "default") -> str:
    """Prompt to run a full argument assessment with the knowledge base."""
    return (
        f"Please assess the following argument using the assess_argument tool "
        f"with collection='{collection}':\n\n{text}\n\n"
        "Provide a detailed critique highlighting any logical weaknesses, factual inaccuracies "
        "based on the knowledge base, and logical fallacies."
    )


@mcp.prompt()
def rag_query_prompt(question: str, collection: str = "default") -> str:
    """Prompt to query the RAG knowledge base."""
    return (
        f"Please search the knowledge base (collection: '{collection}') "
        f"and answer the following question:\n\n{question}\n\n"
        "Use the search or synthesize_answer tool to find relevant information."
    )
