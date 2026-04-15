"""ArgumentAnalysisEngine — the core library, usable standalone or via MCP.

Standalone usage::

    import asyncio
    from aa_mcp import ArgumentAnalysisEngine

    async def my_llm(prompt, system, max_tokens, temperature):
        # call Anthropic / OpenAI / local model here
        ...

    async def main():
        async with ArgumentAnalysisEngine() as engine:
            # Populate knowledge base
            await engine.ingest_file("/path/to/paper.pdf", collection="climate")

            # Check an LLM output
            assessment = await engine.assess_argument(
                text="Climate change is not man-made because...",
                collection="climate",
                llm=my_llm,
            )
            print(assessment.summary)
            for w in assessment.critical_weaknesses:
                print("WEAK:", w.component_type, w.weakness_explanation)

    asyncio.run(main())
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

import anyio

from aa_mcp.analysis.llm_caller import LLMCaller
from aa_mcp.analysis.sampling import SamplingHelper
from aa_mcp.config import ServerConfig
from aa_mcp.models.argument import ArgumentAssessment, FallacyDetection, ToulminComponents
from aa_mcp.models.mapping import ArgumentEdge, ArgumentMap, ArgumentNode, RelationshipType
from aa_mcp.models.rag import Document, SearchResult
from aa_mcp.rag.backends.base import VectorBackend
from aa_mcp.rag.chunker import TextChunker
from aa_mcp.rag.embedder import Embedder
from aa_mcp.rag.pipeline import RAGPipeline
from aa_mcp.store.argument_store import ArgumentStore
from aa_mcp.store.document_store import DocumentStore

# Optional async progress callback: (percent_complete, message) -> None
ProgressCallback = Callable[[int, str], Awaitable[None]] | None

_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".rst", ".html", ".htm", ".csv", ".json"}


async def _progress(callback: ProgressCallback, pct: int, message: str) -> None:
    if callback is not None:
        await callback(pct, message)


def _make_backend(config: ServerConfig) -> VectorBackend:
    if config.rag.backend == "chroma":
        from aa_mcp.rag.backends.chroma import ChromaBackend
        return ChromaBackend(config.chroma_path)
    raise ValueError(f"Unsupported backend: {config.rag.backend!r}. Install the appropriate extra.")


class ArgumentAnalysisEngine:
    """All analysis and RAG functionality in one async context manager.

    Can be used directly as a library or is instantiated once by the MCP
    server lifespan and held as the shared app state.
    """

    def __init__(self, config: ServerConfig | None = None) -> None:
        self.config = config or ServerConfig()
        # Subsystems — populated by initialize()
        self.embedder: Embedder | None = None
        self.backend: VectorBackend | None = None
        self.doc_store: DocumentStore | None = None
        self.arg_store: ArgumentStore | None = None
        self.pipeline: RAGPipeline | None = None
        self.chunker: TextChunker | None = None

    async def initialize(self) -> None:
        """Boot all subsystems. Called automatically by ``async with``."""
        cfg = self.config
        cfg.data_path.mkdir(parents=True, exist_ok=True)

        self.embedder = Embedder(cfg.embedder)
        await anyio.to_thread.run_sync(self.embedder.load)

        self.backend = _make_backend(cfg)
        await self.backend.initialize()

        self.doc_store = DocumentStore(cfg.sqlite_path)
        await self.doc_store.initialize()

        self.arg_store = ArgumentStore(cfg.sqlite_path)
        await self.arg_store.initialize()

        self.pipeline = RAGPipeline(self.backend, self.embedder, self.doc_store, cfg)
        self.chunker = TextChunker(cfg.chunking)

    async def close(self) -> None:
        if self.backend:
            await self.backend.close()

    async def __aenter__(self) -> "ArgumentAnalysisEngine":
        await self.initialize()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    # ── Argument analysis ─────────────────────────────────────────────────────

    async def assess_argument(
        self,
        text: str,
        collection: str = "default",
        llm: LLMCaller | None = None,
        store_result: bool = True,
        on_progress: ProgressCallback = None,
    ) -> ArgumentAssessment:
        """Full pipeline: decompose → RAG evidence per component → assess →
        detect fallacies → synthesize overall report.

        *llm* is required. Pass ``mcp_session_caller(session)`` in an MCP
        context, or any ``async (prompt, system, max_tokens, temperature) -> str``
        callable in standalone use.
        """
        if llm is None:
            raise ValueError("assess_argument requires an LLMCaller (llm= parameter).")

        sampler = SamplingHelper(llm, self.config.sampling)

        await _progress(on_progress, 5, "Decomposing argument structure...")
        toulmin = await sampler.decompose(text)

        components_to_assess: list[tuple[str, str]] = [("claim", toulmin.claim)]
        for datum in toulmin.data:
            components_to_assess.append(("evidence", datum))
        if toulmin.warrant:
            components_to_assess.append(("reasoning", toulmin.warrant))
        if toulmin.backing:
            components_to_assess.append(("support", toulmin.backing))

        total_steps = len(components_to_assess) + 3
        step = 1
        component_assessments = []

        for comp_type, comp_text in components_to_assess:
            step += 1
            pct = int(step / total_steps * 80)
            await _progress(on_progress, pct, f"Assessing {comp_type}...")
            evidence = await self.pipeline.search(comp_text, collection, top_k=self.config.sampling.top_k_retrieval)
            assessment = await sampler.assess_component(comp_type, comp_text, evidence)
            component_assessments.append(assessment)

        await _progress(on_progress, 85, "Detecting fallacies...")
        fallacies = await sampler.detect_fallacies(text)

        await _progress(on_progress, 90, "Synthesizing assessment report...")
        summary, validity, truthfulness = await sampler.synthesize_assessment(
            text, toulmin, component_assessments, fallacies
        )

        critical_weaknesses = [ca for ca in component_assessments if ca.weakness_flag]
        kb_was_used = any(ca.knowledge_base_score is not None for ca in component_assessments)

        result = ArgumentAssessment(
            id=str(uuid.uuid4()),
            source_text=text,
            toulmin=toulmin,
            component_assessments=component_assessments,
            fallacies=fallacies,
            overall_validity_score=validity,
            overall_truthfulness_score=truthfulness,
            critical_weaknesses=critical_weaknesses,
            summary=summary,
            rag_collection_used=collection if kb_was_used else None,
        )

        if store_result:
            await self.arg_store.save_assessment(result)

        await _progress(on_progress, 100, "Done.")
        return result

    async def decompose_argument(self, text: str, llm: LLMCaller) -> ToulminComponents:
        """Extract Toulmin components only (no evidence lookup)."""
        return await SamplingHelper(llm, self.config.sampling).decompose(text)

    async def detect_fallacies(self, text: str, llm: LLMCaller) -> list[FallacyDetection]:
        """Identify logical fallacies in argument text."""
        return await SamplingHelper(llm, self.config.sampling).detect_fallacies(text)

    async def get_assessment(self, assessment_id: str) -> ArgumentAssessment | None:
        """Retrieve a previously stored assessment by ID."""
        return await self.arg_store.get_assessment(assessment_id)

    # ── RAG search ────────────────────────────────────────────────────────────

    async def search(
        self, query: str, collection: str = "default", top_k: int = 5
    ) -> list[SearchResult]:
        """Semantic search over the knowledge base. Returns ranked chunks."""
        return await self.pipeline.search(query, collection, top_k=top_k)

    async def get_document(self, doc_id: str) -> Document | None:
        """Retrieve the full text of a document by its ID."""
        return await self.doc_store.get_document(doc_id)

    async def synthesize_answer(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 5,
        llm: LLMCaller | None = None,
    ) -> str:
        """Search then synthesize a cited answer. Degrades to formatted chunks if llm is None."""
        results = await self.pipeline.search(query, collection, top_k=top_k)
        if not results:
            return "No relevant documents found in the knowledge base."
        if llm is None:
            lines = [f"[{r.rank}] (score={r.score:.3f})\n{r.chunk.text}" for r in results]
            return "\n\n---\n\n".join(lines)
        return await SamplingHelper(llm, self.config.sampling).synthesize_rag_answer(query, results)

    # ── RAG ingest ────────────────────────────────────────────────────────────

    async def ingest_text(
        self,
        text: str,
        title: str,
        collection: str = "default",
        metadata: dict | None = None,
        on_progress: ProgressCallback = None,
    ) -> dict:
        """Ingest raw text into the knowledge base."""
        await _progress(on_progress, 10, "Ingesting text...")
        result = await self._ingest_item(text, title, "inline", collection, metadata or {})
        await _progress(on_progress, 100, "Done.")
        return result

    async def ingest_file(
        self,
        path: str,
        collection: str = "default",
        title: str | None = None,
        metadata: dict | None = None,
        on_progress: ProgressCallback = None,
    ) -> dict:
        """Ingest a local file (PDF, .md, .txt, etc.) into the knowledge base."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"error": "file_not_found", "path": str(p)}

        loader = self._loader_for(p)
        text, detected_title, file_meta = await loader.load(str(p))
        await _progress(on_progress, 10, f"Ingesting {p.name}...")
        result = await self._ingest_item(
            text, title or detected_title, str(p), collection,
            {**file_meta, **(metadata or {})},
        )
        await _progress(on_progress, 100, "Done.")
        return result

    async def ingest_url(
        self,
        url: str,
        collection: str = "default",
        title: str | None = None,
        on_progress: ProgressCallback = None,
    ) -> dict:
        """Fetch and ingest a web page into the knowledge base."""
        from aa_mcp.loaders.web import WebLoader

        text, detected_title, meta = await WebLoader().load(url)
        await _progress(on_progress, 10, f"Ingesting {url[:60]}...")
        result = await self._ingest_item(text, title or detected_title, url, collection, meta)
        await _progress(on_progress, 100, "Done.")
        return result

    async def ingest_directory(
        self,
        path: str,
        collection: str = "default",
        pattern: str = "*",
        recursive: bool = False,
        skip_existing: bool = True,
        on_progress: ProgressCallback = None,
    ) -> dict:
        """Ingest all matching files in a directory.

        Args:
            pattern: Glob pattern, e.g. ``"*.pdf"`` or ``"**/*.md"``.
            recursive: Scan subdirectories recursively.
            skip_existing: Skip sources already in the collection (default True).
        """
        root = Path(path).expanduser().resolve()
        if not root.exists():
            return {"error": "directory_not_found", "path": str(root)}
        if not root.is_dir():
            return {"error": "not_a_directory", "path": str(root)}

        glob_fn = root.rglob if recursive else root.glob
        candidates = [
            p for p in glob_fn(pattern)
            if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
        ]
        if not candidates:
            return {"ingested": 0, "skipped": 0, "errors": [], "message": "No matching files found."}

        return await self._bulk_ingest_files(
            [str(p) for p in candidates], collection, skip_existing, on_progress
        )

    async def ingest_file_list(
        self,
        paths: list[str],
        collection: str = "default",
        skip_existing: bool = True,
        on_progress: ProgressCallback = None,
    ) -> dict:
        """Ingest an explicit list of file paths."""
        if not paths:
            return {"ingested": 0, "skipped": 0, "errors": [], "message": "Empty path list."}
        return await self._bulk_ingest_files(paths, collection, skip_existing, on_progress)

    async def ingest_url_list(
        self,
        urls: list[str],
        collection: str = "default",
        skip_existing: bool = True,
        on_progress: ProgressCallback = None,
    ) -> dict:
        """Fetch and ingest a list of URLs."""
        if not urls:
            return {"ingested": 0, "skipped": 0, "errors": [], "message": "Empty URL list."}

        from aa_mcp.loaders.web import WebLoader

        loader = WebLoader()
        total = len(urls)
        ingested, skipped = 0, 0
        errors: list[dict] = []
        results: list[dict] = []

        for i, url in enumerate(urls):
            pct = int((i / total) * 100)
            await _progress(on_progress, pct, f"[{i + 1}/{total}] {url[:60]}")

            if skip_existing and await self.doc_store.get_document_by_source(url, collection):
                skipped += 1
                results.append({"url": url, "status": "skipped"})
                continue

            try:
                text, title, meta = await loader.load(url)
                r = await self._ingest_item(text, title, url, collection, meta)
                if "error" in r:
                    errors.append({"url": url, "error": r["error"]})
                else:
                    ingested += 1
                    results.append({"url": url, "status": "ingested", **r})
            except Exception as exc:
                errors.append({"url": url, "error": str(exc)})

        await _progress(on_progress, 100, "Done.")
        return {"ingested": ingested, "skipped": skipped, "errors": errors, "total": total, "results": results}

    # ── Argument mapping ──────────────────────────────────────────────────────

    async def create_argument_map(self, title: str, description: str = "") -> ArgumentMap:
        now = datetime.now(timezone.utc)
        map_ = ArgumentMap(
            id=str(uuid.uuid4()), title=title,
            description=description or None, created_at=now, updated_at=now,
        )
        await self.arg_store.create_map(map_)
        return map_

    async def add_argument_node(
        self, map_id: str, text: str, node_type: str = "claim"
    ) -> ArgumentNode:
        node = ArgumentNode(id=str(uuid.uuid4()), map_id=map_id, text=text, node_type=node_type)
        await self.arg_store.add_node(node)
        return node

    async def link_arguments(
        self,
        map_id: str,
        source_id: str,
        target_id: str,
        relationship: str,
        explanation: str = "",
    ) -> ArgumentEdge:
        edge = ArgumentEdge(
            id=str(uuid.uuid4()),
            map_id=map_id,
            source_id=source_id,
            target_id=target_id,
            relationship=RelationshipType(relationship),
            explanation=explanation or None,
        )
        await self.arg_store.add_edge(edge)
        return edge

    async def get_argument_map(self, map_id: str) -> ArgumentMap | None:
        return await self.arg_store.get_map(map_id)

    async def list_argument_maps(self) -> list[dict]:
        return await self.arg_store.list_maps()

    async def delete_argument_map(self, map_id: str) -> bool:
        return await self.arg_store.delete_map(map_id)

    async def auto_map_argument(
        self,
        text: str,
        map_id: str | None = None,
        llm: LLMCaller | None = None,
        on_progress: ProgressCallback = None,
    ) -> dict:
        """Decompose argument → create nodes → infer relationships. Requires llm."""
        if llm is None:
            raise ValueError("auto_map_argument requires an LLMCaller (llm= parameter).")

        sampler = SamplingHelper(llm, self.config.sampling)

        await _progress(on_progress, 10, "Decomposing argument...")
        toulmin = await sampler.decompose(text)

        if not map_id:
            claim_preview = toulmin.claim[:60] + ("..." if len(toulmin.claim) > 60 else "")
            map_ = await self.create_argument_map(f"Map: {claim_preview}")
            map_id = map_.id

        await _progress(on_progress, 30, "Creating argument nodes...")
        node_map: dict[str, ArgumentNode] = {}

        claim_node = ArgumentNode(id=str(uuid.uuid4()), map_id=map_id, text=toulmin.claim, node_type="claim")
        await self.arg_store.add_node(claim_node)
        node_map["claim"] = claim_node

        for i, datum in enumerate(toulmin.data):
            node = ArgumentNode(id=str(uuid.uuid4()), map_id=map_id, text=datum, node_type="premise")
            await self.arg_store.add_node(node)
            node_map[f"datum_{i}"] = node

        if toulmin.warrant:
            node = ArgumentNode(id=str(uuid.uuid4()), map_id=map_id, text=toulmin.warrant, node_type="premise")
            await self.arg_store.add_node(node)
            node_map["warrant"] = node

        if toulmin.rebuttal:
            node = ArgumentNode(id=str(uuid.uuid4()), map_id=map_id, text=toulmin.rebuttal, node_type="rebuttal")
            await self.arg_store.add_node(node)
            node_map["rebuttal"] = node

        await _progress(on_progress, 60, "Inferring relationships...")
        nodes_for_inference = [{"id": n.id, "node_type": n.node_type, "text": n.text} for n in node_map.values()]
        suggested_edges = await sampler.infer_relationships(nodes_for_inference)

        node_ids = {n.id for n in node_map.values()}
        edges_created = []
        for edge_data in suggested_edges:
            src = edge_data.get("source_id")
            tgt = edge_data.get("target_id")
            rel = edge_data.get("relationship")
            if src not in node_ids or tgt not in node_ids:
                continue
            if rel not in {r.value for r in RelationshipType}:
                continue
            edge = ArgumentEdge(
                id=str(uuid.uuid4()), map_id=map_id,
                source_id=src, target_id=tgt,
                relationship=RelationshipType(rel),
                explanation=edge_data.get("explanation"),
            )
            await self.arg_store.add_edge(edge)
            edges_created.append(edge.model_dump(mode="json"))

        await _progress(on_progress, 100, "Done.")
        return {
            "map_id": map_id,
            "nodes_created": len(node_map),
            "edges_created": len(edges_created),
            "claim": toulmin.claim,
        }

    # ── Management ────────────────────────────────────────────────────────────

    async def list_documents(self, collection: str | None = None) -> list[Document]:
        return await self.doc_store.list_documents(collection)

    async def delete_document(self, doc_id: str) -> dict:
        doc = await self.doc_store.get_document(doc_id)
        if not doc:
            return {"error": "not_found", "doc_id": doc_id}
        chunk_count = await self.doc_store.delete_document(doc_id)
        await self.backend.delete_by_doc_id(doc_id, doc.collection)
        return {"deleted": True, "doc_id": doc_id, "chunks_removed": chunk_count}

    async def collection_stats(self, collection: str) -> dict:
        return await self.doc_store.collection_stats(collection)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _ingest_item(
        self,
        text: str,
        title: str,
        source: str,
        collection: str,
        metadata: dict,
    ) -> dict:
        """Core ingest: chunk → embed → store."""
        doc_id = str(uuid.uuid4())
        chunks = self.chunker.chunk(text, doc_id, collection, metadata)
        if not chunks:
            return {"error": "no_chunks_produced", "source": source}

        embeddings = await self.embedder.embed_batch([c.text for c in chunks])

        doc = Document(
            id=doc_id, title=title, source=source, collection=collection,
            full_text=text, metadata=metadata, created_at=datetime.now(timezone.utc),
            chunk_count=len(chunks),
        )
        await self.doc_store.save_document(doc)
        await self.doc_store.save_chunks(chunks)

        chunk_ids = [c.id for c in chunks]
        chroma_metas = [
            {"doc_id": doc_id, "collection": collection, "chunk_index": c.index}
            for c in chunks
        ]
        await self.backend.add_embeddings(chunk_ids, embeddings, collection, chroma_metas)
        return {
            "doc_id": doc_id, "title": title, "source": source,
            "collection": collection, "chunks": len(chunks),
        }

    def _loader_for(self, p: Path):
        if p.suffix.lower() == ".pdf":
            from aa_mcp.loaders.pdf import PDFLoader
            return PDFLoader()
        from aa_mcp.loaders.text import TextLoader
        return TextLoader()

    async def _bulk_ingest_files(
        self,
        paths: list[str],
        collection: str,
        skip_existing: bool,
        on_progress: ProgressCallback,
    ) -> dict:
        total = len(paths)
        ingested, skipped = 0, 0
        errors: list[dict] = []
        results: list[dict] = []

        for i, raw_path in enumerate(paths):
            p = Path(raw_path).expanduser().resolve()
            pct = int((i / total) * 100)
            await _progress(on_progress, pct, f"[{i + 1}/{total}] {p.name}")

            if not p.exists():
                errors.append({"path": str(p), "error": "file_not_found"})
                continue

            if skip_existing and await self.doc_store.get_document_by_source(str(p), collection):
                skipped += 1
                results.append({"path": str(p), "status": "skipped"})
                continue

            try:
                text, title, meta = await self._loader_for(p).load(str(p))
                r = await self._ingest_item(text, title, str(p), collection, meta)
                if "error" in r:
                    errors.append({"path": str(p), "error": r["error"]})
                else:
                    ingested += 1
                    results.append({"path": str(p), "status": "ingested", **r})
            except Exception as exc:
                errors.append({"path": str(p), "error": str(exc)})

        await _progress(on_progress, 100, "Done.")
        return {"ingested": ingested, "skipped": skipped, "errors": errors, "total": total, "results": results}
