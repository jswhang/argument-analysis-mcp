"""RAG pipeline: embed query → vector search → fetch full chunks from SQLite."""

from __future__ import annotations

from aa_mcp.config import ServerConfig
from aa_mcp.models.rag import SearchResult
from aa_mcp.rag.backends.base import VectorBackend
from aa_mcp.rag.embedder import Embedder
from aa_mcp.store.document_store import DocumentStore


class RAGPipeline:
    def __init__(self, backend: VectorBackend, embedder: Embedder, doc_store: DocumentStore, config: ServerConfig) -> None:
        self._backend = backend
        self._embedder = embedder
        self._doc_store = doc_store
        self._config = config

    async def search(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        k = top_k or self._config.sampling.top_k_final
        retrieve_k = max(k, self._config.sampling.top_k_retrieval)

        embedding = await self._embedder.embed(query)
        id_score_pairs = await self._backend.search(embedding, collection, top_k=retrieve_k)

        if not id_score_pairs:
            return []

        chunk_ids = [id_ for id_, _ in id_score_pairs]
        score_map = {id_: score for id_, score in id_score_pairs}

        chunks = await self._doc_store.get_chunks_by_ids(chunk_ids)

        results = [
            SearchResult(chunk=chunk, score=score_map.get(chunk.id, 0.0), rank=i + 1)
            for i, chunk in enumerate(
                sorted(chunks, key=lambda c: score_map.get(c.id, 0.0), reverse=True)
            )
        ]
        return results[:k]
