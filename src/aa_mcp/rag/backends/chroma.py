"""ChromaDB vector backend — persistent, embedded, zero-config."""

from __future__ import annotations

from pathlib import Path

import anyio

from aa_mcp.rag.backends.base import VectorBackend


class ChromaBackend(VectorBackend):
    def __init__(self, data_path: Path) -> None:
        self._data_path = data_path
        self._client = None

    async def initialize(self) -> None:
        self._data_path.mkdir(parents=True, exist_ok=True)
        await anyio.to_thread.run_sync(self._sync_init)

    def _sync_init(self) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(self._data_path))

    def _get_or_create_collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_embeddings(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        collection: str,
        metadatas: list[dict] | None = None,
    ) -> None:
        metas = metadatas or [{} for _ in chunk_ids]

        def _sync():
            col = self._get_or_create_collection(collection)
            col.add(ids=chunk_ids, embeddings=embeddings, metadatas=metas)

        await anyio.to_thread.run_sync(_sync)

    async def search(
        self,
        query_embedding: list[float],
        collection: str,
        top_k: int,
        filter: dict | None = None,
    ) -> list[tuple[str, float]]:
        def _sync():
            try:
                col = self._client.get_collection(name=collection)
            except Exception:
                return []
            kwargs: dict = {"query_embeddings": [query_embedding], "n_results": top_k}
            if filter:
                kwargs["where"] = filter
            results = col.query(**kwargs)
            ids = results["ids"][0] if results["ids"] else []
            distances = results["distances"][0] if results["distances"] else []
            # ChromaDB cosine distance: 0 = identical, 2 = opposite. Convert to similarity.
            return [(id_, 1.0 - (dist / 2.0)) for id_, dist in zip(ids, distances)]

        return await anyio.to_thread.run_sync(_sync)

    async def delete_by_doc_id(self, doc_id: str, collection: str) -> None:
        def _sync():
            try:
                col = self._client.get_collection(name=collection)
                col.delete(where={"doc_id": doc_id})
            except Exception:
                pass

        await anyio.to_thread.run_sync(_sync)

    async def close(self) -> None:
        pass  # PersistentClient auto-flushes
