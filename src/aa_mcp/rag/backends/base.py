"""Abstract base class for vector backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class VectorBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def add_embeddings(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        collection: str,
        metadatas: list[dict] | None = None,
    ) -> None: ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        collection: str,
        top_k: int,
        filter: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) sorted by descending similarity."""
        ...

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str, collection: str) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...
