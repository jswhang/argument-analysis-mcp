"""Base DocumentLoader interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class DocumentLoader(ABC):
    @abstractmethod
    async def load(self, source: str) -> tuple[str, str, dict]:
        """Load a document. Returns (text, title, metadata)."""
        ...
