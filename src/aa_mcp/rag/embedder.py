"""Sentence-transformer embedder with anyio thread dispatch (sync lib → async context)."""

from __future__ import annotations

import anyio

from aa_mcp.config import EmbedderConfig


class Embedder:
    def __init__(self, config: EmbedderConfig) -> None:
        self._config = config
        self._model = None
        self._dim: int = 0

    def load(self) -> None:
        """Synchronous model load — call via anyio.to_thread.run_sync() in async context."""
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self._config.model,
            device=self._config.device,
            cache_folder=self._config.cache_dir,
        )
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        if not self._model:
            raise RuntimeError("Embedder not loaded. Call load() first.")
        return self._dim

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._model:
            raise RuntimeError("Embedder not loaded.")
        result = await anyio.to_thread.run_sync(
            lambda: self._model.encode(
                texts,
                batch_size=self._config.batch_size,
                show_progress_bar=False,
            ).tolist()
        )
        return result
