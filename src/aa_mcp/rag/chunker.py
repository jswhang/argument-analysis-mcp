"""Text chunker with recursive, sentence, and fixed strategies."""

from __future__ import annotations

import re
import uuid

from aa_mcp.config import ChunkingConfig
from aa_mcp.models.rag import Chunk


class TextChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    def chunk(self, text: str, doc_id: str, collection: str, metadata: dict | None = None) -> list[Chunk]:
        strategy = self._config.strategy
        if strategy == "recursive":
            spans = self._recursive_split(text)
        elif strategy == "sentence":
            spans = self._sentence_split(text)
        else:
            spans = self._fixed_split(text)

        chunks = []
        for i, (start, end) in enumerate(spans):
            chunk_text = text[start:end].strip()
            if len(chunk_text) < self._config.min_chunk_size:
                continue
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    collection=collection,
                    text=chunk_text,
                    index=i,
                    start_char=start,
                    end_char=end,
                    metadata=metadata or {},
                )
            )
        return chunks

    def _recursive_split(self, text: str) -> list[tuple[int, int]]:
        """Split on paragraph breaks, then newlines, then sentences, then spaces."""
        separators = ["\n\n", "\n", ". ", " "]
        return self._split_recursive(text, 0, separators)

    def _split_recursive(self, text: str, offset: int, separators: list[str]) -> list[tuple[int, int]]:
        size = self._config.chunk_size
        overlap = self._config.chunk_overlap

        if len(text) <= size:
            return [(offset, offset + len(text))]

        sep = separators[0] if separators else " "
        remaining_seps = separators[1:] if len(separators) > 1 else []

        parts = text.split(sep)
        spans = []
        current_start = 0
        current_len = 0

        for part in parts:
            part_len = len(part) + len(sep)
            if current_len + part_len > size and current_len > 0:
                end = offset + current_start + current_len
                spans.append((offset + current_start, end))
                # overlap: step back by overlap chars
                step_back = 0
                while step_back < len(parts) and step_back * len(sep) < overlap:
                    step_back += 1
                current_start = max(0, current_start + current_len - overlap)
                current_len = 0
            current_len += part_len

        if current_len > 0:
            spans.append((offset + current_start, offset + current_start + current_len))

        return spans

    def _sentence_split(self, text: str) -> list[tuple[int, int]]:
        sentences = list(re.finditer(r"[^.!?]+[.!?]+", text))
        if not sentences:
            return self._fixed_split(text)

        spans = []
        current_start = 0
        current_end = 0

        for m in sentences:
            if m.end() - current_start > self._config.chunk_size and current_end > current_start:
                spans.append((current_start, current_end))
                current_start = max(current_start, current_end - self._config.chunk_overlap)
            current_end = m.end()

        if current_end > current_start:
            spans.append((current_start, current_end))

        return spans

    def _fixed_split(self, text: str) -> list[tuple[int, int]]:
        size = self._config.chunk_size
        overlap = self._config.chunk_overlap
        spans = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            spans.append((start, end))
            start += size - overlap
        return spans
