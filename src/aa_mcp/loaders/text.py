"""Plain text and Markdown loader."""

from __future__ import annotations

from pathlib import Path

import anyio

from aa_mcp.loaders.base import DocumentLoader


class TextLoader(DocumentLoader):
    async def load(self, source: str) -> tuple[str, str, dict]:
        path = Path(source)
        text = await anyio.Path(path).read_text(encoding="utf-8", errors="replace")
        title = path.stem.replace("_", " ").replace("-", " ").title()
        return text, title, {"file_path": str(path), "extension": path.suffix}
