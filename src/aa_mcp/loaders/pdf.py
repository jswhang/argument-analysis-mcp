"""PDF loader using pypdf."""

from __future__ import annotations

from pathlib import Path

import anyio

from aa_mcp.loaders.base import DocumentLoader


class PDFLoader(DocumentLoader):
    async def load(self, source: str) -> tuple[str, str, dict]:
        path = Path(source)

        def _sync_load() -> tuple[str, int]:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages), len(reader.pages)

        text, page_count = await anyio.to_thread.run_sync(_sync_load)
        title = path.stem.replace("_", " ").replace("-", " ").title()
        return text, title, {"file_path": str(path), "page_count": page_count}
