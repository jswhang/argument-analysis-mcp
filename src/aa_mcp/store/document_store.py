"""SQLite store for full document text and chunks — the source of truth for all text content."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from aa_mcp.models.rag import Chunk, Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL,
    collection  TEXT NOT NULL DEFAULT 'default',
    full_text   TEXT NOT NULL,
    mime_type   TEXT NOT NULL DEFAULT 'text/plain',
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection  TEXT NOT NULL DEFAULT 'default',
    text        TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    start_char  INTEGER NOT NULL,
    end_char    INTEGER NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection);
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
"""


class DocumentStore:
    """Manages full document and chunk storage in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    async def save_document(self, doc: Document) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO documents
                    (id, title, source, collection, full_text, mime_type, metadata, created_at, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.id,
                    doc.title,
                    doc.source,
                    doc.collection,
                    doc.full_text,
                    doc.mime_type,
                    json.dumps(doc.metadata),
                    doc.created_at.isoformat(),
                    doc.chunk_count,
                ),
            )
            await db.commit()

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO chunks (id, doc_id, collection, text, idx, start_char, end_char, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.id,
                        c.doc_id,
                        c.collection,
                        c.text,
                        c.index,
                        c.start_char,
                        c.end_char,
                        json.dumps(c.metadata),
                    )
                    for c in chunks
                ],
            )
            await db.commit()

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
            ) as cursor:
                rows = await cursor.fetchall()
        return [_row_to_chunk(r) for r in rows]

    async def get_document(self, doc_id: str) -> Document | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)) as cursor:
                row = await cursor.fetchone()
        return _row_to_document(row) if row else None

    async def list_documents(self, collection: str | None = None) -> list[Document]:
        query = "SELECT * FROM documents"
        params: list[str] = []
        if collection:
            query += " WHERE collection = ?"
            params.append(collection)
        query += " ORDER BY created_at DESC"
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
        return [_row_to_document(r) for r in rows]

    async def delete_document(self, doc_id: str) -> int:
        """Delete document and its chunks. Returns number of chunks deleted."""
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
            ) as cursor:
                row = await cursor.fetchone()
                chunk_count = row[0] if row else 0
            await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            await db.commit()
        return chunk_count

    async def get_document_by_source(self, source: str, collection: str) -> str | None:
        """Return document ID if source already exists in collection, else None."""
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT id FROM documents WHERE source = ? AND collection = ?",
                (source, collection),
            ) as cursor:
                row = await cursor.fetchone()
        return row[0] if row else None

    async def list_collections(self) -> list[str]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT DISTINCT collection FROM documents ORDER BY collection"
            ) as cursor:
                rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def collection_stats(self, collection: str) -> dict:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM documents WHERE collection = ?", (collection,)
            ) as cursor:
                doc_count = (await cursor.fetchone())[0]
            async with db.execute(
                "SELECT COUNT(*) FROM chunks WHERE collection = ?", (collection,)
            ) as cursor:
                chunk_count = (await cursor.fetchone())[0]
        return {"collection": collection, "documents": doc_count, "chunks": chunk_count}


def _row_to_document(row: aiosqlite.Row) -> Document:
    from datetime import datetime

    return Document(
        id=row["id"],
        title=row["title"],
        source=row["source"],
        collection=row["collection"],
        full_text=row["full_text"],
        mime_type=row["mime_type"],
        metadata=json.loads(row["metadata"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        chunk_count=row["chunk_count"],
    )


def _row_to_chunk(row: aiosqlite.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        collection=row["collection"],
        text=row["text"],
        index=row["idx"],
        start_char=row["start_char"],
        end_char=row["end_char"],
        metadata=json.loads(row["metadata"]),
    )
