"""Server configuration via pydantic-settings (AA_MCP_ env prefix + .env file)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedderConfig(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    cache_dir: str | None = None


class RAGConfig(BaseModel):
    backend: Literal["chroma", "sqlite-vec", "lancedb"] = "chroma"
    default_collection: str = "default"
    distance_metric: Literal["cosine", "l2", "ip"] = "cosine"


class ChunkingConfig(BaseModel):
    strategy: Literal["recursive", "sentence", "fixed"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50


class SamplingConfig(BaseModel):
    enabled: bool = True
    max_tokens: int = 2048
    max_tokens_decompose: int = 1024
    max_tokens_assess: int = 1024
    max_tokens_fallacy: int = 1024
    max_tokens_synthesize: int = 2048
    max_tokens_hyde: int = 512
    temperature_analysis: float = 0.1
    temperature_synthesis: float = 0.3
    temperature_hyde: float = 0.7
    top_k_retrieval: int = 10
    top_k_final: int = 5


class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AA_MCP_",
        env_file=".env",
        env_nested_delimiter="__",
    )

    data_dir: str = "~/.aa-mcp/data"
    log_level: str = "INFO"

    embedder: EmbedderConfig = EmbedderConfig()
    rag: RAGConfig = RAGConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    sampling: SamplingConfig = SamplingConfig()

    @field_validator("data_dir", mode="before")
    @classmethod
    def expand_data_dir(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def sqlite_path(self) -> Path:
        return self.data_path / "aa_mcp.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_path / "chroma"
