# Argument Analysis MCP Server — Specification

**Version:** 0.1.0
**Last updated:** 2026-03-31
**Project path:** `/home/jw/Projects/argument-analysis-mcp/`
**Package:** `aa_mcp`
**Distribution name:** `argument-analysis-mcp`
**Entry point script:** `aa-mcp-server`

---

## 1. Purpose and Intent

### 1.1 Problem Statement

Large language models produce confident-sounding arguments that may contain logical errors, unsupported claims, misleading framings, or factual inaccuracies that contradict domain-specific knowledge. There is currently no standard way to submit LLM output for structured critique in the same environment where the LLM is being used.

### 1.2 Solution

An MCP (Model Context Protocol) server that acts as an argument analysis and fact-checking co-processor. The user (or any LLM client) can submit text — often the output of an LLM interaction — and receive a structured, evidence-grounded critique.

The server performs four things in sequence:
1. **Structural decomposition** — breaks the argument into its logical components using the Toulmin model
2. **Evidence retrieval** — searches a user-maintained knowledge base for domain-specific information relevant to each component
3. **Per-component assessment** — evaluates logical validity (is the reasoning sound?) and factual accuracy (does the claim hold up against known evidence?) for each component independently
4. **Synthesis** — produces a final scored report, calling out especially weak components by name

### 1.3 Design Philosophy

- **Local-first.** There is no central server. The end user runs everything on their own machine. No API keys are required for the core functionality. The embedding model and all data remain on-device.
- **Sampling-first.** The server does not bundle a local LLM. Instead, it uses the MCP `sampling/createMessage` protocol to ask the same LLM that invoked it to perform analytical tasks on its behalf. This means the quality of the analysis scales with the quality of the connected client model.
- **Graceful degradation.** Every tool that depends on sampling checks for the capability at runtime and returns a useful error (not a crash) when sampling is unavailable. RAG search works without sampling at all.
- **Two-store architecture.** SQLite is the source of truth for all text. ChromaDB is a pure vector index that holds embeddings and references back to SQLite. No text lives in ChromaDB.
- **Pluggable vector backend.** The `VectorBackend` abstract class means ChromaDB can be swapped for `sqlite-vec` or LanceDB with a single config change.

### 1.4 Target User

A developer or researcher who:
- Uses Claude Desktop (or another MCP-capable client)
- Wants to critically evaluate arguments made by an LLM or by any text source
- Maintains a personal knowledge base of domain documents (papers, articles, references) that serve as the ground truth for factual assessment
- Runs everything locally and controls their own data

---

## 2. Technical Stack

| Layer | Library / Tool | Version | Notes |
|-------|---------------|---------|-------|
| Language | Python | ≥3.12 | Uses `from __future__ import annotations`, match-style typing |
| MCP framework | `mcp[cli]` | ≥1.0.0 | FastMCP from the official Anthropic MCP SDK |
| MCP transport | stdio | — | Client spawns server as subprocess; communicates via stdin/stdout pipes |
| Async runtime | `anyio` | ≥4.7 | All I/O is async; sync libraries dispatched via `anyio.to_thread.run_sync()` |
| Data validation | `pydantic` | ≥2.9 | All models are Pydantic v2 `BaseModel` |
| Configuration | `pydantic-settings` | ≥2.6 | `BaseSettings` with `AA_MCP_` env prefix, `.env` file, nested delimiter `__` |
| Full-text store | `aiosqlite` | ≥0.20 | Async SQLite wrapper; single `.db` file for all persistent state |
| Vector index | `chromadb` | ≥0.5.0 | `PersistentClient`; HNSW index with cosine distance |
| Embeddings | `sentence-transformers` | ≥3.0 | Default model: `all-MiniLM-L6-v2`; loaded once at server startup |
| PDF loading | `pypdf` | ≥5.0 | Text extraction from PDF files |
| Web loading | `httpx` + `beautifulsoup4` | ≥0.27 / ≥4.12 | Async HTTP fetch, HTML stripping |
| Build system | `hatchling` | — | `src/` layout; `pyproject.toml`-based |

### 2.1 Optional Extras

| Extra | Package | Activates |
|-------|---------|-----------|
| `sqlite-vec` | `sqlite-vec` ≥0.1.0 | `SqliteVecBackend` (single-file vector + text) |
| `lancedb` | `lancedb` ≥0.14.0, `pyarrow` ≥18.0 | `LanceBackend` (columnar, better for large corpora) |
| `dev` | pytest, pytest-asyncio, pytest-cov, ruff | Development and testing |

---

## 3. Architecture

### 3.1 Component Diagram

```
MCP Client (e.g. Claude Desktop)
        │  stdio (JSON-RPC 2.0)
        ▼
┌───────────────────────────────────────────────────────┐
│  aa_mcp server process                                │
│                                                       │
│  FastMCP app                                          │
│  ├── tools/analysis.py    ← primary pipeline          │
│  ├── tools/mapping.py     ← argument graph            │
│  ├── tools/rag_ingest.py  ← knowledge base input      │
│  ├── tools/rag_search.py  ← knowledge base query      │
│  ├── tools/manage.py      ← CRUD operations           │
│  ├── resources/           ← read-only URI endpoints   │
│  └── prompts/             ← reusable prompt templates │
│                                                       │
│  AppState (lifespan-scoped singleton)                 │
│  ├── Embedder             (sentence-transformers)     │
│  ├── VectorBackend        (ChromaDB or alternative)   │
│  ├── DocumentStore        (SQLite)                    │
│  ├── ArgumentStore        (SQLite)                    │
│  ├── RAGPipeline          (orchestrates above)        │
│  └── TextChunker                                      │
│                                                       │
│  SamplingHelper                                       │
│  └── calls back to MCP client via create_message()   │
└───────────────────────────────────────────────────────┘
        │                        │
        ▼                        ▼
  aa_mcp.db (SQLite)      chroma/ (ChromaDB)
  • documents             • embeddings (float[])
  • chunks                • chunk_id references only
  • assessments           • collection HNSW indexes
  • argument_maps
  • argument_nodes
  • argument_edges
```

### 3.2 Startup Sequence

The `app_lifespan` async context manager in `lifespan.py` runs once at server start:

1. `ServerConfig()` — reads config from environment / `.env`
2. `data_path.mkdir()` — creates storage directory if absent
3. `Embedder.load()` — loads sentence-transformers model via `anyio.to_thread.run_sync()` (blocks thread, not event loop; downloads model on first run ~22MB)
4. `VectorBackend.initialize()` — opens/creates ChromaDB `PersistentClient`
5. `DocumentStore.initialize()` — runs SQLite schema migrations (`CREATE TABLE IF NOT EXISTS`)
6. `ArgumentStore.initialize()` — runs SQLite schema migrations
7. Constructs `RAGPipeline` and `TextChunker`
8. Yields `AppState` dataclass — all tools access shared state via `ctx.request_context.lifespan_context`
9. On shutdown: `backend.close()` (ChromaDB auto-flushes; no explicit action needed)

### 3.3 MCP Sampling Flow

MCP sampling is the mechanism by which the server calls back to the client's LLM. This is the core mechanism that powers all AI tasks.

```
Tool invoked by LLM client
        │
        ▼
has_sampling_capability(ctx.session)?
        │
        ├─ No  → return {"error": "sampling_not_supported", ...}
        │
        └─ Yes → SamplingHelper._call(prompt, system, max_tokens)
                        │
                        ▼
               ctx.session.create_message(
                   messages=[SamplingMessage(...)],
                   max_tokens=...,
                   system_prompt=...
               )
                        │
                  MCP protocol: sampling/createMessage
                        │
                        ▼
               Client presents to its LLM
               (with human-in-the-loop approval per MCP spec)
                        │
                        ▼
               LLM response returned to server
                        │
                        ▼
               SamplingHelper parses JSON response
               (strips markdown fences, finds first { or [)
```

**Capability detection:** `has_sampling_capability(session)` checks `session.client_params.capabilities.sampling is not None`. This is a required guard before any `create_message()` call.

---

## 4. Storage Design

### 4.1 Two-Store Principle

There are two stores with strictly separate responsibilities:

| Store | Technology | Responsibility |
|-------|-----------|---------------|
| **DocumentStore / ArgumentStore** | SQLite (`aiosqlite`) | Source of truth for **all text**. Full article content, chunk text, assessment results, argument graphs. |
| **VectorBackend** (ChromaDB) | ChromaDB `PersistentClient` | Vector index **only**. Stores floating-point embeddings and `chunk_id` → collection metadata. Never stores text. |

**On search:** ChromaDB returns `chunk_id` strings and similarity scores. The pipeline immediately fetches full chunk text from SQLite using those IDs. The LLM never receives raw embeddings.

**Why:** This separation means text is always accessible without the vector index (e.g. for export, backup, schema evolution), and the vector index can be rebuilt from scratch by re-embedding the SQLite chunks.

### 4.2 SQLite Schema

**File:** `{AA_MCP_DATA_DIR}/aa_mcp.db`

Both `DocumentStore` and `ArgumentStore` connect to the same file. Schema is idempotent (`CREATE TABLE IF NOT EXISTS`).

#### Table: `documents`
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `title` | TEXT | Display title |
| `source` | TEXT | File path, URL, or `"inline"` |
| `collection` | TEXT | Logical grouping (default: `"default"`) |
| `full_text` | TEXT | Complete raw text of the document |
| `mime_type` | TEXT | e.g. `"text/plain"`, `"application/pdf"` |
| `metadata` | TEXT | JSON blob of arbitrary key-value pairs |
| `created_at` | TEXT | ISO 8601 timestamp |
| `chunk_count` | INTEGER | Number of chunks produced during ingest |

#### Table: `chunks`
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID (also used as ChromaDB item ID) |
| `doc_id` | TEXT FK | References `documents.id` ON DELETE CASCADE |
| `collection` | TEXT | Copied from parent document |
| `text` | TEXT | Full text of this chunk |
| `idx` | INTEGER | Position within the document (0-based) |
| `start_char` | INTEGER | Character offset in `full_text` |
| `end_char` | INTEGER | Character offset in `full_text` |
| `metadata` | TEXT | JSON blob |

Indexes: `idx_chunks_doc_id`, `idx_chunks_collection`, `idx_documents_collection`

#### Table: `assessments`
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `source_text` | TEXT | The original argument text submitted |
| `toulmin_json` | TEXT | JSON-serialized `ToulminComponents` |
| `component_assessments_json` | TEXT | JSON array of `ComponentAssessment` objects |
| `fallacies_json` | TEXT | JSON array of `FallacyDetection` objects |
| `overall_validity_score` | REAL | 0.0–1.0 |
| `overall_truthfulness_score` | REAL | 0.0–1.0 |
| `summary` | TEXT | Prose assessment |
| `rag_collection_used` | TEXT | Which RAG collection was queried |
| `metadata_json` | TEXT | JSON blob |
| `created_at` | TEXT | ISO 8601 |

#### Table: `argument_maps`
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `title` | TEXT | Map title |
| `description` | TEXT | Optional description |
| `created_at` | TEXT | ISO 8601 |
| `updated_at` | TEXT | ISO 8601, updated on node/edge add |

#### Table: `argument_nodes`
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `map_id` | TEXT FK | References `argument_maps.id` ON DELETE CASCADE |
| `text` | TEXT | The argument/claim text of this node |
| `node_type` | TEXT | `claim \| premise \| conclusion \| evidence \| rebuttal` |
| `assessment_id` | TEXT | Optional FK to `assessments.id` |
| `metadata_json` | TEXT | JSON blob |

#### Table: `argument_edges`
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `map_id` | TEXT FK | References `argument_maps.id` ON DELETE CASCADE |
| `source_id` | TEXT FK | References `argument_nodes.id` ON DELETE CASCADE |
| `target_id` | TEXT FK | References `argument_nodes.id` ON DELETE CASCADE |
| `relationship` | TEXT | `supports \| attacks \| qualifies \| elaborates` |
| `explanation` | TEXT | Optional one-sentence explanation |

Indexes: `idx_nodes_map_id`, `idx_edges_map_id`

### 4.3 ChromaDB Schema

**Directory:** `{AA_MCP_DATA_DIR}/chroma/`

Collections are created on demand via `get_or_create_collection()`. Each collection corresponds to a user-defined name (default: `"default"`).

Each item stored in ChromaDB:
- **ID:** the `chunk_id` UUID (matches `chunks.id` in SQLite)
- **Embedding:** `float[]` of dimension determined by the embedding model (384 for `all-MiniLM-L6-v2`)
- **Metadata:** `{"doc_id": str, "collection": str, "chunk_index": int}` — minimal payload, no text
- **Distance metric:** cosine (configured as `"hnsw:space": "cosine"`)

**Score conversion:** ChromaDB returns cosine distance in range [0, 2] where 0 = identical. This is converted to similarity score: `score = 1.0 - (distance / 2.0)` → range [0.0, 1.0].

---

## 5. Data Models

All models are in `src/aa_mcp/models/`. All use Pydantic v2 `BaseModel`. Datetimes are timezone-aware UTC.

### 5.1 Argument Models (`models/argument.py`)

```python
class ToulminComponents(BaseModel):
    claim: str                          # Main assertion
    data: list[str]                     # Supporting evidence/facts stated
    warrant: str                        # Reasoning connecting data to claim
    backing: str | None                 # Support for the warrant (optional)
    qualifier: str | None               # Certainty level / limitations (optional)
    rebuttal: str | None                # Anticipated counter-arguments (optional)

class FallacyDetection(BaseModel):
    fallacy_type: str                   # snake_case: "ad_hominem", "straw_man", etc.
    excerpt: str                        # The problematic passage from the text
    explanation: str                    # Why this is a fallacy
    severity: Literal["low", "medium", "high"]

class ComponentAssessment(BaseModel):
    component_type: Literal["claim", "datum", "warrant", "backing", "qualifier", "rebuttal"]
    text: str
    logical_validity_score: float       # 0.0–1.0, constrained by ge/le
    factual_accuracy_score: float       # 0.0–1.0; 0.5 when no RAG evidence available
    supporting_evidence: list[str]      # Short excerpts from RAG chunks
    contradicting_evidence: list[str]   # Short excerpts from RAG chunks
    weakness_flag: bool                 # True when score < 0.4 or directly contradicted
    weakness_explanation: str | None    # Required explanation when weakness_flag=True

class ArgumentAssessment(BaseModel):
    id: str                             # UUID
    source_text: str                    # The original argument text
    toulmin: ToulminComponents
    component_assessments: list[ComponentAssessment]
    fallacies: list[FallacyDetection]
    overall_validity_score: float       # 0.0–1.0
    overall_truthfulness_score: float   # 0.0–1.0
    critical_weaknesses: list[ComponentAssessment]  # Subset where weakness_flag=True
    summary: str                        # Prose verdict
    rag_collection_used: str | None
    created_at: datetime
    metadata: dict[str, Any]
```

### 5.2 Mapping Models (`models/mapping.py`)

```python
class RelationshipType(str, Enum):
    SUPPORTS = "supports"
    ATTACKS = "attacks"
    QUALIFIES = "qualifies"
    ELABORATES = "elaborates"

class ArgumentNode(BaseModel):
    id: str
    map_id: str
    text: str
    node_type: Literal["claim", "premise", "conclusion", "evidence", "rebuttal"]
    assessment_id: str | None
    metadata: dict[str, Any]

class ArgumentEdge(BaseModel):
    id: str
    map_id: str
    source_id: str
    target_id: str
    relationship: RelationshipType
    explanation: str | None

class ArgumentMap(BaseModel):
    id: str
    title: str
    description: str | None
    nodes: list[ArgumentNode]
    edges: list[ArgumentEdge]
    created_at: datetime
    updated_at: datetime
```

### 5.3 RAG Models (`models/rag.py`)

```python
class Document(BaseModel):
    id: str                             # UUID
    title: str
    source: str                         # file path, URL, or "inline"
    collection: str
    full_text: str                      # Complete document text (SQLite only)
    mime_type: str
    metadata: dict[str, Any]
    created_at: datetime
    chunk_count: int

class Chunk(BaseModel):
    id: str                             # UUID (also ChromaDB item ID)
    doc_id: str
    collection: str
    text: str                           # Full chunk text (SQLite only)
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any]

class SearchResult(BaseModel):
    chunk: Chunk                        # Full text from SQLite
    score: float                        # Similarity score 0.0–1.0 from ChromaDB
    rank: int
    reranked: bool
```

---

## 6. Configuration

**Class:** `ServerConfig` in `src/aa_mcp/config.py`
**Base:** `pydantic_settings.BaseSettings`
**Env prefix:** `AA_MCP_`
**Env file:** `.env` (in working directory)
**Nested delimiter:** `__` (double underscore)

### 6.1 All Configuration Keys

| Env variable | Default | Type | Description |
|---|---|---|---|
| `AA_MCP_DATA_DIR` | `~/.aa-mcp/data` | str | Root directory for SQLite and ChromaDB. Tilde-expanded at startup. |
| `AA_MCP_LOG_LEVEL` | `INFO` | str | Python logging level |
| `AA_MCP_EMBEDDER__MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | str | HuggingFace model ID or local path |
| `AA_MCP_EMBEDDER__DEVICE` | `cpu` | str | `cpu`, `cuda`, `mps` |
| `AA_MCP_EMBEDDER__BATCH_SIZE` | `32` | int | Embedding batch size |
| `AA_MCP_EMBEDDER__CACHE_DIR` | `None` | str\|null | Override HuggingFace cache dir |
| `AA_MCP_RAG__BACKEND` | `chroma` | str | `chroma`, `sqlite-vec`, `lancedb` |
| `AA_MCP_RAG__DEFAULT_COLLECTION` | `default` | str | Collection used when none specified |
| `AA_MCP_RAG__DISTANCE_METRIC` | `cosine` | str | `cosine`, `l2`, `ip` |
| `AA_MCP_CHUNKING__STRATEGY` | `recursive` | str | `recursive`, `sentence`, `fixed` |
| `AA_MCP_CHUNKING__CHUNK_SIZE` | `512` | int | Target chunk size in characters |
| `AA_MCP_CHUNKING__CHUNK_OVERLAP` | `64` | int | Overlap between adjacent chunks |
| `AA_MCP_CHUNKING__MIN_CHUNK_SIZE` | `50` | int | Chunks shorter than this are discarded |
| `AA_MCP_SAMPLING__ENABLED` | `true` | bool | Master switch for sampling calls |
| `AA_MCP_SAMPLING__MAX_TOKENS` | `2048` | int | Default max tokens |
| `AA_MCP_SAMPLING__MAX_TOKENS_DECOMPOSE` | `1024` | int | For Toulmin decomposition call |
| `AA_MCP_SAMPLING__MAX_TOKENS_ASSESS` | `1024` | int | Per-component assessment call |
| `AA_MCP_SAMPLING__MAX_TOKENS_FALLACY` | `1024` | int | Fallacy detection call |
| `AA_MCP_SAMPLING__MAX_TOKENS_SYNTHESIZE` | `2048` | int | Final synthesis / RAG answer call |
| `AA_MCP_SAMPLING__MAX_TOKENS_HYDE` | `512` | int | HyDE query expansion (unused by default) |
| `AA_MCP_SAMPLING__TEMPERATURE_ANALYSIS` | `0.1` | float | Low temp for deterministic analysis |
| `AA_MCP_SAMPLING__TEMPERATURE_SYNTHESIS` | `0.3` | float | Moderate temp for prose generation |
| `AA_MCP_SAMPLING__TEMPERATURE_HYDE` | `0.7` | float | Higher temp for HyDE diversity |
| `AA_MCP_SAMPLING__TOP_K_RETRIEVAL` | `10` | int | Chunks fetched from ChromaDB per query |
| `AA_MCP_SAMPLING__TOP_K_FINAL` | `5` | int | Chunks returned after pipeline |

### 6.2 Derived Properties

```python
config.data_path   → Path(data_dir)
config.sqlite_path → data_path / "aa_mcp.db"
config.chroma_path → data_path / "chroma"
```

---

## 7. MCP Tools — Complete Reference

All tools are async Python functions decorated with `mcp.tool()` in `server.py`. They receive a `Context` object as their last parameter. `AppState` is retrieved via `ctx.request_context.lifespan_context`.

### 7.1 Analysis Tools (`tools/analysis.py`)

---

#### `assess_argument`

**The primary tool.** Runs the full assessment pipeline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | The argument text to assess |
| `collection` | str | `"default"` | RAG collection to query for evidence |
| `store_result` | bool | `True` | Whether to persist the assessment to SQLite |

**Pipeline:**
1. Check sampling capability — return error if absent
2. `SamplingHelper.decompose(text)` → `ToulminComponents`
3. Build component list: `[("claim", claim), ("datum", d) for d in data, ("warrant", warrant), ("backing", backing)]` — qualifier and rebuttal are not assessed individually
4. For each component: `RAGPipeline.search(component_text, collection, top_k=top_k_retrieval)` → `SamplingHelper.assess_component(type, text, results)` → `ComponentAssessment`
5. `SamplingHelper.detect_fallacies(text)` → `list[FallacyDetection]`
6. `SamplingHelper.synthesize_assessment(text, toulmin, components, fallacies)` → `(summary, validity_score, truthfulness_score)`
7. Filter `critical_weaknesses = [ca for ca in components if ca.weakness_flag]`
8. Construct and optionally persist `ArgumentAssessment`
9. Return `_format_assessment()` dict

**Return format (dict):**
```json
{
  "id": "uuid",
  "overall_validity_score": 0.0–1.0,
  "overall_truthfulness_score": 0.0–1.0,
  "summary": "2-4 sentence prose verdict",
  "rag_collection_used": "default",
  "toulmin": { "claim": "...", "data": [...], "warrant": "...", ... },
  "fallacies": [ { "fallacy_type": "...", "excerpt": "...", "explanation": "...", "severity": "low|medium|high" } ],
  "component_assessments": [ { "component_type": "...", "text": "...", "logical_validity_score": 0.0, ... } ],
  "⚠️ CRITICAL_WEAKNESSES": [   ← only present when weaknesses exist
    {
      "component": "claim|datum|warrant|backing",
      "text": "first 200 chars",
      "weakness": "explanation string",
      "validity_score": 0.0–1.0,
      "accuracy_score": 0.0–1.0,
      "contradicted_by": ["quote1", "quote2"]
    }
  ],
  "created_at": "ISO 8601"
}
```

**Requires sampling:** Yes. Returns `{"error": "sampling_not_supported", ...}` otherwise.

**Progress events:** emits `ctx.report_progress()` at ~5%, per-component percentage, 85%, 90%, 100%.

---

#### `decompose_argument`

Extracts Toulmin components without evidence lookup or assessment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Argument text |
| `store_result` | bool | `False` | Persist to SQLite (unused in current implementation) |

Returns: `ToulminComponents.model_dump()` — dict with keys `claim`, `data`, `warrant`, `backing`, `qualifier`, `rebuttal`.

**Requires sampling:** Yes.

---

#### `detect_fallacies`

Identifies logical fallacies in text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Argument text |

Returns: `list[dict]` where each dict is `{fallacy_type, excerpt, explanation, severity}`.

Known fallacy types include: `ad_hominem`, `straw_man`, `false_dichotomy`, `appeal_to_authority`, `slippery_slope`, `hasty_generalization`, `circular_reasoning`, `red_herring`. The LLM may return others.

**Requires sampling:** Yes.

---

#### `get_assessment`

Retrieves a previously stored assessment.

| Parameter | Type | Description |
|-----------|------|-------------|
| `assessment_id` | str | UUID of a stored `ArgumentAssessment` |

Returns: Same format as `assess_argument`, or `{"error": "not_found", "id": "..."}`.

---

### 7.2 Argument Mapping Tools (`tools/mapping.py`)

---

#### `create_argument_map`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | str | required | Human-readable title |
| `description` | str | `""` | Optional description |

Returns: `{"id": uuid, "title": str}`

---

#### `add_argument_node`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `map_id` | str | required | UUID of target map |
| `text` | str | required | The argument text for this node |
| `node_type` | str | `"claim"` | `claim`, `premise`, `conclusion`, `evidence`, or `rebuttal` |

Returns: `{"id": uuid, "map_id": str, "node_type": str, "text": str}`

---

#### `link_arguments`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `map_id` | str | required | UUID of target map |
| `source_id` | str | required | UUID of source node |
| `target_id` | str | required | UUID of target node |
| `relationship` | str | required | `supports`, `attacks`, `qualifies`, or `elaborates` |
| `explanation` | str | `""` | Optional one-sentence explanation |

Returns: `{"id": uuid, "source_id": str, "target_id": str, "relationship": str}`

---

#### `get_argument_map`

| Parameter | Type | Description |
|-----------|------|-------------|
| `map_id` | str | UUID of the map |

Returns: Full `ArgumentMap.model_dump(mode="json")` with nested `nodes` and `edges` arrays.

---

#### `auto_map_argument`

Full automated pipeline: decompose text → create argument nodes → infer relationships.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Argument text |
| `map_id` | str | `None` | Existing map to add to; creates new map if absent |

**Pipeline:**
1. `SamplingHelper.decompose(text)` → ToulminComponents
2. Creates a new map if `map_id` is None (title = first 60 chars of claim)
3. Creates nodes: `claim` → one claim node; `data` → one premise node each; `warrant` → premise node; `rebuttal` → rebuttal node
4. `SamplingHelper.infer_relationships(nodes)` → suggested edges (validated against known node IDs and relationship types before persisting)
5. Returns: `{"map_id": str, "nodes_created": int, "edges_created": int, "claim": str}`

**Requires sampling:** Yes.

---

### 7.3 RAG Ingest Tools (`tools/rag_ingest.py`)

All ingest tools share a common `_ingest()` helper with this pipeline:
1. Load document text via appropriate `DocumentLoader`
2. `TextChunker.chunk(text, doc_id, collection, metadata)` → `list[Chunk]`
3. `Embedder.embed_batch([c.text for c in chunks])` → `list[list[float]]`
4. `DocumentStore.save_document(doc)` + `DocumentStore.save_chunks(chunks)` (SQLite)
5. `VectorBackend.add_embeddings(chunk_ids, embeddings, collection, metadatas)` (ChromaDB — chunk_ids only, no text)
6. Progress reported via `ctx.report_progress()`

---

#### `ingest_file`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | required | Absolute or `~`-prefixed local file path |
| `collection` | str | `"default"` | Target collection |
| `title` | str | `None` | Override auto-detected title |
| `metadata` | dict | `{}` | Additional metadata to store |

Loader selection: `.pdf` → `PDFLoader` (pypdf), all others → `TextLoader`.

Returns: `{"doc_id": uuid, "title": str, "source": str, "collection": str, "chunks": int}`

---

#### `ingest_url`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | required | HTTP/HTTPS URL |
| `collection` | str | `"default"` | Target collection |
| `title` | str | `None` | Override page title |

Uses `WebLoader` (httpx async client, BeautifulSoup strip of nav/footer/script/style).

---

#### `ingest_text`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Raw text content |
| `title` | str | required | Document title |
| `collection` | str | `"default"` | Target collection |
| `metadata` | dict | `{}` | Additional metadata |

Source stored as `"inline"`.

---

### 7.4 RAG Search Tools (`tools/rag_search.py`)

---

#### `search`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Search query |
| `collection` | str | `"default"` | Collection to search |
| `top_k` | int | `5` | Number of results to return |

**Pipeline:** `Embedder.embed(query)` → `VectorBackend.search()` → `DocumentStore.get_chunks_by_ids()` → sorted by score.

Returns: `list[dict]` with keys `rank`, `score`, `text`, `doc_id`, `chunk_id`, `metadata`. Does not require sampling.

---

#### `get_document`

| Parameter | Type | Description |
|-----------|------|-------------|
| `doc_id` | str | UUID of the document |

Returns full document including `full_text`. Does not require sampling.

---

#### `synthesize_answer`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Question to answer |
| `collection` | str | `"default"` | Collection to search |
| `top_k` | int | `5` | Chunks to retrieve |

Searches then calls `SamplingHelper.synthesize_rag_answer(query, results)`. System prompt instructs the LLM to answer only from provided context and cite sources with `[N]`.

**Graceful degradation:** If sampling unavailable, returns markdown-formatted raw chunks instead of synthesized prose.

---

### 7.5 Management Tools (`tools/manage.py`)

| Tool | Parameters | Returns | Description |
|------|-----------|---------|-------------|
| `list_argument_maps` | — | `list[dict]` | All maps: id, title, description, created_at |
| `delete_argument_map` | `map_id: str` | `{"deleted": bool, "map_id": str}` | Deletes map + all nodes/edges (CASCADE) |
| `export_argument_map` | `map_id: str` | `ArgumentMap.model_dump(mode="json")` | Full JSON export |
| `list_documents` | `collection: str\|None` | `list[dict]` | All docs with id, title, source, chunk_count, created_at |
| `delete_document` | `doc_id: str` | `{"deleted": bool, "chunks_removed": int}` | Removes from SQLite + ChromaDB |
| `collection_stats` | `collection: str = "default"` | `{"collection": str, "documents": int, "chunks": int}` | Count query |

---

## 8. MCP Resources

Resources provide read-only structured data accessible at URIs. They are defined in `server.py`. Resources create their own short-lived store connections (they do not have access to `AppState` in the current implementation).

| URI | Description | Returns |
|-----|-------------|---------|
| `aa://rag/stats` | All collections with document/chunk counts | JSON object |
| `aa://rag/documents/{collection}` | Documents in a specific collection | JSON array |
| `aa://maps` | All argument maps (id, title, description, created_at) | JSON array |
| `aa://maps/{map_id}` | Full argument map with nodes and edges | JSON object |

---

## 9. MCP Prompts

Prompt templates that MCP clients can inject into conversations.

| Prompt name | Parameters | Description |
|-------------|-----------|-------------|
| `assess_argument_prompt` | `text: str, collection: str = "default"` | Instructs the LLM to call `assess_argument` on the provided text |
| `rag_query_prompt` | `question: str, collection: str = "default"` | Instructs the LLM to use `search` or `synthesize_answer` |

---

## 10. Sampling Prompts (Internal)

These are the prompts sent to the client LLM via `create_message()`. They are defined in `SamplingHelper` in `analysis/sampling.py`.

### 10.1 `decompose(text)`

**System:** Expert in argument analysis and Toulmin model. JSON only, no markdown.
**User prompt:** Instructs extraction of `{claim, data, warrant, backing, qualifier, rebuttal}` as a JSON object.
**Temperature:** 0.1 (deterministic)
**Max tokens:** `max_tokens_decompose` (default 1024)
**Expected response:** JSON object matching `ToulminComponents`

### 10.2 `assess_component(component_type, component_text, evidence_chunks)`

**System:** Expert in critical thinking, logic, fact-checking. JSON only.
**User prompt:** Provides the component type and text, formatted evidence from RAG (or note that none is available), and requests `{logical_validity_score, factual_accuracy_score, supporting_evidence, contradicting_evidence, weakness_flag, weakness_explanation}`.
**Temperature:** 0.1
**Max tokens:** `max_tokens_assess` (default 1024)
**Weakness flag rule:** explicitly states `weakness_flag=true` when score < 0.4 on either dimension or directly contradicted.

### 10.3 `detect_fallacies(text)`

**System:** Expert in logic and informal fallacies. JSON only.
**User prompt:** Requests a JSON array of `{fallacy_type, excerpt, explanation, severity}`. Returns `[]` if none found.
**Temperature:** 0.1
**Max tokens:** `max_tokens_fallacy` (default 1024)

### 10.4 `synthesize_assessment(source_text, toulmin, components, fallacies)`

**System:** Expert argument analyst. JSON only.
**User prompt:** Provides original argument (truncated to 500 chars), claim, per-component scores, pre-formatted critical weakness block, pre-formatted fallacy block. Requests `{overall_validity_score, overall_truthfulness_score, summary}`.
**Temperature:** 0.3
**Max tokens:** `max_tokens_synthesize` (default 2048)
**Summary instruction:** Start with verdict. Explicitly call out critical weaknesses. Cite specific components.

### 10.5 `synthesize_rag_answer(query, results)`

**System:** Answer using only provided context. Cite sources with [N].
**User prompt:** Formatted evidence chunks followed by the question.
**Temperature:** 0.3
**Max tokens:** `max_tokens_synthesize`

### 10.6 `infer_relationships(nodes)`

**System:** Expert in argument structure. JSON only.
**User prompt:** Formatted list of nodes with short IDs and types. Requests JSON array of edges with `{source_id, target_id, relationship, explanation}`. Instructs to return `[]` if uncertain.
**Temperature:** 0.1
**Max tokens:** 1024 (hardcoded)

---

## 11. RAG Subsystem

### 11.1 Text Chunking (`rag/chunker.py`)

`TextChunker` splits text into overlapping chunks. Strategy is configurable.

| Strategy | Behaviour |
|----------|-----------|
| `recursive` | Tries separators in order: `\n\n`, `\n`, `. `, ` `. Produces semantically-aware chunks preserving paragraph/sentence boundaries. **Default.** |
| `sentence` | Uses regex boundary detection `[^.!?]+[.!?]+`. Accumulates sentences until chunk size exceeded. Best for factual prose. |
| `fixed` | Hard character splits with overlap. Best for code or highly structured data. |

All strategies respect `min_chunk_size` — chunks shorter than this threshold are discarded. Chunks below threshold are typically trailing fragments.

Each `Chunk` records `start_char` / `end_char` offsets into the original `full_text`, enabling precise source attribution.

### 11.2 Embedding (`rag/embedder.py`)

`Embedder` wraps `sentence_transformers.SentenceTransformer`. Key behaviours:

- `load()` is synchronous and must be called via `anyio.to_thread.run_sync(embedder.load)` during lifespan startup. This blocks a thread but not the event loop.
- `embed_batch(texts)` is async, dispatches `model.encode(texts, ...)` to a thread.
- Model is downloaded from HuggingFace Hub on first run and cached to `~/.cache/huggingface/` (or `CACHE_DIR` if configured).
- Default model `all-MiniLM-L6-v2`: 384-dimensional embeddings, ~22MB, fast CPU inference.

### 11.3 Vector Backend (`rag/backends/`)

`VectorBackend` is an abstract class defining the contract:

```python
async def initialize() -> None
async def add_embeddings(chunk_ids, embeddings, collection, metadatas) -> None
async def search(query_embedding, collection, top_k, filter) -> list[tuple[str, float]]
async def delete_by_doc_id(doc_id, collection) -> None
async def close() -> None
```

`ChromaBackend` implements this using `chromadb.PersistentClient`. All ChromaDB calls are synchronous and dispatched via `anyio.to_thread.run_sync()`. Collections are created on demand with `get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`.

**`search()` returns:** `list[tuple[chunk_id, similarity_score]]` sorted by descending score.

**Stub implementations** exist for `sqlite-vec` (stub, not implemented) and `lancedb` (not yet implemented). Activating a non-`chroma` backend raises `ValueError` with a message explaining the required extra.

### 11.4 RAG Pipeline (`rag/pipeline.py`)

`RAGPipeline.search(query, collection, top_k)`:
1. Embeds query text
2. Queries ChromaDB for `top_k_retrieval` results (uses `config.sampling.top_k_retrieval` as retrieve count, `top_k` as final count)
3. Extracts `chunk_id` list from results
4. Fetches full `Chunk` objects from SQLite by IDs
5. Sorts by similarity score (score_map lookup)
6. Returns `top_k` `SearchResult` objects

---

## 12. Document Loaders (`loaders/`)

All implement `DocumentLoader` ABC with `async def load(source: str) -> tuple[str, str, dict]` returning `(text, title, metadata)`.

| Loader | File | Handles | Notes |
|--------|------|---------|-------|
| `TextLoader` | `loaders/text.py` | `.txt`, `.md`, any text | Uses `anyio.Path.read_text()` |
| `PDFLoader` | `loaders/pdf.py` | `.pdf` | `pypdf.PdfReader` in thread; extracts per-page text joined by `\n\n` |
| `WebLoader` | `loaders/web.py` | HTTP/HTTPS URLs | `httpx.AsyncClient` with `follow_redirects=True`; strips script/style/nav/footer/header/aside tags |

Loader selection in `ingest_file`: `.pdf` extension → `PDFLoader`, otherwise → `TextLoader`.

---

## 13. Project File Structure

```
argument-analysis-mcp/
├── pyproject.toml                   # Build config, dependencies, entry point
├── .env.example                     # Template for user configuration
├── README.md                        # Quick-start and tool reference
├── specification.md                 # This document
│
├── src/
│   └── aa_mcp/
│       ├── __init__.py              # Package version
│       ├── __main__.py              # `python -m aa_mcp` → mcp.run(transport="stdio")
│       ├── server.py                # FastMCP app, tool/resource/prompt registration
│       ├── config.py                # ServerConfig and sub-configs (pydantic-settings)
│       ├── lifespan.py              # app_lifespan context manager, AppState dataclass
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   └── sampling.py          # SamplingHelper, has_sampling_capability()
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── argument.py          # ToulminComponents, FallacyDetection, ComponentAssessment, ArgumentAssessment
│       │   ├── mapping.py           # ArgumentNode, ArgumentEdge, ArgumentMap, RelationshipType
│       │   └── rag.py               # Document, Chunk, SearchResult
│       │
│       ├── store/
│       │   ├── __init__.py
│       │   ├── document_store.py    # DocumentStore (documents + chunks tables)
│       │   └── argument_store.py    # ArgumentStore (assessments + maps + nodes + edges tables)
│       │
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── chunker.py           # TextChunker (recursive/sentence/fixed)
│       │   ├── embedder.py          # Embedder (sentence-transformers, thread dispatch)
│       │   ├── pipeline.py          # RAGPipeline.search()
│       │   └── backends/
│       │       ├── __init__.py
│       │       ├── base.py          # VectorBackend ABC
│       │       └── chroma.py        # ChromaBackend
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── analysis.py          # assess_argument, decompose_argument, detect_fallacies, get_assessment
│       │   ├── mapping.py           # create/add/link/get/auto argument map tools
│       │   ├── rag_ingest.py        # ingest_file, ingest_url, ingest_text
│       │   ├── rag_search.py        # search, get_document, synthesize_answer
│       │   └── manage.py            # list/delete maps and documents, collection_stats
│       │
│       ├── loaders/
│       │   ├── __init__.py
│       │   ├── base.py              # DocumentLoader ABC
│       │   ├── text.py              # TextLoader (.txt, .md)
│       │   ├── pdf.py               # PDFLoader (pypdf)
│       │   └── web.py               # WebLoader (httpx + BeautifulSoup)
│       │
│       ├── resources/
│       │   └── __init__.py          # Resources defined inline in server.py
│       │
│       └── prompts/
│           └── __init__.py          # Prompts defined inline in server.py
│
└── tests/
    ├── conftest.py                  # Fixtures: tmp_db, tmp_data_dir
    ├── test_chunker.py              # TextChunker unit tests
    └── test_argument_store.py       # DocumentStore + ArgumentStore round-trip tests
```

---

## 14. Installation and Deployment

### 14.1 End User (pip)

```bash
pip install argument-analysis-mcp
```

The `aa-mcp-server` script is placed in the Python environment's `bin/` directory.

### 14.2 End User (uvx — no install required)

```bash
uvx --from argument-analysis-mcp aa-mcp-server
```

### 14.3 Development

```bash
git clone <repo>
cd argument-analysis-mcp
pip install -e ".[dev]"
python -m pytest tests/
```

### 14.4 Claude Desktop Configuration

**Linux:** `~/.config/Claude/claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "argument-analysis": {
      "command": "aa-mcp-server",
      "env": {
        "AA_MCP_DATA_DIR": "/home/user/.aa-mcp/data"
      }
    }
  }
}
```

For GPU acceleration on Apple Silicon:
```json
"env": {
  "AA_MCP_DATA_DIR": "/Users/user/.aa-mcp/data",
  "AA_MCP_EMBEDDER__DEVICE": "mps"
}
```

---

## 15. Known Limitations and Future Work

### 15.1 Current Limitations

- **No LanceDB or sqlite-vec implementation.** The `VectorBackend` ABC has stubs for both but only `ChromaBackend` is implemented. Activating either raises `ValueError`.
- **Resources re-open SQLite.** MCP resources (`aa://rag/stats`, etc.) construct their own short-lived store connections rather than using the lifespan `AppState`. This is slightly inefficient but functionally correct.
- **No concurrent write safety.** aiosqlite uses `INSERT OR REPLACE` (upsert) semantics, but simultaneous ingest operations from multiple tool calls within the same session may interleave. Single-session use is assumed.
- **ChromaDB cosine distance conversion.** The formula `1.0 - (distance / 2.0)` assumes ChromaDB HNSW returns cosine distance in [0, 2]. This is consistent with current ChromaDB behavior but is not formally guaranteed across versions.
- **Embedding model dimension mismatch.** If the user changes `AA_MCP_EMBEDDER__MODEL` after documents have been ingested, the existing ChromaDB vectors have the wrong dimension and searches will fail. A `reindex_collection` tool is listed in the plan but not yet implemented.
- **`store_result=False` in `decompose_argument` is unused.** The parameter exists for API symmetry but the function doesn't persist anything regardless.
- **Qualifier and rebuttal not individually assessed.** The `assess_argument` pipeline only assesses `claim`, `datum`, `warrant`, and `backing` components. Qualifier and rebuttal are extracted by decomposition but not evidence-checked.
- **Sampling temperature is not user-configurable per-call.** The temperature for each sampling operation is fixed in `SamplingConfig` and cannot be overridden in individual tool calls.

### 15.2 Planned / Suggested Future Work

- **Implement `reindex_collection`** — re-embed all chunks in a collection after an embedding model change.
- **Implement `sqlite-vec` backend** — enables single-file deployment with no ChromaDB dependency.
- **Implement `lancedb` backend** — better performance for large corpora (>500k chunks).
- **HyDE search enhancement** — `SamplingHelper.hyde_expand()` is implemented but not wired into the pipeline. Activating it would improve recall on vague queries by embedding a hypothetical answer document rather than the query itself.
- **Qualifier and rebuttal assessment** — extend `assess_argument` to also assess these Toulmin components.
- **Batch assessment** — accept multiple argument texts in one call, returning a comparative assessment.
- **Argument versioning** — track how an argument changes across iterations (useful for LLM red-teaming workflows).
- **Export formats** — Graphviz DOT, Mermaid diagram, Argdown format exports for argument maps.
- **Full-text search** — add SQLite FTS5 extension to `documents`/`chunks` for keyword-based retrieval alongside vector search.
- **Hybrid search** — combine BM25 (FTS5) and vector similarity scores (reciprocal rank fusion) for better retrieval.
- **Assessment history** — add `list_assessments()` management tool.
- **Async batching** — assess multiple components concurrently (fan-out) rather than sequentially in `assess_argument`. Currently sequential to avoid overwhelming the sampling queue.
- **Streaming** — FastMCP supports streaming tool responses; `assess_argument` could stream component results as they complete.
- **Structured output types** — some MCP clients support requesting structured output from sampling; this could replace the JSON-parsing-from-prose approach.

---

## 16. Testing

**Test framework:** pytest + pytest-asyncio (`asyncio_mode = "auto"`)
**Test directory:** `tests/`

### Current Tests

| File | Coverage |
|------|----------|
| `test_chunker.py` | Fixed split produces chunks, recursive split on short text, unique IDs, min_chunk_size filter |
| `test_argument_store.py` | Document round-trip, chunk round-trip, assessment round-trip, argument map round-trip |

### Test Fixtures (`conftest.py`)

- `tmp_db` — `Path` to a temporary `.db` file (pytest `tmp_path`)
- `tmp_data_dir` — `Path` to a temporary data directory

### Not Covered (future)

- `SamplingHelper` — requires mocking `ServerSession.create_message()`
- `RAGPipeline` — requires mock `VectorBackend` and `Embedder`
- `ChromaBackend` — requires temporary ChromaDB directory
- Tool functions — require mock `Context` and `AppState`
- `DocumentLoader` variants — file I/O and network tests
