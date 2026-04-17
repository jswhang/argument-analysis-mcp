# Argument Analysis MCP Server

A local MCP server that assesses the validity and truthfulness of arguments.

**Core workflow**: Feed it LLM-generated (or any) argument text → it decomposes the argument into its structural components → retrieves domain-specific evidence from your knowledge base → assesses logical validity and factual accuracy per component → flags especially weak parts → returns a structured report.

## Quick Start

### Install

```bash
pip install argument-analysis-mcp
# or without installing:
# uvx --from argument-analysis-mcp aa-mcp-server
```

### Add to Claude Code

**From a pip install:**

```bash
claude mcp add argument-analysis -- aa-mcp-server
```

**From a local clone (development):**

```bash
git clone <repo>
cd argument-analysis-mcp
pip install -e ".[dev]"

claude mcp add argument-analysis -- aa-mcp-server
```

The editable install wires `aa-mcp-server` to your local source, so code changes are reflected without reinstalling.

Alternatively, run directly from the repo without installing the script:

```bash
claude mcp add argument-analysis -- python -m aa_mcp
```

To pass environment variables (either method):

```bash
claude mcp add argument-analysis \
  -e AA_MCP_DATA_DIR=/home/yourname/.aa-mcp/data \
  -- aa-mcp-server
```

Verify it's connected:

```bash
claude mcp list
```

Then start a `claude` session — all tools are available immediately.

### Add to Claude Desktop

Edit `~/.config/Claude/claude_desktop_config.json` (Linux) or
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "argument-analysis": {
      "command": "aa-mcp-server",
      "env": {
        "AA_MCP_DATA_DIR": "/home/yourname/.aa-mcp/data"
      }
    }
  }
}
```

## Usage

### 1. Populate the knowledge base

```
ingest_file("/path/to/domain-paper.pdf", collection="climate")
ingest_url("https://example.com/article", collection="climate")
ingest_text("...", title="Reference text", collection="climate")
```

### 2. Assess an argument

```
assess_argument(
  text="Climate change is not caused by humans because temperatures fluctuated naturally in the past.",
  collection="climate"
)
```

Returns:
- Argument decomposition (claim, evidence, reasoning, support, limitations, counterarguments)
- Per-component logical validity and factual accuracy scores (0–1)
- RAG evidence supporting or contradicting each component
- Detected logical fallacies with severity
- `⚠️ CRITICAL_WEAKNESSES` — prominently flagged weak components
- Overall validity and truthfulness scores
- Prose summary

### 3. Build argument maps

```
map_id = create_argument_map("Climate debate")["id"]
n1 = add_argument_node(map_id, "CO2 causes warming", node_type="claim")["id"]
n2 = add_argument_node(map_id, "Past natural fluctuations occurred", node_type="premise")["id"]
link_arguments(map_id, n2, n1, relationship="attacks")

# Or automatically:
auto_map_argument("Climate change is not caused by humans because...")
```

## Available Tools

| Tool | Description |
|------|-------------|
| `assess_argument` | **Primary** — decompose + RAG evidence + per-component assessment + report |
| `decompose_argument` | Extract argument components only |
| `detect_fallacies` | Identify logical fallacies |
| `get_assessment` | Retrieve a stored assessment |
| `create_argument_map` | Create an argument graph |
| `add_argument_node` | Add a node (claim/premise/evidence/rebuttal) |
| `link_arguments` | Add edge (supports/attacks/qualifies/elaborates) |
| `get_argument_map` | Retrieve a full map |
| `auto_map_argument` | Auto-decompose and map an argument |
| `ingest_file` | Ingest PDF, .md, .txt into knowledge base |
| `ingest_url` | Fetch and ingest a web page |
| `ingest_text` | Ingest raw text |
| `search` | Semantic search |
| `get_document` | Retrieve full document text |
| `synthesize_answer` | Search + LLM synthesis |
| `list_documents` | List knowledge base documents |
| `delete_document` | Remove a document |
| `list_argument_maps` | List all argument maps |
| `delete_argument_map` | Delete a map |
| `export_argument_map` | Export map as JSON |
| `collection_stats` | Document/chunk counts |

## Configuration

```bash
AA_MCP_DATA_DIR=~/.aa-mcp/data
AA_MCP_RAG__BACKEND=chroma             # chroma (default)
AA_MCP_EMBEDDER__MODEL=sentence-transformers/all-MiniLM-L6-v2
AA_MCP_EMBEDDER__DEVICE=cpu            # cpu | cuda | mps
AA_MCP_SAMPLING__ENABLED=true
AA_MCP_SAMPLING__MAX_TOKENS=2048
```

## Storage

- **SQLite** (`AA_MCP_DATA_DIR/aa_mcp.db`) — source of truth for all text: full articles, chunks, argument maps, assessments
- **ChromaDB** (`AA_MCP_DATA_DIR/chroma/`) — vector index only (embeddings + chunk ID references back to SQLite)

## Requirements

- Python 3.12+
- MCP client with `sampling/createMessage` support (Claude Desktop)
- Embedding model downloads automatically on first run (~22MB)

## Development

```bash
git clone ...
pip install -e ".[dev]"
python -m pytest tests/
```
