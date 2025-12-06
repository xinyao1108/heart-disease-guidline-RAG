# MedAgenticSystem (English Guideline RAG)

Heart-disease-guideline-RAG implements the v1 architecture from `overroll_design_v1.md`: a FastAPI service that answers cardiovascular questions using only the English PDFs shipped under `RAG-guidelines/` (the `心内科指南` subtree is ignored). Retrieval uses open-source components—Tantivy for BM25, FlagEmbedding `BAAI/bge-m3` for embeddings, Qdrant for vector search, and `BAAI/bge-reranker-v2-m3` for reranking—while generation relies on OpenAI `gpt-4.1-mini`.

## 1. Prerequisites

- Python 3.12 (already pinned by `.python-version`)
- [uv](https://github.com/astral-sh/uv) (the repo already contains `.venv` metadata; run commands via `uv`)
- Docker + docker-compose if you prefer containerized deployment
- A valid `OPENAI_API_KEY`

## 2. Project setup with uv

```bash
# install dependencies
UV_CACHE_DIR=.uv_cache uv sync

# copy env template and fill in secrets / paths
cp .env.example .env
```

Key `.env` fields:

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | Secret key for `gpt-4.1-mini` |
| `GUIDELINE_ROOT` | Absolute path to the English PDF folder (mount `RAG-guidelines/` or copy it into `data/raw/english_guidelines/`) |
| `PARSED_DOCS_PATH`, `CHUNKS_PATH`, `BM25_INDEX_DIR` | Artifacts produced by the ingestion pipeline |
| `QDRANT_URL`, `QDRANT_COLLECTION` | Dense index endpoint and collection name |
| `ALLOW_TIKTOKEN_FALLBACK` | Set to `true` to automatically use whitespace token estimates if `tiktoken` cannot be downloaded |

## 3. Ingestion pipeline (PDF → chunks → indexes)

All commands run through uv to ensure the virtualenv is isolated:

```bash
# 1. Scan metadata (optional helper)
UV_CACHE_DIR=.uv_cache uv run python -m app.ingestion.scan_guidelines

# 2. Parse PDFs into paragraph JSONL
UV_CACHE_DIR=.uv_cache uv run python -m app.ingestion.parse_pdfs

# 3. Chunk paragraphs with tiktoken sizing
UV_CACHE_DIR=.uv_cache uv run python -m app.ingestion.chunking

# 4. Build Tantivy BM25 index
UV_CACHE_DIR=.uv_cache uv run python -m app.ingestion.index_bm25

# 5. Start Qdrant (docker compose example below) and load dense vectors
UV_CACHE_DIR=.uv_cache uv run python -m app.ingestion.index_vectors
```

Notes:

- `scan_guidelines` derives `guideline_id` deterministically from each PDF filename (lowercase, spaces → `-`), so downstream modules can join on this key.
- Parsed paragraph JSONL (`app.ingestion.parse_pdfs`) uses the `Paragraph` schema: document metadata, `section_id`/`section_title`, `page`, `order`, `lang`, `text`. `section_id` comes from the detected heading (e.g., `1.2`). When a heading has no numbering, the parser assigns a monotonically increasing sequence (`"1"`, `"2"`, …) so IDs stay unique and never revert.
- `chunking` relies on `tiktoken` (`cl100k_base`) for accurate token estimates.
- If `tiktoken` data is unavailable, scripts now prompt you (or respect `ALLOW_TIKTOKEN_FALLBACK=true`) before falling back to whitespace token counts, ensuring operators explicitly approve the approximation.
- The pipeline only reads PDFs outside any `心内科指南` directory to satisfy the “English-only” constraint.

## 4. Running the API locally

```bash
UV_CACHE_DIR=.uv_cache uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Test it via the built-in docs (`http://localhost:8000/docs`) or curl:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the guideline-recommended therapy for chronic coronary disease?"}'
```

The response contains the generated answer, structured evidences with `[Doc #]` identifiers, and a medical disclaimer appended by the server.

## 5. Docker & docker-compose

`docker/Dockerfile` installs uv, syncs dependencies via `uv.lock`, and runs FastAPI. `docker/docker-compose.yml` orchestrates Qdrant + the API container:

```bash
cd docker
docker compose up -d
```

The compose file mounts:

- `../data` → `/app/data` (parsed docs, chunks, indexes)
- `../RAG-guidelines` → `/app/data/raw/english_guidelines`

Rebuild the API image whenever `pyproject.toml`/`uv.lock` changes:

```bash
cd docker
docker compose build medagenticsystem-api
docker compose up -d medagenticsystem-api
```

## 6. Component summary

| Layer | Implementation |
| --- | --- |
| API | FastAPI + Pydantic models |
| PDF parsing | PyMuPDF (`fitz`) |
| Token estimation | tiktoken with whitespace fallback |
| Sparse search | Tantivy BM25 index |
| Dense search | Qdrant + FlagEmbedding `BAAI/bge-m3` |
| Reranker | FlagEmbedding `BAAI/bge-reranker-v2-m3` |
| LLM | OpenAI `gpt-4.1-mini` via Responses API |

## 7. Useful uv commands

```bash
# Run tests or type-checkers if/when added
uv run pytest
uv run mypy

# Launch the API without reload
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

This repository is ready for future extensions (Chinese guidelines, structured recommendation graphs, etc.) without altering the public API surface.

## 8. Known limitations

- **PDF parsing**: PyMuPDF only extracts linear text blocks; tables or figures embedded as images/complex layouts are not captured, and table structure is flattened/lost. Headers/footers may require manual cleaning.
- **Chunking**: Recommendation class/level extraction is regex-based and may miss non-standard phrasing. Overlap settings are heuristic; there is no sophisticated semantic splitting yet.
- **Dense indexing**: FlagEmbedding BGE-M3 runs on CPU by default and can be slow for large corpora. Provision GPU/accelerators if you need faster ingestion.
- **Hybrid retrieval**: Requires both Tantivy index files and a live Qdrant instance; missing either part reduces coverage (hybrid code logs warnings but overall recall drops).
- **OpenAI generation**: Depends on external OpenAI API; failures propagate as HTTP 500. There is no offline model fallback.
- **Fallback behavior**: When `tiktoken` can’t download tokenizer files, ingestion pauses for operator confirmation (or uses whitespace counts if `ALLOW_TIKTOKEN_FALLBACK=true`), so token sizes become approximate.
