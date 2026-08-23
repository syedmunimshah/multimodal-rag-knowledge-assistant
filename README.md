# Multimodal RAG Knowledge Assistant

An intelligent document assistant that answers questions over a mixed
knowledge base of PDFs and images. Retrieval runs on **pgvector** semantic
search; answering runs through a **LangGraph** multi-step reasoning pipeline
that plans sub-queries, retrieves, grades sufficiency, and refines before
answering — instead of a single embed-and-stuff RAG call.

## How it works

```
question
   │
   ▼
 plan ──► retrieve ──► grade ──┬──► answer
             ▲                 │
             └──── refine ◄────┘ (insufficient context, hops < MAX_REASONING_HOPS)
```

1. **plan** — the question is broken into up to `MAX_REASONING_HOPS` focused
   sub-queries.
2. **retrieve** — each sub-query is embedded and searched against pgvector.
3. **grade** — the model judges whether the gathered context is sufficient.
4. **refine** — if not, it identifies what's missing and generates follow-up
   sub-queries, looping back to retrieve.
5. **answer** — once sufficient (or the hop budget is spent), the model
   answers strictly from the retrieved context, citing `[source:page]`.

### Multimodal ingestion

- PDFs are parsed page-by-page and chunked (`RecursiveCharacterTextSplitter`).
- Images are captioned by a vision-capable chat model at ingest time; the
  caption is what gets embedded. This keeps both modalities in a single
  text-embedding space so a query can retrieve either without a separate
  image index.

### Context engineering

`format_context` in [`src/prompts.py`](src/prompts.py) dedupes retrieved
chunks by `(source, page)` and enforces a character budget, so the answer
prompt's size is predictable regardless of how many reasoning hops ran.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements-dev.txt
copy .env.example .env        # then fill in OPENAI_API_KEY and PG_CONNECTION_STRING
```

Requires a Postgres database with the `vector` extension enabled:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Usage

```bash
# Ingest a folder of PDFs/images (or a single file)
python -m src.ingest ./data/sample_docs

# Ask questions interactively
python -m src.cli
```

## Tests

```bash
pytest
```

## Tools used

- **Python** — project language
- **LangChain** — document loaders, text splitting, embeddings/chat model wrappers
- **LangGraph** — the plan → retrieve → grade → refine → answer state machine
- **pgvector** (via `langchain-postgres`) — Postgres-native vector similarity search
- **OpenAI API** — chat, vision captioning, and embedding models (swappable via `.env`)
- **pypdf** — PDF text extraction
- **pytest** — unit tests
