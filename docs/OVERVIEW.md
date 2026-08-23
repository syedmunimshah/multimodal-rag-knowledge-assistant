# Project Overview

## What this is

A **Multimodal RAG (Retrieval-Augmented Generation) Knowledge Assistant** —
a question-answering system that reads through a knowledge base of PDFs and
images, and answers natural-language questions using only what it finds in
those documents, with sources cited. It does not rely on the LLM's own
training knowledge, and it does not search the internet.

## What problem it solves

A general-purpose LLM (like plain ChatGPT) only knows what it was trained
on — it has no access to a company's internal documents, reports, or
product manuals. Feeding an entire document library into a prompt every
time is not viable: it's slow, expensive, and most models have a hard limit
on how much text fits in one request.

RAG solves this by only pulling in the small number of passages that are
actually relevant to a given question, instead of the whole knowledge base.

## How it works, end to end

### 1. Ingestion (done once per document)

```
document (PDF or image)
   │
   ▼
 split into small overlapping chunks (PDFs) / captioned by a vision model (images)
   │
   ▼
 each chunk converted to an embedding (a vector of numbers representing its meaning)
   │
   ▼
 stored in a pgvector-backed Postgres table
```

- **PDFs** ([`src/loaders/pdf_loader.py`](../src/loaders/pdf_loader.py)) are
  parsed page by page and split into ~1000-character chunks with overlap, so
  a retrieved chunk stays self-contained instead of cutting a sentence in
  half.
- **Images** ([`src/loaders/image_loader.py`](../src/loaders/image_loader.py))
  are sent to a vision-capable chat model, which produces a factual text
  caption (visible text, chart data, objects). That caption — not the raw
  pixels — is what gets embedded. This is what makes it "multimodal": both
  text and image content end up in the same embedding space, so one query
  can retrieve either.

### 2. Answering a question (every time a user asks)

The reasoning is not a single "search once, answer" step. It's a small
state machine built with **LangGraph**
([`src/graph.py`](../src/graph.py)):

```
question
   │
   ▼
 plan ──► retrieve ──► grade ──┬──► answer
             ▲                 │
             └──── refine ◄────┘  (context still insufficient, hop budget not spent)
```

| Step | What it does |
|---|---|
| **plan** | Breaks the question into up to `MAX_REASONING_HOPS` focused sub-queries, so a multi-part question gets fully covered instead of only its top-level phrasing. |
| **retrieve** | Embeds each sub-query and runs a similarity search against the pgvector table — this is the only place data is looked up; there is no external search engine involved. |
| **grade** | Asks the model whether what's been retrieved so far is actually enough to answer confidently. |
| **refine** | If not, works out what's missing and generates follow-up sub-queries, then loops back to retrieve. |
| **answer** | Once sufficient (or the hop budget runs out), answers strictly from the retrieved context and cites `[source:page]` for text or `[source]` for images. |

### 3. Context engineering

`format_context` in [`src/prompts.py`](../src/prompts.py) deduplicates
retrieved chunks by `(source, page)` and caps the total context by
character budget. Without this, a multi-hop question could accumulate
duplicate or excessive context and blow past the prompt's token budget in
an unpredictable way.

## What it explicitly does NOT do

- It does not search the internet — retrieval is limited to whatever has
  been ingested into the pgvector table.
- A query is always a text question — the current flow does not accept a
  document as the query itself (e.g. "check this new PDF against my
  existing ones").
- It has no persistent memory across separate questions; each `ask()` call
  starts a fresh reasoning run.

## Tech stack

| Tool | Role |
|---|---|
| Python | Project language |
| LangChain | Document loaders, text splitting, embeddings/chat model wrappers |
| LangGraph | The plan → retrieve → grade → refine → answer state machine |
| pgvector (via `langchain-postgres`) | Postgres-native vector similarity search |
| OpenAI API | Chat, vision captioning, and embedding models (swappable via `.env`) |
| pypdf | PDF text extraction |
| pytest | Unit tests |
