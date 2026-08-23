# FAQ — Common Questions About This Project

## What kind of project is this, and what does it do?

It's a **Multimodal RAG Knowledge Assistant** — a question-answering system
that reads a knowledge base of PDFs and images, and answers natural-language
questions strictly from what it finds in those documents, citing the source.
It does not answer from the LLM's general training knowledge and it does not
search the internet. See [OVERVIEW.md](OVERVIEW.md) for the full breakdown.

## When a document is ingested, why does "embedding" happen more than once? Is the same thing being stored twice?

It isn't the same thing twice — it's two different pieces of data, at two
different points in time:

1. **At ingest time** (once per document): the *document's* text is split
   into chunks, each chunk is embedded, and stored permanently in the
   pgvector table.
2. **At query time** (every time someone asks a question): the *question*
   itself is embedded too — but only temporarily, in memory, purely so it
   can be compared against the stored document chunks. The question's
   embedding is never written to the database.

So step 1 embeds the document's content; step 2 embeds the user's question.
They look similar ("turn text into an embedding") but they're operating on
different inputs for different reasons.

## When it retrieves an answer, where does it actually search — the database, or the internet (e.g. a search engine)?

**Only the pgvector database — nothing external.** The retrieval step
(`store.similarity_search(...)` in
[`src/graph.py`](../src/graph.py)) compares the question's embedding
against the embeddings already stored from ingested documents, using vector
similarity (cosine similarity) — a purely mathematical "which stored chunks
are numerically closest to this question" comparison. No search engine, no
web request, no external API other than the LLM/embedding provider itself
is involved in retrieval.

This also means the assistant can only answer questions about content that
was explicitly ingested. If a document was never added to the knowledge
base, the assistant has no way to know about it.

## Does the "question" being asked have to be text, or can someone send a new PDF as the question?

The query is always a **text** question. This project's current flow is:

- **Ingest**: PDF/image → becomes part of the searchable knowledge base.
- **Query**: text question → answered using that knowledge base.

Sending a new document *as the query itself* (e.g. "check this PDF against
what's already indexed") is a different feature this project does not
implement.

## Can this be hosted live for free, so it can be shown to someone?

Yes, with two additions, since right now it's a CLI script with no web
interface:

1. **A web UI** — Streamlit is the fastest way to get a shareable chat-style
   interface with minimal code.
2. **Free-tier hosting for each piece:**

| Component | Free option |
|---|---|
| App hosting | Streamlit Community Cloud or Hugging Face Spaces |
| Postgres + pgvector | Supabase (pgvector built in) or Neon |
| LLM (chat + embeddings) | OpenAI is paid per call — for a fully free stack, swap to a free-tier provider such as Groq |

This has not been built yet — it's a follow-up if/when a live demo link is
needed.
