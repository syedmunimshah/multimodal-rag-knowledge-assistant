# Understanding the Multimodal RAG Knowledge Assistant

This document walks through what this project is and how it works, the way
it was actually understood — starting from the plain question "what does
this do", through the points that were confusing at first, to the final
clear picture.

## What is this project?

It's an assistant that answers questions about your own documents — PDFs
and images — instead of relying on general knowledge the way a normal
chatbot does. You give it a folder of documents once; after that, anyone
can ask it a question in plain English and get an answer that's actually
grounded in those documents, with the source cited.

Two things make it more than a basic "chat with your PDF" tool:

1. It's **multimodal** — it doesn't just read PDFs, it also understands
   images (charts, screenshots, scanned pages) by having a vision model
   describe them first.
2. It doesn't just search once and answer. It reasons in multiple steps —
   plans, retrieves, checks whether it actually has enough to answer, and
   goes back for more if it doesn't.

## How the documents get in — ingestion

Before any question can be answered, the documents have to be prepared:

- A PDF is read page by page and cut into small overlapping chunks of text
  (~1000 characters each). Overlap matters — without it, a sentence that
  falls right on a chunk boundary would get cut in half and lose meaning.
- An image is shown to a vision-capable model, which writes a factual
  description of it — what text is visible, what a chart shows, what
  objects appear. That description is what actually gets used, not the
  raw pixels.
- Every chunk (from a PDF) or caption (from an image) is converted into an
  **embedding** — a list of numbers that represents its meaning — and saved
  into a Postgres database with the **pgvector** extension.

This only happens once per document, when it's ingested.

## What happens when someone asks a question

This is where the first confusion came up: it looked like the same thing
was being embedded twice. It isn't — it's two different things, at two
different points in time.

- **Ingestion time** (once): the *document's* content gets embedded and
  permanently stored.
- **Question time** (every single time someone asks something): the
  *question* also gets turned into an embedding — but only in memory, for
  that one moment, just so it can be compared against what's already
  stored. It's never saved to the database.

So "the document" and "the question" are not the same data being embedded
twice — one is the knowledge base being built, the other is a lookup key
being generated on the fly to search that knowledge base.

## Where does it actually search?

The second point that needed clarifying: does it search the database, or
does it go out to the internet (like a search engine)?

**It only ever searches the pgvector database.** The question's embedding
is compared against the stored document-chunk embeddings using a
mathematical similarity measure (cosine similarity) — essentially "which
stored chunks are numerically closest in meaning to this question." No web
request, no search engine, nothing external is involved in retrieval
itself. The only outside calls are to the LLM/embedding provider, to
generate the embeddings and the final answer text.

The direct consequence: the assistant can only ever answer about content
that was explicitly ingested. If a document was never added, it simply has
no way to know about it — it will say so rather than guess.

## Is the "question" always text, or could someone submit a document as the question?

Always text. The two flows are separate and don't mix:

- **Ingest**: a PDF or image goes in, and becomes searchable knowledge.
- **Query**: a text question goes in, and gets answered from that
  knowledge.

Sending a brand-new document *as* the question (e.g. "does this new PDF
match anything in what I already indexed") isn't something this version
does — that would be a different feature.

## How a question actually gets answered — the reasoning loop

Once retrieval is understood, the last piece is what happens with what gets
retrieved. Instead of "search once, hand the results to the LLM, done", it
runs a small loop:

```
question
   │
   ▼
 plan ──► retrieve ──► grade ──┬──► answer
             ▲                 │
             └──── refine ◄────┘   (not enough context yet, hop budget not spent)
```

- **plan** breaks the question into a few focused sub-queries, so a
  multi-part question actually gets fully covered.
- **retrieve** runs each sub-query against the database.
- **grade** asks the model itself: "is what I've gathered actually enough
  to answer this confidently?"
- if not, **refine** works out what's missing and generates follow-up
  sub-queries, then goes back to retrieve.
- once there's enough (or the hop budget runs out), **answer** responds
  using only the retrieved context, citing sources.

## Summary

- The assistant reads a fixed set of documents (PDFs and images) once and
  builds a searchable, meaning-based index of them (pgvector).
- Every question is compared against that index only — never against the
  open internet — using vector similarity.
- Documents and questions are both turned into embeddings, but at different
  times and for different reasons: one builds the index, the other queries
  it.
- Answering isn't a single lookup; it's a small planned, self-checking loop
  (LangGraph) that can go back for more context before committing to an
  answer.
- The assistant is bounded by design: it only knows what was ingested, and
  it only accepts text as a question.

## Conclusion

This project demonstrates a retrieval system that stays grounded in a
specific, controlled knowledge base rather than in an LLM's general
training or the open internet — and it does so across two input types
(text and images) using one shared reasoning pipeline. The multi-step
plan → retrieve → grade → refine → answer loop is what separates it from a
basic "embed and answer" RAG setup: the system checks its own work before
answering instead of assuming one retrieval pass was enough.
