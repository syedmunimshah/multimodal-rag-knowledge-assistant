from langchain_core.documents import Document

PLAN_PROMPT = """You are planning how to answer a question using a knowledge base.
Break the question into 1-{max_hops} focused sub-queries that, together, retrieve
everything needed to answer it. If the question is already narrow, return just one
sub-query.

Question: {question}

Return one sub-query per line, nothing else."""

ANSWER_PROMPT = """Answer the question using ONLY the context below. If the context
does not contain the answer, say so explicitly instead of guessing.

Cite sources inline as [source:page] or [source] for images.

Context:
{context}

Question: {question}

Answer:"""

SUFFICIENCY_PROMPT = """Given the question and the context gathered so far, decide if
there is enough information to answer confidently.

Question: {question}

Context so far:
{context}

Reply with exactly one word: SUFFICIENT or INSUFFICIENT."""


def format_context(docs: list[Document], max_chars: int = 6000) -> str:
    """Context engineering: dedupe by source+page and cap total size so the
    answer prompt stays within a predictable token budget instead of growing
    unbounded with each reasoning hop."""
    seen = set()
    blocks = []
    budget = max_chars

    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)

        label = doc.metadata.get("source", "unknown")
        if doc.metadata.get("page"):
            label = f"{label}:{doc.metadata['page']}"

        block = f"[{label}] {doc.page_content.strip()}"
        if len(block) > budget:
            block = block[:budget]
        blocks.append(block)
        budget -= len(block)
        if budget <= 0:
            break

    return "\n\n".join(blocks)
