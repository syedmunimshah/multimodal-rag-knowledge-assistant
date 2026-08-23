from langchain_core.documents import Document

from src.prompts import format_context


def test_format_context_dedupes_by_source_and_page():
    docs = [
        Document(page_content="A", metadata={"source": "doc.pdf", "page": 1}),
        Document(page_content="A duplicate", metadata={"source": "doc.pdf", "page": 1}),
        Document(page_content="B", metadata={"source": "doc.pdf", "page": 2}),
    ]

    context = format_context(docs)

    assert context.count("doc.pdf:1") == 1
    assert "doc.pdf:2" in context


def test_format_context_respects_char_budget():
    docs = [Document(page_content="x" * 100, metadata={"source": f"doc{i}.pdf", "page": i}) for i in range(10)]

    context = format_context(docs, max_chars=250)

    assert len(context) <= 250 + 20  # small allowance for source labels
