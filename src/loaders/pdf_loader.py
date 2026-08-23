from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import settings


def load_pdf(path: Path) -> list[Document]:
    """Extract text per page, then split into overlapping chunks so the
    retriever can return a tightly-scoped passage instead of a whole page."""
    reader = PdfReader(str(path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        for chunk in splitter.split_text(text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": path.name,
                        "page": page_num,
                        "modality": "text",
                    },
                )
            )
    return docs
