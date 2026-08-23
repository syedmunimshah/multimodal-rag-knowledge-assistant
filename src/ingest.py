import argparse
from pathlib import Path

from langchain_core.documents import Document

from src.loaders.image_loader import load_image
from src.loaders.pdf_loader import load_pdf
from src.vectorstore import get_vectorstore

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_file(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file_path)
    if suffix in IMAGE_EXTS:
        return load_image(file_path)
    return []


def ingest(path: Path) -> int:
    store = get_vectorstore()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())

    total = 0
    for file_path in files:
        docs = load_file(file_path)
        if not docs:
            continue
        store.add_documents(docs)
        total += len(docs)
        print(f"Indexed {len(docs)} chunk(s) from {file_path.name}")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the pgvector knowledge base.")
    parser.add_argument("path", type=Path, help="File or directory to ingest")
    args = parser.parse_args()

    total = ingest(args.path)
    print(f"\nDone. Indexed {total} chunk(s) total.")


if __name__ == "__main__":
    main()
