import base64
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.config import settings

CAPTION_PROMPT = (
    "Describe this image in factual detail: any text/labels visible, "
    "charts or diagrams, objects, and what it likely documents. "
    "This description will be embedded and searched over, so be specific "
    "and avoid vague adjectives."
)


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def load_image(path: Path) -> list[Document]:
    """Multimodal ingestion: a vision-capable chat model captions the image,
    and the caption text is what actually gets embedded and indexed. This
    keeps retrieval on a single text-embedding space for both modalities."""
    vision_model = ChatOpenAI(model=settings.vision_model, api_key=settings.openai_api_key)

    image_b64 = _encode_image(path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": CAPTION_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ]
    )
    caption = vision_model.invoke([message]).content

    return [
        Document(
            page_content=caption,
            metadata={
                "source": path.name,
                "modality": "image",
                "image_path": str(path),
            },
        )
    ]
