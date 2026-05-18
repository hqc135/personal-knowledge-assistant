from __future__ import annotations

from uuid import uuid4

from ..config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能小于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunked: list[dict] = []
    for doc in documents:
        for index, chunk in enumerate(chunk_text(doc["content"])):
            chunked.append(
                {
                    "id": uuid4().hex,
                    "content": chunk,
                    "source": doc["source"],
                    "chunk_index": index,
                }
            )
    return chunked
