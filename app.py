from __future__ import annotations

from pathlib import Path
from typing import Iterable
from uuid import uuid4
import hashlib
import os

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Personal Knowledge Assistant")

RAG_PROMPT = """请使用以下上下文回答问题。
上下文:
{context}

问题: {question}
答案:
"""

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = "knowledge_base"
EMBEDDING_DIM = 128

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


class IngestRequest(BaseModel):
    path: str = Field(..., description="文件或目录路径")


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class QueryRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    top_k: int = Field(3, ge=1, le=20, description="返回的相似片段数量")


class QueryResponse(BaseModel):
    answer: str
    prompt: str
    sources: list[str]


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for extension in ("*.txt", "*.md"):
        yield from path.rglob(extension)


def load_documents(input_path: str) -> list[dict]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {input_path}")

    documents: list[dict] = []
    for file_path in _iter_files(path):
        content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if content:
            documents.append({"content": content, "source": str(file_path)})

    if not documents:
        raise ValueError("未找到可加载的文档")

    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    step = max(chunk_size - overlap, 1)
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


def embed_text(text: str, dimension: int = EMBEDDING_DIM) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [(byte - 128) / 128 for byte in digest]
    return [values[index % len(values)] for index in range(dimension)]


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [embed_text(text) for text in texts]


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    try:
        documents = load_documents(request.path)
        chunked = chunk_documents(documents)
        embeddings = embed_texts([chunk["content"] for chunk in chunked])
        _collection.add(
            ids=[chunk["id"] for chunk in chunked],
            documents=[chunk["content"] for chunk in chunked],
            embeddings=embeddings,
            metadatas=[
                {"source": chunk["source"], "chunk_index": chunk["chunk_index"]}
                for chunk in chunked
            ],
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(documents=len(documents), chunks=len(chunked))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if _collection.count() == 0:
        raise HTTPException(status_code=400, detail="向量库为空，请先调用 /ingest")

    query_embedding = embed_text(request.query)
    result = _collection.query(
        query_embeddings=[query_embedding],
        n_results=request.top_k,
        include=["documents", "metadatas"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    sources = [metadata.get("source", "") for metadata in metadatas]
    context = "\n\n".join(documents)
    prompt = RAG_PROMPT.format(context=context, question=request.query)

    return QueryResponse(
        answer=prompt,
        prompt=prompt,
        sources=sorted({source for source in sources if source}),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
