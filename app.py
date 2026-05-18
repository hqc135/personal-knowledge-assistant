from __future__ import annotations

from pathlib import Path
from typing import Iterable
from uuid import uuid4
import hashlib
import os

import chromadb
from chromadb.utils import embedding_functions
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
DOCUMENTS_ROOT = Path(os.getenv("DOCUMENTS_ROOT", "documents")).resolve()
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

_client = chromadb.PersistentClient(path=CHROMA_PATH)
class HashEmbeddingFunction:
    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension or EMBEDDING_DIM

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            values.extend((byte - 127.5) / 127.5 for byte in digest)
            counter += 1
        return values[: self.dimension]


def build_embedding_function() -> object:
    if EMBEDDING_BACKEND == "hash":
        return HashEmbeddingFunction()
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


_embedding_function = build_embedding_function()
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embedding_function,
    metadata={"hnsw:space": "cosine"},
)


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


def load_documents(path: Path = DOCUMENTS_ROOT) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

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


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    try:
        documents = load_documents()
        chunked = chunk_documents(documents)
        _collection.add(
            ids=[chunk["id"] for chunk in chunked],
            documents=[chunk["content"] for chunk in chunked],
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

    result = _collection.query(
        query_texts=[request.query],
        n_results=request.top_k,
        include=["documents", "metadatas"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    sources = [metadata.get("source", "") for metadata in metadatas]
    context = "\n\n".join(documents)
    prompt = RAG_PROMPT.format(context=context, question=request.query)

    answer = f"相关内容：{' '.join(documents)}" if documents else "未找到相关内容"

    return QueryResponse(
        answer=answer,
        prompt=prompt,
        sources=sorted({source for source in sources if source}),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
