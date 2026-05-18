from __future__ import annotations

import hashlib
from typing import Any, List, Optional
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROMPT_TEMPLATE = """你是一个知识助理。请根据下面的内容回答问题。
内容:
{context}

问题: {question}
回答:"""


class HashEmbeddingFunction:
    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index % len(digest)] / 255 for index in range(self.dimension)]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        return []
    step = max(1, chunk_size - chunk_overlap)
    chunks: List[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


class DocumentInput(BaseModel):
    id: Optional[str] = None
    text: str = Field(min_length=1)


class LoadRequest(BaseModel):
    documents: List[DocumentInput] = Field(min_length=1)
    chunk_size: int = Field(default=500, ge=50, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)


class LoadResponse(BaseModel):
    indexed_chunks: int
    source_documents: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class Match(BaseModel):
    id: str
    document: str
    metadata: Optional[dict[str, Any]] = None
    distance: Optional[float] = None


class QueryResponse(BaseModel):
    prompt: str
    context: str
    matches: List[Match]


app = FastAPI(title="Personal Knowledge Assistant")
embedding_function = HashEmbeddingFunction()
client = chromadb.Client(Settings(is_persistent=False))
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function,
)


@app.post("/documents", response_model=LoadResponse)
def load_documents(request: LoadRequest) -> LoadResponse:
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size")

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict[str, Any]] = []

    for doc in request.documents:
        doc_id = doc.id or f"doc-{uuid4().hex}"
        chunks = chunk_text(doc.text, request.chunk_size, request.chunk_overlap)
        for index, chunk in enumerate(chunks):
            ids.append(f"{doc_id}-{index}")
            documents.append(chunk)
            metadatas.append({"source_id": doc_id, "chunk_index": index})

    if not documents:
        raise HTTPException(status_code=400, detail="No content to index")

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return LoadResponse(indexed_chunks=len(documents), source_documents=len(request.documents))


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents loaded")

    results = collection.query(
        query_texts=[request.question],
        n_results=request.top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]

    matches: List[Match] = []
    for index, document in enumerate(documents):
        matches.append(
            Match(
                id=ids[index] if index < len(ids) else "",
                document=document,
                metadata=metadatas[index] if index < len(metadatas) else None,
                distance=distances[index] if index < len(distances) else None,
            )
        )

    context = "\n\n".join(documents)
    prompt = PROMPT_TEMPLATE.format(context=context or "无匹配内容。", question=request.question)
    return QueryResponse(prompt=prompt, context=context, matches=matches)
