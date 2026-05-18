from __future__ import annotations

import os
from pathlib import Path

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
DOCUMENTS_ROOT = Path(os.getenv("DOCUMENTS_ROOT", "documents")).resolve()

COLLECTION_NAME = "knowledge_base"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
