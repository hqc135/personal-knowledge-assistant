from __future__ import annotations

import hashlib

from chromadb.utils import embedding_functions

from ..config import EMBEDDING_BACKEND, EMBEDDING_DIM, EMBEDDING_MODEL


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


def build_embedding_function(
) -> HashEmbeddingFunction | embedding_functions.SentenceTransformerEmbeddingFunction:
    if EMBEDDING_BACKEND == "hash":
        return HashEmbeddingFunction()
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
