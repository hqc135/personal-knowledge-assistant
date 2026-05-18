from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..config import DOCUMENTS_ROOT


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for extension in ("*.txt", "*.md"):
        yield from path.rglob(extension)


def load_documents(path: Path | None = None) -> list[dict]:
    target = path or DOCUMENTS_ROOT
    if not target.exists():
        raise FileNotFoundError(f"路径不存在: {target}")

    documents: list[dict] = []
    for file_path in _iter_files(target):
        content = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            documents.append({"content": content, "source": str(file_path)})

    if not documents:
        raise ValueError("未找到可加载的文档")

    return documents
