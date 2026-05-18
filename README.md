# personal-knowledge-assistant

最小化的 RAG API，使用 FastAPI 和 ChromaDB，实现文档加载、文本分块、向量存储与查询 Prompt 模板。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 启动服务

```bash
uvicorn app.main:app --reload
```

## 示例

加载文档：

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "doc-1", "text": "这是第一段测试文本。它用于演示 RAG。"},
      {"id": "doc-2", "text": "第二段文本用于检索。"}
    ],
    "chunk_size": 80,
    "chunk_overlap": 10
  }'
```

查询：

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "RAG 接口做了什么？",
    "top_k": 3
  }'
```
