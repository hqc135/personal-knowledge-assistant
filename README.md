# personal-knowledge-assistant

一个最简单的 RAG 接口示例，使用 FastAPI + ChromaDB，支持文档加载、文本分块、入库与查询。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
uvicorn app:app --reload --port 8000
```

## 接口示例

### 写入文档

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "./documents"}'
```

`path` 支持文件或目录，目录下会自动读取 `.txt` 和 `.md` 文件。

### 查询

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "总结一下项目目标", "top_k": 3}'
```

查询结果会返回 `prompt`（带模板的上下文）以及 `answer`（当前为模板内容）。
