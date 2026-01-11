# MedAgenticSystem（英文指南 RAG）

## 1. 项目概览

- **目标**：仅使用 `RAG-guidelines/` 中的英文 PDF（排除 `心内科指南/`）构建心血管指南问答服务。
- **主要组件**：
  - PDF 解析：PyMuPDF（段落、章节、页码）
  - Chunk 切分：tiktoken 估算 200–400 token，抽取推荐等级/证据等级
  - 检索：Tantivy BM25 + Qdrant（FlagEmbedding `BAAI/bge-m3`）混合检索 + RRF
  - 精排：FlagEmbedding `BAAI/bge-reranker-v2-m3`
  - 生成：OpenAI `gpt-4.1-mini`
- **API**：FastAPI 暴露 `/health` 与 `/ask`
- **环境**：Python 3.12 + uv；Docker Compose 提供一键部署

## 2. 安装与准备

1. 克隆代码后，运行：
   ```bash
   UV_CACHE_DIR=.uv_cache uv sync
   cp .env.example .env
   ```
2. `.env` 关键变量（已改为全量路径）：
   ```
   OPENAI_API_KEY=...
   QDRANT_URL=http://qdrant:6333
   QDRANT_COLLECTION=guideline_chunks_en
   PARSED_DOCS_PATH=/app/data/parsed/test_english_docs.jsonl
   CHUNKS_PATH=/app/data/chunks/test_english_chunks_tk.jsonl
   ```
3. tiktoken 离线缓存：将 `cl100k_base.*` 置于 `tiktoken_cache/` 并执行
   ```bash
   export TIKTOKEN_CACHE_DIR=/home/ubuntu/df/medAgenticSystem/heart-disease-guidline-RAG/tiktoken_cache
   ```
   也可写入 `.env`。

## 3. 数据处理流水线

全部使用 uv 运行，必要时覆盖 `.env` 的路径：

```bash
# 1. 扫描英文指南（可选）
UV_CACHE_DIR=.uv_cache uv run python -m app.ingestion.scan_guidelines

# 2. PDF -> 段落
UV_CACHE_DIR=.uv_cache uv run python -m app.ingestion.parse_pdfs

# 3. 段落 -> Chunk（使用 tiktoken）
UV_CACHE_DIR=.uv_cache uv run python -m app.ingestion.chunking

# 4. 建 BM25 索引
UV_CACHE_DIR=.uv_cache uv run python -m app.ingestion.index_bm25

# 5. 启动 Qdrant 后构建向量索引
docker compose -f docker/docker-compose.yml up -d qdrant
UV_CACHE_DIR=.uv_cache uv run python -m app.ingestion.index_vectors
```

> 注：FlagEmbedding 在 CPU 上编码速度慢，建议在较长会话或 GPU 环境执行；若需分批处理，可修改 `CHUNKS_PATH` 指向样本文件。

## 4. 运行服务

### 本地开发
```bash
UV_CACHE_DIR=.uv_cache uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
访问 `http://localhost:8000/docs` 或：
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the recommended therapy for HFrEF?"}'
```

### Docker

```bash
cd docker
docker compose up -d    # 启动 Qdrant + API（需预先跑完索引）
```

## 5. 向量库运行方式

### 宿主机直接写入

在宿主机执行索引脚本时，需要将 `.env` 中写给容器的路径替换为本地路径。例如：

```bash
CHUNKS_PATH=data/chunks/test_english_chunks_tk.jsonl \
QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=guideline_chunks_en \
TIKTOKEN_CACHE_DIR=/home/ubuntu/df/medAgenticSystem/heart-disease-guidline-RAG/tiktoken_cache \
UV_CACHE_DIR=.uv_cache \
uv run python -m app.ingestion.index_vectors
```

该命令在宿主机读取 `data/chunks/...`，并直接写入本地 Qdrant 服务（在此之前先 `docker compose -f docker/docker-compose.yml up -d qdrant`）。

### 容器内执行

如果希望沿用 `.env` 内 `/app/data/...` 路径，可在启动 Docker Compose 后进入 API 容器执行：

```bash
cd docker
docker compose up -d      # 启动 qdrant + medagenticsystem-api
docker compose exec medagenticsystem-api uv run python -m app.ingestion.index_vectors
```

容器中 `/app/data` 已挂载 `../data`，因此索引脚本会读取 `.env` 中的 `/app/data/...` 路径并写入 Qdrant。

同理，构建 BM25 索引时也可以：

```bash
cd docker
docker compose up -d medagenticsystem-api
docker compose exec medagenticsystem-api uv run python -m app.ingestion.index_bm25
```

若要在宿主机直接执行（绕过容器路径），则使用本地路径覆盖 `CHUNKS_PATH`：

```bash
CHUNKS_PATH=data/chunks/test_english_chunks_tk.jsonl \
UV_CACHE_DIR=.uv_cache \
uv run python -m app.ingestion.index_bm25
```

> 提示：`uv run python -m app.ingestion.index_vectors` 在 CPU 上运行较慢，可在 `docker/` 目录内执行：
> ```bash
> docker compose exec -T medagenticsystem-api \
>   sh -c 'uv run python -m app.ingestion.index_vectors \
>   >> /tmp/index_vectors.log 2>&1 &'
> docker compose exec medagenticsystem-api tail -f /tmp/index_vectors.log
> ```
> 这样向量索引会在容器内后台运行，日志写入 `/tmp/index_vectors.log`。执行过程中，日志开头通常只有 collection 重建与模型加载信息；随着批次 upsert，会持续打印 `PUT ... points`，最终出现 “Indexed … chunks into Qdrant collection …” 表示完成。如果日志突然停止且未出现成功信息，说明任务被中断。

## 6. 检索与排序策略

1. **BM25**：Tantivy 搜索 top 32（字段：text、section_title、guideline_title），输出 `sparse_score`。
2. **向量检索**：FlagEmbedding `BAAI/bge-m3` 生成查询向量，Qdrant top 32，得到 `dense_score`。
3. **融合**：Reciprocal Rank Fusion（RRF）合并两路，得到 `fused_score`。
4. **精排**：FlagEmbedding `BAAI/bge-reranker-v2-m3` 对融合候选做 cross-encoder rerank，取前 10。
5. **证据块**：按 `guideline_id + section_id` 合并相邻 chunk，使用 tiktoken 控制总 token ≤ 3000，保留页码/推荐等级。
6. **生成**：OpenAI `gpt-4.1-mini` 接收问题 + evidence，输出答案并附加免责声明。

## 7. 已知限制

- **PDF 表格/图形**：当前解析仅提取线性文本，表格结构/图片不会被识别；若需此信息需额外 OCR 或手动标注。
- **推荐等级抽取**：使用简单正则（Class I/II、Level A/B/C），非标准描述可能漏检或误判。
- **向量构建耗时**：FlagEmbedding 在 CPU 上编码大量 chunk 较慢，建议使用 GPU 或分批运行。
- **Qdrant 版本**：默认容器为 1.12.0，如需与客户端版本一致可在 `docker-compose.yml` 中调整镜像。
- **OpenAI 依赖**：无离线 LLM；若 OpenAI API 不可用会返回 500。
- **tiktoken 回退**：如缓存缺失则会提示操作员选择继续或中断；使用 `ALLOW_TIKTOKEN_FALLBACK=true` 可自动允许空格近似。
- **向量库分区**：`QDRANT_COLLECTION` 每次可设不同名字实现逻辑分区；Qdrant payload 中可附加 `guideline_id/section_title` 等元信息，用于备注或过滤。

## 8. 常用命令

```bash
# 运行测试（若后续添加 pytest）
UV_CACHE_DIR=.uv_cache uv run pytest

# 快速检查 Qdrant 集合
curl http://localhost:6333/collections/guideline_chunks_sample40
```

## 9. 备注

- 所有密钥（如 `.env`）均被 `.gitignore` 排除，实际部署时请自行维护。
- 需要更多调试信息，可参考 `tmp.txt` 中记录的需求与问题列表。
