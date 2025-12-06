# MedAgenticSystem v1 Overall Design（英文心血管指南 RAG）

## 0. 目标与范围

**v1 范围：**

* 仅使用以下目录中的**英文指南 PDF**：

  * `heart-disease-guidline-RAG/RAG-guidelines/*.pdf`
  * 显式排除：`heart-disease-guidline-RAG/RAG-guidelines/心内科指南/`
* 提供一个 HTTP API：

  * `GET /health`：健康检查
  * `POST /ask`：输入英文问题，输出：

    * 基于指南的回答（英文）
    * 清晰的引用（具体 guideline + section）
    * 医疗免责声明
* 架构上预留：

  * 未来添加中文指南
  * 未来添加 “recommendation-level / graph/spec-aware RAG”

**约束：**

* 对话 / 生成：使用 **OpenAI `gpt-4.1-mini`**
* 其他部分（解析、检索、向量库、rerank 全部）：

  * 使用 **开源 / 免费** 实现
* 使用 **uv** 管理 Python 环境
* 使用 `.env` 管理 OpenAI API Key
* 提供 **Docker / docker-compose** 一键运行

---

## 1. 技术栈总览

### 1.1 语言 & 环境

* Python 3.12/3.11（或 3.10+）看系统上有哪个优先用哪个。
* 包管理与虚拟环境：**uv**
* 类型与配置：

  * `pydantic`, `pydantic-settings`
  * `mypy`（可选）

### 1.2 核心依赖

| 模块         | 选择                                   | 说明                          |
| ---------- | ------------------------------------ | --------------------------- |
| Web API    | `fastapi`, `uvicorn[standard]`       | 轻量、类型友好                     |
| PDF 解析     | `pymupdf` (`fitz`)                   | 解析效果好，支持字体信息                |
| Token 估算   | `tiktoken`                           | 控制 chunk 长度                 |
| Sparse 检索  | `pytantivy`                          | Rust Tantivy 封装，本地 BM25，性能好 |
| 向量模型       | `FlagEmbedding` + `BAAI/bge-m3`      | 开源多语 embedding（未来支持中英）      |
| 向量库        | `qdrant-client` + Qdrant Docker      | 开源向量数据库                     |
| Reranker   | `BAAI/bge-reranker-v2-m3`            | 开源 cross-encoder reranker   |
| 配置 & env   | `pydantic-settings`, `python-dotenv` | 读取 `.env`                   |
| OpenAI 客户端 | `openai`                             | 调用 `gpt-4.1-mini`           |

（测试相关如 `pytest` 略）

---

## 2. 项目结构设计

根目录假设为 `medagenticsystem/`：

```text
medagenticsystem/
  overroll_design.md
  pyproject.toml
  uv.lock
  .env               # 不入库
  .env.example       # 示例
  .gitignore

  app/
    __init__.py
    config.py        # 全局配置（Pydantic Settings）

    models/
      __init__.py
      document.py    # DocumentMeta, Section, Paragraph
      chunk.py       # Chunk, ChunkMeta
      retrieval.py   # RetrievalRequest/Response, EvidenceBlock
      qa.py          # QARequest/QAResponse

    ingestion/
      __init__.py
      scan_guidelines.py   # 扫描英文 PDF 列表
      parse_pdfs.py        # PDF → 段落 + section 结构
      chunking.py          # 段落 → chunk
      index_bm25.py        # 建立 BM25 索引（Tantivy）
      index_vectors.py     # 建立向量索引（Qdrant）

    retrieval/
      __init__.py
      bm25_store.py        # BM25 查询封装
      vector_store.py      # 向量查询封装
      hybrid_retriever.py  # 混合检索 + 分数融合
      reranker.py          # cross-encoder rerank

    llm/
      __init__.py
      openai_client.py     # OpenAI client 封装
      prompts.py           # system / user prompt 模板
      answer_generator.py  # evidence → answer

    api/
      __init__.py
      main.py              # FastAPI app, 路由定义

    eval/
      __init__.py
      offline_eval.py      # 可选，检索/回答评估脚本

  data/
    raw/                   # 原始 PDF（挂载到容器）
    parsed/                # 结构化文档 JSONL
    chunks/                # chunk JSONL
    bm25_index/            # Tantivy 索引目录
    logs/                  # 日志

  docker/
    Dockerfile
    docker-compose.yml
```

---

## 3. 配置与密钥管理

### 3.1 `.env` 文件

在项目根目录创建 `.env`（不提交到 git）：

```env
# OpenAI 配置
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_CHAT=gpt-4.1-mini

# Qdrant 配置
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=        # 默认空即可（本地无鉴权）

# 索引与数据路径（容器内路径）
GUIDELINE_ROOT=/app/data/raw/english_guidelines
PARSED_DOCS_PATH=/app/data/parsed/english_docs.jsonl
CHUNKS_PATH=/app/data/chunks/english_chunks.jsonl
BM25_INDEX_DIR=/app/data/bm25_index

# 其他
LOG_LEVEL=INFO
```

`.env.example` 保留相同字段，但空值，用于示例。

### 3.2 配置类（`app/config.py`）

使用 `pydantic-settings`：

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model_chat: str = "gpt-4.1-mini"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    guideline_root: str = "data/raw/english_guidelines"
    parsed_docs_path: str = "data/parsed/english_docs.jsonl"
    chunks_path: str = "data/chunks/english_chunks.jsonl"
    bm25_index_dir: str = "data/bm25_index"

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

---

## 4. 使用 uv 的环境管理

### 4.1 初始化项目

在 `medagenticsystem` 目录外层：

```bash
uv init medagenticsystem
cd medagenticsystem
```

修改自动生成的 `pyproject.toml`，把 `name` 设置为 `"medagenticsystem"`，`dependencies` 留空或最小。

### 4.2 添加依赖

```bash
uv add fastapi "uvicorn[standard]" \
       pymupdf tiktoken \
       pydantic pydantic-settings python-dotenv \
       qdrant-client pytantivy \
       FlagEmbedding \
       openai \
       httpx
```

（如果你打算用 pytest/mypy 再 `uv add --dev pytest mypy`）

### 4.3 运行 / 开发

本地开发：

```bash
uv run uvicorn app.api.main:app --reload
```

---

## 5. 离线数据流水线设计（PDF → chunk → 索引）

### 5.1 扫描英文 PDF（`scan_guidelines.py`）

* 从 `settings.guideline_root` 读取所有 `.pdf` 文件（排除中文子目录）。
* 对每个文件，简单抽取：

  * `guideline_id`（用文件名简化，例如 `heidenreich-2022-hf`）
  * `title`（可从第一页解析或先用文件名）
  * `year`, `org`（后续可改成从封面/元数据提取）
* 保存到 `parsed/english_docs.jsonl` 的 meta 部分，或单独一个 meta 文件。

### 5.2 PDF 解析与段落切分（`parse_pdfs.py`）

使用 `pymupdf`：

* 遍历每个 PDF：

  * 对每一页：

    * 提取文本块（保持基本段落结构）
    * 尝试用规则识别章节标题：

      * 正则 `^\d+(\.\d+)*\s+[A-Z].+`
      * 或字体更大 / 加粗的行
  * 构建：

    * `DocumentMeta`：`guideline_id`, `title`, `year`, `org`, `lang="en"`
    * `Section`：`section_id`, `section_title`, `page_start`, `page_end`
    * `Paragraph`：`section_id`, `order`, `text`
* 输出 JSONL：每行一个 paragraph 记录，包含：

  * `guideline_id`, `section_id`, `section_title`, `page`, `text`, `lang="en"`

写入：`settings.parsed_docs_path`

### 5.3 Chunking（`chunking.py`）

目标：在不跨 section 的前提下，构造 200–400 token 的 chunk。

* 使用 `tiktoken` 估算 token 数。
* 算法（针对某个 section 的段落序列）：

  1. 逐段落追加到当前 buffer，直到 token 数 ≥ 300；
  2. 若超过上限 400，则回退最后一个段落；
  3. 写出一个 chunk，并让下一个 chunk 从上一个 chunk 的最后一个段落开始重叠。
* 对 chunk 计算/附带：

  * `chunk_id`：`{guideline_id}-{section_id}-{running_index}`
  * `guideline_id`, `section_id`, `section_title`, `page_start/end`
  * `lang="en"`
  * 简单规则抽取推荐等级信息（可选）：

    * 正则扫文本，识别 `Class I`, `Class IIa`, `Level A/B/C`，存入 `rec_class_list`, `loe_list`

输出：`settings.chunks_path`，JSONL，每行一个 chunk。

---

## 6. 索引构建

### 6.1 BM25 索引（`index_bm25.py`）

使用 `pytantivy`：

* 字段设计：

  * `chunk_id`（stored, indexed）
  * `guideline_id`（stored, indexed）
  * `section_title`（stored, indexed, boost）
  * `text`（stored, indexed）
  * `year`（stored, indexed）
* 创建索引目录：`settings.bm25_index_dir`
* 索引策略：

  * `section_title` 加权更高（详见 Tantivy schema 中的 `TEXT | STORED` 并设置权重）
* 运行命令（示例脚本）：

  ```bash
  uv run python -m app.ingestion.index_bm25
  ```

### 6.2 向量索引（`index_vectors.py`）

使用 `FlagEmbedding` + Qdrant：

* 模型：

  * embedding：`BAAI/bge-m3`
  * 通过 `FlagEmbedding.BGEM3Embedding` 加载（CPU 也能跑，只是慢一点）
* Qdrant：

  * 使用 docker 运行一个 Qdrant 实例（后面有 compose）
  * collection 名：`guideline_chunks_en`
  * 向量维度：`1024`
  * 度量：`cosine`
* 每条向量 payload：

  * `chunk_id`, `guideline_id`, `section_id`, `section_title`
  * `year`, `org`, `lang`, `page_start`, `page_end`
* 脚本入口：

  ```bash
  uv run python -m app.ingestion.index_vectors
  ```

---

## 7. 检索与重排（Retrieval Pipeline）

### 7.1 BM25 封装（`bm25_store.py`）

* 使用 `pytantivy` 的 search 接口：

  * 输入：`query_text`, `top_k`
  * 输出：列表 `[(chunk_id, score)]`
* 对 specific 字段加权（使用 Tantivy Query / MultiFieldQuery）

### 7.2 向量检索封装（`vector_store.py`）

* 使用 `qdrant-client`：

  * 输入：embedding 向量、`filter`（`lang="en"`）、`top_k`
  * 输出：`[(chunk_id, score)]`

### 7.3 混合检索（`hybrid_retriever.py`）

* 接口：

  ```python
  class HybridRetriever:
      def retrieve(self, query: str, top_k_sparse: int = 32, top_k_dense: int = 32,
                   top_k_final: int = 50) -> list[RetrievedChunk]:
          ...
  ```

* 步骤：

  1. 用 BGE embedding 计算 query 向量（复用 index 的 embedding 模型）
  2. BM25 检索 top-k_sparse
  3. 向量检索 top-k_dense
  4. 用 Reciprocal Rank Fusion（RRF）或简单加权：

     * 对每个 `chunk_id` 计算综合分数
  5. 选出 top_k_final 个候选

### 7.4 Reranker（`reranker.py`）

* 模型：`BAAI/bge-reranker-v2-m3`

* 使用 `FlagEmbedding.BGEQuantizedReranker` 或普通 `BGERReranker`。

* 接口：

  ```python
  class Reranker:
      def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 20) -> list[RetrievedChunk]:
          ...
  ```

* 对每个候选 chunk，构造 `[query, chunk.text]` 输入，得到一个分数，排序后取 top_k。

### 7.5 EvidenceBlock 组装（`retrieval` 层）

* 对 rerank 后的 top_k：

  * 按 `guideline_id`、`section_id` 分组
  * 合并相邻 chunk 的文本，构造 `EvidenceBlock`：

    ```python
    class EvidenceBlock(BaseModel):
        id: str
        doc_id: str
        guideline_title: str
        year: int
        section_id: str
        section_title: str
        page_range: tuple[int, int] | None
        text: str
        rec_class_list: list[str] = []
        loe_list: list[str] = []
    ```
  * 控制总 token 数在 2–3k 以内

---

## 8. 回答生成（Answer Generation）

### 8.1 OpenAI 客户端（`llm/openai_client.py`）

* 使用 `openai` 官方 Python SDK：

  * 从 `settings.openai_api_key` 和 `settings.openai_base_url` 初始化 client
  * 只在这里依赖 OpenAI，其余模块完全开源

### 8.2 Prompt 设计（`llm/prompts.py`）

* system prompt 主要内容：

  * 你是一个心血管指南助手
  * 只能使用提供的 evidence 文本回答
  * 每个结论后要附带 `[Doc X]` 引用
  * 尽量指出指南中的：人群、疾病阶段、推荐类别（Class of Recommendation）、证据等级（Level of Evidence）
  * 必须在结尾加 disclaimer

* user prompt：

  * 用户问题 + 整理好的 evidence block 文本

### 8.3 回答函数（`llm/answer_generator.py`）

接口设计：

```python
from app.models.retrieval import EvidenceBlock

def answer_question(question: str, evidences: list[EvidenceBlock]) -> str:
    ...
```

流程：

1. 拼接 content，形如：

   ```text
   Question:
   {question}

   Evidence:
   [Doc 1] 2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure
   Section 3.2 Diagnosis of HFrEF (p.35–37)
   Text:
   {evidence_1.text}

   [Doc 2] ...
   ```

2. 调用 `gpt-4.1-mini`：

   * `model=settings.openai_model_chat`

3. 对返回答案简单后处理（确保引用格式统一），返回给 API。

---

## 9. API 设计（FastAPI）

### 9.1 路由

`app/api/main.py`：

* `GET /health`

  * 返回简单 JSON：`{"status": "ok"}`

* `POST /ask`

  * 请求体（`QARequest`）：

    ```json
    {
      "question": "string"
    }
    ```
  * 响应体（`QAResponse`）：

    ```json
    {
      "answer": "string",
      "evidences": [
        {
          "id": "Doc 1",
          "guideline_id": "...",
          "guideline_title": "...",
          "year": 2022,
          "section_id": "3.2",
          "section_title": "...",
          "page_range": [35, 37]
        }
      ]
    }
    ```

后端内部流程：

1. `HybridRetriever.retrieve(question)` → candidates
2. `Reranker.rerank(question, candidates)` → top_k
3. 组装 `EvidenceBlock` 列表
4. 调 `answer_question(question, evidences)` → answer
5. 返回

---

## 10. Docker 化与运行

### 10.1 Dockerfile（`docker/Dockerfile`）

示例（可按需调整）：

```dockerfile
FROM python:3.11-slim

# 安装系统依赖（pymupdf / pytantivy 等需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# 复制项目文件
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# 暴露 FastAPI 端口
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> 注意：你需要在构建前确保 `uv.lock` 已经存在（本地跑过 `uv sync` 一次）。

### 10.2 docker-compose（`docker/docker-compose.yml`）

```yaml
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:v1.12.0
    container_name: qdrant
    ports:
      - "6333:6333"
    volumes:
      - ../data/qdrant:/qdrant/storage

  medagenticsystem-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: medagenticsystem-api
    env_file:
      - ../.env
    volumes:
      - ../data:/app/data       # 挂载数据目录（原始 PDF + 索引）
      - ../heart-disease-guidline-RAG/RAG-guidelines:/app/data/raw/english_guidelines
    depends_on:
      - qdrant
    ports:
      - "8000:8000"
```

### 10.3 初始化 & 运行步骤

1. **准备数据目录**

   ```bash
   mkdir -p data/parsed data/chunks data/bm25_index data/qdrant
   ```

2. **本地构建索引（推荐先在宿主机跑一次）**

   ```bash
   # 解析 + chunk
   uv run python -m app.ingestion.parse_pdfs
   uv run python -m app.ingestion.chunking

   # BM25 索引
   uv run python -m app.ingestion.index_bm25

   # 向量索引（要求 qdrant 已经运行，或用 docker compose 先启动 qdrant）
   docker compose -f docker/docker-compose.yml up -d qdrant
   uv run python -m app.ingestion.index_vectors
   ```

3. **启动完整服务**

   ```bash
   cd docker
   docker compose up -d
   ```

4. **测试**

   * 打开 `http://localhost:8000/docs`
   * 调用 `POST /ask`，填入英文问题，如：

     > What is the recommended first-line therapy for HFrEF according to recent guidelines?

---

## 11. 未来扩展挂载点

* **中文指南支持**：

  * 在 `parse_pdfs.py` / `chunking.py` 增加 `lang="zh"` 分支；
  * embedding 依旧用 `bge-m3`，Qdrant 增加 `lang` 过滤；
  * BM25 索引新增中文字段 / 分词。

* **spec-aware / graph RAG**：

  * 在 ingestion 阶段新增 `recommendations.jsonl`：

    * 结构化存储 “单条推荐”（class, LoE, population, intervention, outcome…）
  * 新增 `recommendation_retriever.py`，根据 query 直接检索推荐条目；
  * 生成阶段让 LLM 同时看结构化推荐 + 文本 evidence，输出表格/对比。
