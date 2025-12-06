先直接给一套**现在就能落地、以后好扩展到中英 + spec-aware 的设计方案**，然后把每个模块怎么选型说清楚。

---

## 0. 设计目标 & 原则

* 现阶段：

  * 只用 `RAG-guidelines` 里 **英文心血管指南（25 个 PDF）**
  * 面向医生/科研用户的 **严谨、可溯源** 临床问答（不搞“聊天型医生”）
* 未来扩展：

  1. 添加 `心内科指南` 里的中文 PDF，支持中英双语问答
  2. 在 RAG 之上，叠一层 **spec-aware / graph RAG**：把“推荐条目（recommendation）+证据等级+人群”结构化成图/表来查

**总体策略：**

> 现在就做：**混合检索（BM25 + 向量）+ 医疗 reranker + 强约束生成**
> 架构上留出两条将来的“支路”：
>
> * 一个是 `lang` 维度（中英文）
> * 一个是 `rec_unit` 维度（spec-aware 的“推荐条目库”）

---

## 1. 离线流水线：文档 → 分片 → 索引 → 元数据

### 1.1 文档采集

* 输入目录：`heart-disease-guidline-RAG/RAG-guidelines`
* 目前只扫：

  * 根目录下 25 个英文 PDF
* 工具 & 技术选型：

  * 用 Python 脚本 + `GitPython` 周期性 `git pull`
  * 每次扫描新文件 / 版本号，增量更新索引

### 1.2 PDF 解析 + 结构化切分

目标：别做“死分块”，要尽量保留指南的 **标题层次 + 推荐表格**。

**解析工具推荐：**

* `PyMuPDF` 或 `pdfplumber` 提取文本 + 基本布局
* 对章节结构（1. Introduction / 2. Methods / 3. Recommendations…）做简单的 heading 识别：

  * 正则抓 `^[0-9\.]+\s+[A-Z]` 之类
  * 再配合字体大小 / 粗体（PyMuPDF 可以拿到大致信息）

**切分策略（三层结构）：**

1. **Section 文档层（粗粒度）**

   * 每个 guideline → 若干个 section node（例如 “3.1 Diagnosis of HFrEF”）
   * 字段示例：

     ```json
     {
       "guideline_id": "heidenreich-2022-hf",
       "section_id": "3.1",
       "section_title": "Diagnosis of HFrEF",
       "text": "... full section text ...",
       "lang": "en",
       "year": 2022,
       "org": "AHA/ACC/HFSA"
     }
     ```

2. **Chunk 层（RAG 真正检索的单位）**

   * 在 section 内按段落 + 滑动窗口切：

     * 每块大约 **400–800 token**，上下文重叠 100–150 token
   * 每个 chunk 带上：

     * `guideline_id / section_id / section_title / page_range / lang / year / org / rec_class / level_of_evidence(若有)`
   * 让 chunk 尽量不要跨“推荐表格”和正文，避免语义断裂

3. **推荐条目层（为 spec-aware 做准备，先悄悄建起来）**

   * 识别类似：

     * `COR I, LOE A` / `Class IIb, Level of Evidence B-NR`
     * 或者 guideline 里 “Recommendations” 表格每一条

   * 用规则 + LLM 混合抽：

     * 规则：按“Class/Level”表格结构抓
     * LLM：对表格行做结构化抽取

   * 抽成类似：

     ```json
     {
       "rec_id": "heidenreich-2022-hf-3.2-rec-01",
       "guideline_id": "heidenreich-2022-hf",
       "section_id": "3.2",
       "text": "In patients with HFrEF, ACE inhibitors are recommended to reduce morbidity and mortality.",
       "condition": "heart failure with reduced ejection fraction",
       "population": "adult patients with HFrEF",
       "intervention": "ACE inhibitor",
       "class": "I",
       "loe": "A",
       "year": 2022,
       "lang": "en",
       "evidence_span_ids": ["chunk_123", "chunk_124"]
     }
     ```

   * 现在这层不用接入主服务，单独存库即可，将来启用 spec-aware 时直接挂上。

### 1.3 元数据存储

* 选型：

  * 关系型：**PostgreSQL**
* 存：

  * `guidelines` 表（id, title, org, year, url, lang…）
  * `sections` 表（section_id, guideline_id, title, page_start/end…）
  * `chunks` 表（chunk_id, section_id, lang, token_count…）
  * `recommendations` 表（即上面的 rec_units）

这层是“真相来源”，检索系统只是索引这些东西。

---

## 2. 检索层：混合检索 + rerank

### 2.1 向量检索

**关键点：一开始就选“多语”embedding，方便后面加中文，不用重索引。**

* 推荐：`BAAI/bge-m3` 或 `bge-large-en-v1.5` + 备用 `bge-large-zh`

  * 如果确定后面要中英一起检索，直接用 `bge-m3` 这种一体多语模型
* 向量库：

  * **Qdrant** / Milvus / Weaviate 都可以，偏向 Qdrant（部署简单）
* 向量维度：随模型（e.g. 1024）
* 索引粒度：

  * 主索引：以 **chunk 为单位**（每个 chunk 一个 embedding）
* 过滤：

  * 索引时就写 `lang=en`，检索时可以 `filter: {lang: "en"}`

### 2.2 稀疏检索（BM25）

* 选型：

  * **OpenSearch / Elasticsearch**
* 索引字段：

  * `title^3`, `section_title^2`, `text`, `org`, `year`
* 分词：

  * 英文直接 standard analyzer 即可

### 2.3 混合检索融合

查询时：

1. 用 BM25 取 top-50 `chunk_id`
2. 用向量检索取 top-50 `chunk_id`
3. 做 RRF（Reciprocal Rank Fusion） 或 简单线性加权

   * e.g. `score = 1/(k + rank_bm25) + 1/(k + rank_dense)`
4. 得到融合后的 top-50 chunk

### 2.4 医疗跨编码 reranker

为了“效果最好”，rerank 很重要。

* 模型选型：

  * 开源：`BAAI/bge-reranker-v2-m3`（多语 + 医疗效果也不错）
  * 商用：Cohere Rerank v3 / OpenAI re-ranking（看预算）
* 流程：

  * 输入：query + 50 个候选 chunk
  * 输出：重新排序后的得分，取 top-15 / top-20
* 为避免过度集中于一个 section，可以再做：

  * 简单 diversification：限制同一个 `section_id` 不超过 N 个 chunk

---

## 3. 生成层：Prompt & 安全策略

### 3.1 上下文拼装

从 top-20 chunks 里：

1. 按 `guideline_id` 分组，找出**最相关的 3–5 本指南**

2. 每本指南最多取 3–4 个 chunk，控制总 token ≈ 2–3k（看用的 LLM 上限）

3. 为每个 chunk 增加一个“引用头”：

   ```text
   [Doc 1] 2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure
   Section 3.2 Diagnosis of HFrEF, p.35–37
   Text: ...
   ```

4. 生成时要求模型在回答中标 `[Doc 1]` / `[Doc 2]` 这样的引用

### 3.2 Query 预处理（可选但推荐）

* 语言检测（fastText / langdetect）：

  * 如果检测到非英文：

    * 现在阶段：提示“目前版本仅支持英文问题”
    * 将来：路由到中文/双语 pipeline
* Query 重写（LLM 小工具）：

  * input: 原始 query
  * output: 结构化 query 信息：

    * 疾病（HFpEF, HFrEF, STEMI…）
    * 人群（pregnant women, elderly…）
    * 信息需求类型（diagnosis / treatment / risk factor / follow-up）
  * 重写出一个更“检索友好”的短 query：

    * 例如 input: “在 EF 40% 的心衰患者中，有哪些推荐的一线用药？”

      * 重写成：`guideline-based first-line pharmacologic therapy for HFrEF with LVEF ~40%`

### 3.3 LLM 生成 prompt 关键要点

对 LLM 的系统提示要非常硬核：

* 只允许使用上下文中的信息回答
* 必须明确指出引用来源
* 不知道就说不知道，不允许胡编
* 对每个推荐写明：

  * 是否 guideline 推荐（Class of Recommendation, Level of Evidence）
  * 是否与患者问题完全匹配（人群/疾病阶段/并发症）

示意 prompt 片段（伪英文）：

> You are a cardiology guideline assistant.
> Only use the provided guideline excerpts to answer.
> For each recommendation, if class/level of evidence is available in the text, explicitly state it.
> When you state a fact, attach citation like [Doc 2].
> If the guidelines do not cover the question, say so and suggest consulting a specialist rather than guessing.

输出结构建议：

1. 简短摘要（2–4 句）
2. 具体推荐列表（带 class/LoE）
3. 适用人群/排除条件
4. 明确说明“本系统只用于辅助学习和决策，不替代医生临床判断”的 disclaimer

---

## 4. 面向未来：中文 & 中英双语支持

### 4.1 多语 embedding 的好处

如果一开始就用 `bge-m3` 这类多语模型：

* 现在：只索引 `lang=en` 的 chunk
* 将来：加中文指南时，直接同一模型 encode，写 `lang=zh`
* 检索时：

  * 单语模式：`filter: lang="en"` 或 `lang="zh"`
  * 双语模式：`filter: lang in ["en","zh"]`

### 4.2 中文指南 pipeline 差异点

* PDF 解析：

  * 同样用 PyMuPDF；中文分词建议加一层 `jieba` / `pkuseg` 做分句
* section 识别：

  * 识别中文标题模式：`第X章`, `第X节`, `…指南`, 等
* 元数据：

  * `title_en`, `title_zh` 两个字段（有的中文指南有英文标题）

### 4.3 双语问答策略（将来）

* 用户问中文：

  * 直接在 `lang=zh` 检索
  * 可选：同时检索英文 `lang=en` 以补充信息
* 用户问英文：

  * 默认查英文
  * 如果英文指南没有覆盖某细分疾病，而中文有，可以备用查 `lang=zh`，但需要 LLM 翻译或提示“以下为中文指南信息，已翻译为英文”

---

## 5. spec-aware / graph RAG 的“挂载位”

你现在的系统只做 chunk-level RAG，但我们已经在第 1.2 建了 `recommendations` 这张表，这就是将来 spec-aware 的入口。

将来的演进可以是：

1. **推荐级检索分支**

   * 对 query 先检索 `recommendations` 表（BM25 + 向量）
   * 直接拿到若干条结构化的 Rec（带 class/LoE、人群、疾病）
2. **文本证据补充**

   * 对每条 Rec 的 `evidence_span_ids` 去 chunk 索引里补上下文
3. **生成阶段**

   * LLM 输入同时包含：

     * 结构化 recommendation 列表（json）
     * 对应的 chunk 文本
   * 让模型输出类似“证据表 + 总结”形式：

     * 每行：Condition / Population / Intervention / Class / LoE / Guideline / Year / Citation

这样可以平滑过渡：现在只是用 chunk，未来加上 rec-unit 检索，**而不用推翻现有架构**。

---

## 6. 模块与技术选型总表

粗简列一下模块 → 技术栈：

* **文档同步**

  * Python + GitPython，定时 `git pull`
* **PDF/CAJ 解析**

  * 英文/中文 PDF：PyMuPDF / pdfplumber
  * CAJ：先用外部工具转 PDF（未来中文阶段考虑）
* **结构化切分 & 推荐抽取**

  * Python 脚本
  * LLM 辅助抽取推荐表格（调用 OpenAI / 其他 LLM）
* **元数据 & rec-unit 存储**

  * PostgreSQL
* **向量检索**

  * Qdrant
  * Embedding 模型：`BAAI/bge-m3`（推荐）
* **稀疏检索**

  * OpenSearch / Elasticsearch（自托管）
* **Rerank**

  * 开源：`BAAI/bge-reranker-v2-m3`
  * 或云服务 rerank API
* **API 服务**

  * Python + FastAPI
  * 对外提供：`/ask` 接口，内部调用“query → hybrid retrieve → rerank → assemble → LLM”
* **Orchestration（可选）**

  * 你想快点搞完可以用 LangChain / LlamaIndex 做 glue，但建议数据层（Postgres/Qdrant/ES）自己掌握
* **监控 & 评估**

  * Prometheus + Grafana（基础监控）
  * 自建一批 Q&A 测试样本，用 LLM-as-judge 做离线评估

---

## 7. 大概资源/空间量级（给你一个感觉）

以 25 本英文指南估算：

* PDF 本身：几十到几百 MB
* 解析后文本：< 100 MB
* chunks：

  * 假设每本 200 页，每页 2 chunk → 25 × 200 × 2 ≈ 10k chunks
  * embedding：10k × 1024 维 × 4 bytes ≈ 40 MB + 索引额外开销，整体 < 200 MB
* BM25 索引：一般也在几百 MB 内

整体落地后，哪怕以后加上中文指南，**全系统（不含模型权重）几个 GB 就很宽裕**。


